//! In-process inference with typed, owned CPU/GPU tensors and explicit placement.
//! Input allocations are retained until REQUEST_RELEASE_ALL, not merely FINAL.
use crate::{DataType, ResponseAllocator, Server};
use anyhow::Result;
use std::{
    collections::HashMap,
    ffi::{CStr, CString, c_char, c_void},
    ptr,
    sync::{Arc, Mutex, mpsc},
};
use triton_ng_sys as sys;

const CPU: u32 = sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
const GPU: u32 = sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_GPU;
const FP32: DataType = DataType::Fp32;
const FP16: DataType = DataType::Fp16;
const INT64: DataType = DataType::Int64;

#[derive(Clone, Copy)]
enum FloatEncoding {
    Fp16,
    Fp32,
}

impl FloatEncoding {
    fn width(self) -> usize {
        match self {
            Self::Fp16 => 2,
            Self::Fp32 => 4,
        }
    }

    fn decode(self, data: &[u8]) -> Vec<f32> {
        match self {
            Self::Fp32 => data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&bytes| f32::from_le_bytes(bytes))
                .collect(),
            Self::Fp16 => data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&bytes| half::f16::from_bits(u16::from_le_bytes(bytes)).to_f32())
                .collect(),
        }
    }
}

fn checked(error: *mut sys::TRITONSERVER_Error) -> Result<()> {
    if error.is_null() {
        return Ok(());
    }
    // Caller transfers ownership of errors returned by API calls, unlike
    // InferenceResponseError, whose error is borrowed from its response.
    let text = unsafe { CStr::from_ptr(sys::TRITONSERVER_ErrorMessage(error)) }
        .to_string_lossy()
        .into_owned();
    unsafe { sys::TRITONSERVER_ErrorDelete(error) };
    anyhow::bail!("{text}")
}

fn error(message: &str) -> *mut sys::TRITONSERVER_Error {
    let text = CString::new(message.replace('\0', "?")).unwrap_or_default();
    unsafe {
        sys::TRITONSERVER_ErrorNew(
            sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INTERNAL,
            text.as_ptr(),
        )
    }
}

// Existing explicitly linked mode remains supported.
#[cfg(not(feature = "cuda-dynamic"))]
unsafe extern "C" {
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaMalloc(pointer: *mut *mut c_void, bytes: usize) -> i32;
    fn cudaFree(pointer: *mut c_void) -> i32;
}
#[cfg(feature = "cuda-dynamic")]
type SetDevice = unsafe extern "C" fn(i32) -> i32;
#[cfg(feature = "cuda-dynamic")]
type Malloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
#[cfg(feature = "cuda-dynamic")]
type Free = unsafe extern "C" fn(*mut c_void) -> i32;

struct Cuda {
    device: i32,
    #[cfg(feature = "cuda-dynamic")]
    set_device: SetDevice,
    #[cfg(feature = "cuda-dynamic")]
    malloc: Malloc,
    #[cfg(feature = "cuda-dynamic")]
    free: Free,
    // Buffers retain their pool, which retains Cuda and this library. The last
    // buffer is returned/freed before the runtime can be unloaded.
    #[cfg(feature = "cuda-dynamic")]
    _library: libloading::Library,
}
impl Cuda {
    fn new(device: i32) -> Result<Self> {
        (device >= 0)
            .then_some(())
            .ok_or_else(|| anyhow::anyhow!("negative GPU ID"))?;
        #[cfg(feature = "cuda-dynamic")]
        let cuda = Self::load_runtime(device)?;
        #[cfg(not(feature = "cuda-dynamic"))]
        let cuda = Self { device };
        cuda.select()?;
        Ok(cuda)
    }
    #[cfg(feature = "cuda-dynamic")]
    fn load_runtime(device: i32) -> Result<Self> {
        const NAMES: [&str; 4] = [
            "libcudart.so",
            "libcudart.so.13",
            "libcudart.so.12",
            "libcudart.so.11.0",
        ];
        // Resolve symbols once per client, never on the decoder's hot path.
        NAMES
            .into_iter()
            .find_map(|name| {
                // The library names and function ABIs are fixed SDK constants.
                let library = unsafe { libloading::Library::new(name) }.ok()?;
                Some(
                    (|| -> Result<Self> {
                        Ok(unsafe {
                            Self {
                                device,
                                set_device: *library.get::<SetDevice>(b"cudaSetDevice\0")?,
                                malloc: *library.get::<Malloc>(b"cudaMalloc\0")?,
                                free: *library.get::<Free>(b"cudaFree\0")?,
                                _library: library,
                            }
                        })
                    })()
                    .map_err(|error| anyhow::anyhow!("loading CUDA runtime {name}: {error:#}")),
                )
            })
            .unwrap_or_else(|| {
                Err(anyhow::anyhow!(
                    "CUDA runtime unavailable; tried {}",
                    NAMES.join(", ")
                ))
            })
    }

    fn select(&self) -> Result<()> {
        #[cfg(feature = "cuda-dynamic")]
        let rc = unsafe { (self.set_device)(self.device) };
        #[cfg(not(feature = "cuda-dynamic"))]
        let rc = unsafe { cudaSetDevice(self.device) };
        (rc == 0)
            .then_some(())
            .ok_or_else(|| anyhow::anyhow!("cudaSetDevice({}) failed: {rc}", self.device))?;
        Ok(())
    }
    fn allocate(&self, bytes: usize) -> Result<usize> {
        self.select()?;
        let mut pointer = ptr::null_mut();
        #[cfg(feature = "cuda-dynamic")]
        let rc = unsafe { (self.malloc)(&mut pointer, bytes) };
        #[cfg(not(feature = "cuda-dynamic"))]
        let rc = unsafe { cudaMalloc(&mut pointer, bytes) };
        (rc == 0 && !pointer.is_null())
            .then_some(())
            .ok_or_else(|| anyhow::anyhow!("cudaMalloc({bytes}) failed: {rc}"))?;
        Ok(pointer as usize)
    }
    fn release(&self, pointer: usize) {
        if self.select().is_ok() {
            #[cfg(feature = "cuda-dynamic")]
            unsafe {
                (self.free)(pointer as *mut c_void);
            }
            #[cfg(not(feature = "cuda-dynamic"))]
            unsafe {
                cudaFree(pointer as *mut c_void);
            }
        }
    }
}
#[derive(Default)]
struct Idle {
    bins: HashMap<usize, Vec<usize>>,
    bytes: usize,
}
struct Pool {
    cuda: Cuda,
    idle: Mutex<Idle>,
    limit: usize,
}
impl Pool {
    fn get(self: &Arc<Self>, bytes: usize) -> Result<Arc<Buffer>> {
        let capacity = bytes
            .max(1)
            .checked_add(65535)
            .ok_or_else(|| anyhow::anyhow!("allocation overflow"))?
            / 65536
            * 65536;
        let pointer = {
            let mut idle = self
                .idle
                .lock()
                .map_err(|_| anyhow::anyhow!("GPU pool poisoned"))?;
            let pointer = idle.bins.get_mut(&capacity).and_then(Vec::pop);
            if pointer.is_some() {
                idle.bytes -= capacity;
            }
            pointer
        };
        let pointer = match pointer {
            Some(p) => p,
            None => match self.cuda.allocate(capacity) {
                Ok(pointer) => pointer,
                Err(_) => {
                    // Reclaim idle allocations before treating CUDA OOM as fatal.
                    let bins = {
                        let mut idle = self
                            .idle
                            .lock()
                            .map_err(|_| anyhow::anyhow!("GPU pool poisoned"))?;
                        idle.bytes = 0;
                        std::mem::take(&mut idle.bins)
                    };
                    for bin in bins.values() {
                        for &pointer in bin {
                            self.cuda.release(pointer);
                        }
                    }
                    self.cuda.allocate(capacity)?
                }
            },
        };
        Ok(Arc::new(Buffer::Gpu {
            pointer,
            capacity,
            bytes,
            pool: self.clone(),
        }))
    }
    fn put(&self, pointer: usize, capacity: usize) {
        if let Ok(mut idle) = self.idle.lock()
            && capacity <= self.limit.saturating_sub(idle.bytes)
        {
            idle.bins.entry(capacity).or_default().push(pointer);
            idle.bytes += capacity;
            return;
        }
        self.cuda.release(pointer);
    }
}
impl Drop for Pool {
    fn drop(&mut self) {
        if let Ok(idle) = self.idle.get_mut() {
            for bin in idle.bins.values() {
                for &pointer in bin {
                    self.cuda.release(pointer);
                }
            }
        }
    }
}
enum Buffer {
    Host {
        pointer: usize,
        bytes: usize,
    },
    Gpu {
        pointer: usize,
        capacity: usize,
        bytes: usize,
        pool: Arc<Pool>,
    },
}
impl Buffer {
    fn host(bytes: usize) -> Result<Arc<Self>> {
        let layout = std::alloc::Layout::from_size_align(bytes.max(1), 64)?;
        let pointer = unsafe { std::alloc::alloc_zeroed(layout) };
        (!pointer.is_null())
            .then_some(())
            .ok_or_else(|| anyhow::anyhow!("host allocation failed"))?;
        Ok(Arc::new(Self::Host {
            pointer: pointer as usize,
            bytes,
        }))
    }
    fn pointer(&self) -> *const c_void {
        match self {
            Self::Host { pointer, .. } => *pointer as *const c_void,
            Self::Gpu { pointer, .. } => *pointer as *const c_void,
        }
    }
    fn bytes(&self) -> usize {
        match self {
            Self::Host { bytes, .. } => *bytes,
            Self::Gpu { bytes, .. } => *bytes,
        }
    }
    fn memory(&self) -> (u32, i64) {
        match self {
            Self::Host { .. } => (CPU, 0),
            Self::Gpu { pool, .. } => (GPU, i64::from(pool.cuda.device)),
        }
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        match self {
            Self::Gpu {
                pointer,
                capacity,
                pool,
                ..
            } => pool.put(*pointer, *capacity),
            Self::Host { pointer, bytes } => {
                if let Ok(layout) = std::alloc::Layout::from_size_align((*bytes).max(1), 64) {
                    unsafe {
                        std::alloc::dealloc(*pointer as *mut u8, layout);
                    }
                }
            }
        }
    }
}
#[derive(Clone)]
pub struct Tensor {
    pub shape: Vec<i64>,
    pub dtype: DataType,
    buffer: Arc<Buffer>,
}
impl Tensor {
    pub fn host(shape: Vec<i64>, dtype: DataType, bytes: Vec<u8>) -> Result<Self> {
        let buffer = Buffer::host(bytes.len())?;
        // Unique allocation, not published to Triton yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.pointer().cast_mut().cast::<u8>(),
                bytes.len(),
            );
        }
        let tensor = Self {
            shape,
            dtype,
            buffer,
        };
        tensor.validate()?;
        Ok(tensor)
    }
    pub fn ids(shape: Vec<i64>, values: &[i64]) -> Result<Self> {
        Self::host(
            shape,
            INT64,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }
    fn validate(&self) -> Result<()> {
        let elements = self.shape.iter().try_fold(1usize, |n, &d| {
            (d > 0)
                .then_some(())
                .ok_or_else(|| anyhow::anyhow!("nonpositive tensor dimension"))?;
            n.checked_mul(d as usize)
                .ok_or_else(|| anyhow::anyhow!("tensor size overflow"))
        })?;
        let width = match self.dtype {
            FP16 => 2,
            FP32 => 4,
            INT64 => 8,
            _ => anyhow::bail!("unsupported tensor datatype"),
        };
        (elements.checked_mul(width) == Some(self.buffer.bytes()))
            .then_some(())
            .ok_or_else(|| anyhow::anyhow!("tensor shape/byte size mismatch"))?;
        Ok(())
    }
    pub fn floats(&self) -> Result<Vec<f32>> {
        let (data, encoding) = self.float_bytes()?;
        Ok(encoding.decode(data))
    }
    pub fn integers(&self) -> Result<Vec<i64>> {
        let Buffer::Host { pointer, bytes } = &*self.buffer else {
            anyhow::bail!("refusing to read GPU data on CPU");
        };
        (self.dtype == INT64)
            .then_some(())
            .ok_or_else(|| anyhow::anyhow!("expected INT64 output"))?;
        let data = unsafe { std::slice::from_raw_parts(*pointer as *const u8, *bytes) };
        Ok(data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&bytes| i64::from_le_bytes(bytes))
            .collect())
    }
    /// Decode directly into rows without an intermediate flat FP32 copy.
    pub fn float_rows(&self, width: usize) -> Result<Vec<Vec<f32>>> {
        let (data, encoding) = self.float_bytes()?;
        let row_bytes = width
            .checked_mul(encoding.width())
            .filter(|&n| n > 0)
            .ok_or_else(|| anyhow::anyhow!("invalid floating point row width"))?;
        (data.len().is_multiple_of(row_bytes))
            .then_some(())
            .ok_or_else(|| anyhow::anyhow!("incomplete floating point row"))?;
        Ok(data
            .chunks_exact(row_bytes)
            .map(|row| encoding.decode(row))
            .collect())
    }
    fn float_bytes(&self) -> Result<(&[u8], FloatEncoding)> {
        let Buffer::Host { pointer, bytes } = &*self.buffer else {
            anyhow::bail!("refusing to read GPU data on CPU");
        };
        let encoding = match self.dtype {
            FP32 => FloatEncoding::Fp32,
            FP16 => FloatEncoding::Fp16,
            _ => anyhow::bail!("expected floating point output"),
        };
        // Host outputs are published only after response completion. Inputs are
        // immutable after construction; the internal model only reads them.
        let data = unsafe { std::slice::from_raw_parts(*pointer as *const u8, *bytes) };
        Ok((data, encoding))
    }
    pub fn require_gpu(&self, device: i32) -> Result<()> {
        (self.buffer.memory() == (GPU, i64::from(device)))
            .then_some(())
            .ok_or_else(|| anyhow::anyhow!("state/cache output is not on configured GPU"))?;
        Ok(())
    }
}
struct RequestInputs {
    _tensors: Vec<Tensor>,
    _allocator: Arc<ResponseAllocator>,
}
struct Context {
    pool: Arc<Pool>,
    host_outputs: Arc<[CString]>,
    _allocator: Arc<ResponseAllocator>,
    tx: mpsc::Sender<Result<HashMap<String, Tensor>>>,
    result: Option<Result<HashMap<String, Tensor>>>,
}
// Only the inference thread owns these handles. Triton invokes allocation and
// completion callbacks; allocations themselves use a synchronized CUDA pool.
unsafe extern "C" fn allocate(
    _: *mut sys::TRITONSERVER_ResponseAllocator,
    name: *const c_char,
    bytes: usize,
    _: u32,
    _: i64,
    user: *mut c_void,
    buffer: *mut *mut c_void,
    tag: *mut *mut c_void,
    actual: *mut u32,
    device: *mut i64,
) -> *mut sys::TRITONSERVER_Error {
    let ctx = unsafe { &*user.cast::<Context>() };
    let host = host_output(ctx, name);
    let allocation = if host {
        Buffer::host(bytes)
    } else {
        ctx.pool.get(bytes)
    };
    match allocation {
        Ok(value) => {
            let (kind, id) = value.memory();
            unsafe {
                *buffer = value.pointer().cast_mut();
                *actual = kind;
                *device = id;
                *tag = Box::into_raw(Box::new(value)).cast();
            }
            ptr::null_mut()
        }
        Err(e) => error(&format!("output allocation: {e:#}")),
    }
}
fn host_output(context: &Context, name: *const c_char) -> bool {
    !name.is_null()
        && context
            .host_outputs
            .iter()
            .any(|output| output.as_c_str() == unsafe { CStr::from_ptr(name) })
}
unsafe extern "C" fn query(
    _: *mut sys::TRITONSERVER_ResponseAllocator,
    user: *mut c_void,
    name: *const c_char,
    _: *mut usize,
    memory: *mut u32,
    device: *mut i64,
) -> *mut sys::TRITONSERVER_Error {
    let ctx = unsafe { &*user.cast::<Context>() };
    unsafe {
        *memory = if host_output(ctx, name) { CPU } else { GPU };
        *device = if *memory == GPU {
            i64::from(ctx.pool.cuda.device)
        } else {
            0
        };
    }
    ptr::null_mut()
}
unsafe extern "C" fn release_buffer(
    _: *mut sys::TRITONSERVER_ResponseAllocator,
    _: *mut c_void,
    tag: *mut c_void,
    _: usize,
    _: u32,
    _: i64,
) -> *mut sys::TRITONSERVER_Error {
    if !tag.is_null() {
        unsafe {
            drop(Box::from_raw(tag.cast::<Arc<Buffer>>()));
        }
    }
    ptr::null_mut()
}
unsafe extern "C" fn release_request(
    request: *mut sys::TRITONSERVER_InferenceRequest,
    flags: u32,
    user: *mut c_void,
) {
    if flags & sys::tritonserver_requestreleaseflag_enum_TRITONSERVER_REQUEST_RELEASE_ALL != 0 {
        unsafe {
            let _ = checked(sys::TRITONSERVER_InferenceRequestDelete(request));
            drop(Box::from_raw(user.cast::<RequestInputs>()));
        }
    }
}
unsafe fn response_tensors(
    response: *mut sys::TRITONSERVER_InferenceResponse,
) -> Result<HashMap<String, Tensor>> {
    let error = unsafe { sys::TRITONSERVER_InferenceResponseError(response) };
    if !error.is_null() {
        // This error is response-owned; never delete it separately.
        anyhow::bail!(
            "{}",
            unsafe { CStr::from_ptr(sys::TRITONSERVER_ErrorMessage(error)) }.to_string_lossy()
        );
    }
    let mut count = 0;
    unsafe {
        checked(sys::TRITONSERVER_InferenceResponseOutputCount(
            response, &mut count,
        ))?;
    }
    let mut result = HashMap::with_capacity(count as usize);
    for index in 0..count {
        let (mut name, mut dims, mut base, mut tag) =
            (ptr::null(), ptr::null(), ptr::null(), ptr::null_mut());
        let (mut dtype, mut rank, mut bytes, mut memory, mut device) = (0, 0, 0, 0, 0);
        unsafe {
            checked(sys::TRITONSERVER_InferenceResponseOutput(
                response,
                index,
                &mut name,
                &mut dtype,
                &mut dims,
                &mut rank,
                &mut base,
                &mut bytes,
                &mut memory,
                &mut device,
                &mut tag,
            ))?;
        }
        (!tag.is_null() && !name.is_null() && !dims.is_null() && rank > 0 && rank <= 8)
            .then_some(())
            .ok_or_else(|| anyhow::anyhow!("invalid internal tensor metadata"))?;
        let buffer = unsafe { &*tag.cast::<Arc<Buffer>>() }.clone();
        (base == buffer.pointer()
            && bytes == buffer.bytes()
            && (memory, device) == buffer.memory())
        .then_some(())
        .ok_or_else(|| anyhow::anyhow!("output allocation mismatch"))?;
        let tensor = Tensor {
            shape: unsafe { std::slice::from_raw_parts(dims, rank as usize) }.to_vec(),
            dtype: DataType::from_sys(dtype),
            buffer,
        };
        tensor.validate()?;
        result.insert(
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned(),
            tensor,
        );
    }
    Ok(result)
}
unsafe extern "C" fn complete(
    response: *mut sys::TRITONSERVER_InferenceResponse,
    flags: u32,
    user: *mut c_void,
) {
    let ctx = unsafe { &mut *user.cast::<Context>() };
    if !response.is_null() {
        let result = unsafe { response_tensors(response) };
        if ctx.result.is_some() {
            ctx.result = Some(Err(anyhow::anyhow!(
                "decoupled internal models are unsupported"
            )));
        } else {
            ctx.result = Some(result);
        }
        unsafe {
            let _ = checked(sys::TRITONSERVER_InferenceResponseDelete(response));
        }
    }
    if flags & sys::tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL != 0
    {
        let mut ctx = unsafe { Box::from_raw(user.cast::<Context>()) };
        let result = ctx
            .result
            .take()
            .unwrap_or_else(|| Err(anyhow::anyhow!("internal model returned no response")));
        let _ = ctx.tx.send(result);
    }
}

/// Immutable names for a repeatedly executed model. Tensor order must match
/// input order; only tensor data and shapes are supplied at inference time.
pub struct InferencePlan {
    model: CString,
    inputs: Vec<CString>,
    outputs: Vec<CString>,
}

impl InferencePlan {
    pub fn new(model: &str, inputs: Vec<String>, outputs: Vec<String>) -> Result<Self> {
        Ok(Self {
            model: CString::new(model)?,
            inputs: inputs
                .into_iter()
                .map(CString::new)
                .collect::<std::result::Result<_, _>>()?,
            outputs: outputs
                .into_iter()
                .map(CString::new)
                .collect::<std::result::Result<_, _>>()?,
        })
    }

    pub fn input_names(&self) -> impl ExactSizeIterator<Item = &CStr> {
        self.inputs.iter().map(CString::as_c_str)
    }

    pub fn output_name(&self, index: usize) -> Result<&str> {
        Ok(self
            .outputs
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("inference plan output index out of range"))?
            .to_str()?)
    }

    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }
}

/// Synchronous in-process inference retaining GPU outputs for reuse as inputs.
/// Owns its output allocator and bounded device pool. Output placement is explicit;
/// the default is GPU, with named host outputs supplied at construction.
pub struct DeviceInference {
    server: Server,
    host_outputs: Arc<[CString]>,
    pool: Arc<Pool>,
    allocator: Arc<ResponseAllocator>,
}
// Triton's server handle is thread-safe; it is borrowed from the enclosing model.
// Calls are serialized by the frontend engine mutex; callbacks may run elsewhere.
unsafe impl Send for DeviceInference {}
impl DeviceInference {
    pub fn new(
        server: Server,
        device: i32,
        host_outputs: &[&str],
        pool_limit: usize,
    ) -> Result<Self> {
        Ok(Self {
            server,
            host_outputs: host_outputs
                .iter()
                .map(|name| CString::new(*name))
                .collect::<std::result::Result<Vec<_>, _>>()?
                .into(),
            pool: Arc::new(Pool {
                cuda: Cuda::new(device)?,
                idle: Mutex::new(Idle::default()),
                limit: pool_limit,
            }),
            allocator: Arc::new(ResponseAllocator::with_callbacks(
                Some(allocate),
                Some(release_buffer),
                Some(query),
            )?),
        })
    }

    pub fn infer(
        &self,
        model: &str,
        inputs: Vec<(String, Tensor)>,
        outputs: &[String],
    ) -> Result<HashMap<String, Tensor>> {
        let (names, tensors) = inputs.into_iter().unzip();
        let plan = InferencePlan::new(model, names, outputs.to_vec())?;
        self.infer_plan(&plan, tensors)
    }

    pub fn infer_plan(
        &self,
        plan: &InferencePlan,
        inputs: Vec<Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        (inputs.len() == plan.input_names().len())
            .then_some(())
            .ok_or_else(|| anyhow::anyhow!("inference plan input count mismatch"))?;
        // All generated models use immutable, pinned version 1.
        let mut request = ptr::null_mut();
        unsafe {
            checked(sys::TRITONSERVER_InferenceRequestNew(
                &mut request,
                self.server.as_ptr(),
                plan.model.as_ptr(),
                1,
            ))?;
        }
        // Until submission this guard owns deletion, including early errors.
        struct Pending(*mut sys::TRITONSERVER_InferenceRequest);
        impl Drop for Pending {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    unsafe {
                        let _ = checked(sys::TRITONSERVER_InferenceRequestDelete(self.0));
                    }
                }
            }
        }
        let mut pending = Pending(request);
        for (name, tensor) in plan.input_names().zip(&inputs) {
            tensor.validate()?;
            let (memory, device) = tensor.buffer.memory();
            unsafe {
                checked(sys::TRITONSERVER_InferenceRequestAddInput(
                    request,
                    name.as_ptr(),
                    tensor.dtype.to_sys(),
                    tensor.shape.as_ptr(),
                    tensor.shape.len() as u64,
                ))?;
                checked(sys::TRITONSERVER_InferenceRequestAppendInputData(
                    request,
                    name.as_ptr(),
                    tensor.buffer.pointer(),
                    tensor.buffer.bytes(),
                    memory,
                    device,
                ))?;
            }
        }
        for name in &plan.outputs {
            unsafe {
                checked(sys::TRITONSERVER_InferenceRequestAddRequestedOutput(
                    request,
                    name.as_ptr(),
                ))?;
            }
        }
        let allocator = &self.allocator;
        let (tx, rx) = mpsc::channel();
        let ctx = Box::into_raw(Box::new(Context {
            pool: self.pool.clone(),
            host_outputs: self.host_outputs.clone(),
            _allocator: allocator.clone(),
            tx,
            result: None,
        }));
        let owned_inputs = Box::into_raw(Box::new(RequestInputs {
            _tensors: inputs,
            _allocator: allocator.clone(),
        }));
        let setup = unsafe {
            checked(sys::TRITONSERVER_InferenceRequestSetReleaseCallback(
                request,
                Some(release_request),
                owned_inputs.cast(),
            ))
            .and_then(|()| {
                checked(sys::TRITONSERVER_InferenceRequestSetResponseCallback(
                    request,
                    allocator.as_ptr(),
                    ctx.cast(),
                    Some(complete),
                    ctx.cast(),
                ))
            })
            .and_then(|()| {
                checked(sys::TRITONSERVER_ServerInferAsync(
                    self.server.as_ptr(),
                    request,
                    ptr::null_mut(),
                ))
            })
        };
        if let Err(e) = setup {
            // No ownership transfer on synchronous ServerInferAsync failure.
            unsafe {
                drop(Box::from_raw(ctx));
            }
            // Delete request before dropping its referenced allocations.
            drop(pending);
            unsafe {
                drop(Box::from_raw(owned_inputs));
            }
            return Err(e);
        }
        pending.0 = ptr::null_mut();
        // No premature timeout: a submitted request still owns input pointers.
        rx.recv()
            .map_err(|_| anyhow::anyhow!("internal inference callback channel closed"))?
    }
}
