use crate::error::TritonError;
use crate::model::Model;
use std::ffi::c_void;
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::sync::Arc;
use triton_ng_macros::triton_call;

pub(crate) type MemoryType = triton_ng_sys::TRITONSERVER_MemoryType;

pub(crate) const CPU: MemoryType =
    triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
const CPU_PINNED: MemoryType =
    triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU_PINNED;
const GPU: MemoryType = triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_GPU;

/// Triton's memory manager: buffers from the server's own pools. Batches get
/// it from [`crate::Context::memory`].
///
/// - [`MemoryManager::ram`]: plain CPU memory.
/// - [`MemoryManager::pinned`]: page-locked CPU memory from the pool sized by
///   `--pinned-memory-pool-byte-size`, copied to and from the GPU faster than
///   plain memory.
/// - [`MemoryManager::gpu`]: GPU memory from the pool sized by
///   `--cuda-memory-pool-byte-size` (64 MB per GPU by default), shared with
///   Triton itself. The pools do not grow.
///
/// Every buffer frees its memory on drop. A buffer must not outlive the
/// server: keep buffers in the state of the model or the instance, which is
/// dropped before the server stops. `MemoryManager` is `Send + Sync + Clone`.
///
/// # Examples
///
/// ```ignore
/// async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
///     let mut tokens = ctx.memory().pinned(requests.len() * 8)?;
///     tokens.copy_from_slice(&encode_tokens(&requests));
///     let state = ctx.memory().gpu(ctx.device_id(), requests.len() * 320 * 4)?;
///     // ... pass `tokens` as an input and `state` as an output of a BLS call ...
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct MemoryManager {
    ptr: *mut triton_ng_sys::TRITONBACKEND_MemoryManager,
}

// SAFETY: the pointer is the memory manager of the core, a single object for
// the whole process (TRITONBACKEND_BackendMemoryManager returns the same
// static for every backend). Allocate and Free are thread-safe: the GPU pool
// (cnmem) and the pinned pool lock a mutex, CPU memory is malloc/free.
unsafe impl Send for MemoryManager {}
unsafe impl Sync for MemoryManager {}

impl MemoryManager {
    pub(crate) fn of_model(model: &Model) -> Result<Self, TritonError> {
        let backend = triton_call!(triton_ng_sys::TRITONBACKEND_ModelBackend(
            model.as_ptr(),
            &mut _
        ))?;
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_BackendMemoryManager(
            backend,
            &mut _
        ))?;
        if ptr.is_null() {
            return Err(TritonError::from_message(
                "TRITONBACKEND_BackendMemoryManager returned null",
            ));
        }
        Ok(Self { ptr })
    }

    /// Allocates `len` bytes of CPU memory, filled with zeros.
    ///
    /// Fails when the memory cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut image = ctx.memory().ram(28 * 28 * 4)?;
    /// image.copy_from_slice(request.get_input("Input3")?.as_bytes()?);
    /// ```
    pub fn ram(&self, len: usize) -> Result<RamBuffer, TritonError> {
        Ok(RamBuffer(Allocation::zeroed(self, len, CPU, 0)?))
    }

    /// Allocates `len` bytes of pinned CPU memory, filled with zeros.
    ///
    /// Fails when the pinned pool has no free block of `len` bytes or is
    /// disabled (`--pinned-memory-pool-byte-size=0`), and on servers built
    /// without GPU support: there is no fallback to plain memory.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut tokens = ctx.memory().pinned(batch * 8)?;
    /// tokens.copy_from_slice(&token_bytes);
    /// ```
    pub fn pinned(&self, len: usize) -> Result<PinnedRamBuffer, TritonError> {
        Ok(PinnedRamBuffer(Allocation::zeroed(
            self, len, CPU_PINNED, 0,
        )?))
    }

    /// Allocates `len` bytes on GPU `device_id`. The memory is not
    /// initialized and cannot be written from the CPU: the buffer becomes a
    /// [`GpuBuffer`] once a model writes its output into it.
    ///
    /// Fails when the GPU pool has no free block of `len` bytes (Triton
    /// reports it as an INTERNAL error), when `device_id` is negative, and on
    /// servers built without GPU support.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let state = ctx.memory().gpu(ctx.device_id(), batch * 320 * 4)?;
    /// ```
    pub fn gpu(&self, device_id: i32, len: usize) -> Result<UninitGpuBuffer, TritonError> {
        if device_id < 0 {
            return Err(TritonError::from_message(format!(
                "GPU device id must not be negative, got {device_id}"
            )));
        }
        Ok(UninitGpuBuffer(Allocation::new(
            self,
            len,
            GPU,
            i64::from(device_id),
        )?))
    }
}

/// Memory from [`MemoryManager`], freed on drop. Zero-length allocations do
/// not reach Triton and hold a null pointer.
struct Allocation {
    manager: MemoryManager,
    ptr: *mut c_void,
    len: usize,
    memory_type: MemoryType,
    memory_type_id: i64,
}

// SAFETY: the allocation is plain memory owned by this value, and the memory
// manager is `Send + Sync`.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Allocation {
    fn new(
        manager: &MemoryManager,
        len: usize,
        memory_type: MemoryType,
        memory_type_id: i64,
    ) -> Result<Self, TritonError> {
        let ptr = if len == 0 {
            ptr::null_mut()
        } else {
            let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_MemoryManagerAllocate(
                manager.ptr,
                &mut _,
                memory_type,
                memory_type_id,
                len as u64,
            ))?;
            if ptr.is_null() {
                return Err(TritonError::from_message(
                    "TRITONBACKEND_MemoryManagerAllocate returned null",
                ));
            }
            ptr
        };
        Ok(Self {
            manager: manager.clone(),
            ptr,
            len,
            memory_type,
            memory_type_id,
        })
    }

    fn zeroed(
        manager: &MemoryManager,
        len: usize,
        memory_type: MemoryType,
        memory_type_id: i64,
    ) -> Result<Self, TritonError> {
        let allocation = Self::new(manager, len, memory_type, memory_type_id)?;
        if len > 0 {
            unsafe { ptr::write_bytes(allocation.ptr.cast::<u8>(), 0, len) };
        }
        Ok(allocation)
    }

    /// The bytes of a CPU-accessible allocation.
    fn bytes(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: CPU or pinned memory of `len` bytes, initialized by
        // `zeroed`, and borrowed from its only owner.
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len) }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: as in `bytes`, borrowed mutably from its only owner.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.cast::<u8>(), self.len) }
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if let Err(e) = triton_call!(triton_ng_sys::TRITONBACKEND_MemoryManagerFree(
            self.manager.ptr,
            self.ptr,
            self.memory_type,
            self.memory_type_id,
        )) {
            crate::log::error(&format!("failed to free a memory manager buffer: {e}"));
        }
    }
}

/// CPU memory from [`MemoryManager::ram`], readable and writable as `[u8]`.
pub struct RamBuffer(Allocation);

/// Pinned CPU memory from [`MemoryManager::pinned`], readable and writable
/// as `[u8]`.
pub struct PinnedRamBuffer(Allocation);

/// GPU memory a model has written its output into. It cannot be read on the
/// CPU: pass it as an input of the next BLS call, or turn it back into an
/// [`UninitGpuBuffer`] to receive another output.
pub struct GpuBuffer(Allocation);

/// GPU memory from [`MemoryManager::gpu`] whose content is not defined yet.
/// It can only receive the output of a BLS call.
pub struct UninitGpuBuffer(Allocation);

impl Deref for RamBuffer {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.0.bytes()
    }
}

impl DerefMut for RamBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.0.bytes_mut()
    }
}

impl Deref for PinnedRamBuffer {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.0.bytes()
    }
}

impl DerefMut for PinnedRamBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.0.bytes_mut()
    }
}

impl GpuBuffer {
    /// Size in bytes.
    pub fn len(&self) -> usize {
        self.0.len
    }

    /// Whether the buffer has no bytes.
    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }

    /// The GPU the memory is on.
    pub fn device_id(&self) -> i32 {
        self.0.memory_type_id as i32
    }

    /// Gives up the content, so the memory can receive another output.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let reused: UninitGpuBuffer = cached.into_uninit();
    /// ```
    pub fn into_uninit(self) -> UninitGpuBuffer {
        UninitGpuBuffer(self.0)
    }
}

impl UninitGpuBuffer {
    /// Size in bytes.
    pub fn len(&self) -> usize {
        self.0.len
    }

    /// Whether the buffer has no bytes.
    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }

    /// The GPU the memory is on.
    pub fn device_id(&self) -> i32 {
        self.0.memory_type_id as i32
    }
}

impl Allocation {
    fn raw(&self) -> RawBuffer {
        RawBuffer {
            ptr: self.ptr,
            len: self.len,
            memory_type: self.memory_type,
            memory_type_id: self.memory_type_id,
        }
    }
}

mod sealed {
    use super::MemoryType;
    use std::ffi::c_void;

    /// Memory Triton reads or writes: where it is, its address and size.
    pub struct RawBuffer {
        pub(crate) ptr: *mut c_void,
        pub(crate) len: usize,
        pub(crate) memory_type: MemoryType,
        pub(crate) memory_type_id: i64,
    }

    pub trait Sealed {
        fn raw(&self) -> RawBuffer;
    }
}

pub(crate) use sealed::{RawBuffer, Sealed};

/// Data of a BLS input: `Vec<u8>`, [`RamBuffer`], [`PinnedRamBuffer`],
/// [`GpuBuffer`], or an `Arc` of any of them to keep using the buffer, for
/// example in a cache, while Triton reads it.
///
/// The request owns the buffer until Triton releases the request, which can
/// be after the response has arrived: an `Arc` passed as an input is not
/// unique again before that.
///
/// # Examples
///
/// ```ignore
/// let state: Arc<GpuBuffer> = Arc::new(result.take_gpu("ho")?);
/// inference.add_input_with_buffer("hi", DataType::Fp32, &[1, batch, 320], Arc::clone(&state))?;
/// ```
pub trait InputBuffer: sealed::Sealed + Send + 'static {}

impl Sealed for Vec<u8> {
    fn raw(&self) -> RawBuffer {
        RawBuffer {
            ptr: self.as_ptr().cast_mut().cast(),
            len: self.len(),
            memory_type: CPU,
            memory_type_id: 0,
        }
    }
}

impl InputBuffer for Vec<u8> {}

impl Sealed for RamBuffer {
    fn raw(&self) -> RawBuffer {
        self.0.raw()
    }
}

impl InputBuffer for RamBuffer {}

impl Sealed for PinnedRamBuffer {
    fn raw(&self) -> RawBuffer {
        self.0.raw()
    }
}

impl InputBuffer for PinnedRamBuffer {}

impl Sealed for GpuBuffer {
    fn raw(&self) -> RawBuffer {
        self.0.raw()
    }
}

impl InputBuffer for GpuBuffer {}

impl<T: InputBuffer + Sync> Sealed for Arc<T> {
    fn raw(&self) -> RawBuffer {
        self.as_ref().raw()
    }
}

impl<T: InputBuffer + Sync> InputBuffer for Arc<T> {}

/// Memory a BLS output is written into, see
/// [`crate::InferenceRequest::add_requested_output_into`]. Its length must be
/// the exact size of the output.
pub enum OutputBuffer {
    Ram(RamBuffer),
    Pinned(PinnedRamBuffer),
    Gpu(UninitGpuBuffer),
}

impl From<RamBuffer> for OutputBuffer {
    fn from(buffer: RamBuffer) -> Self {
        Self::Ram(buffer)
    }
}

impl From<PinnedRamBuffer> for OutputBuffer {
    fn from(buffer: PinnedRamBuffer) -> Self {
        Self::Pinned(buffer)
    }
}

impl From<UninitGpuBuffer> for OutputBuffer {
    fn from(buffer: UninitGpuBuffer) -> Self {
        Self::Gpu(buffer)
    }
}

/// A buffer a model has written its output into, see
/// [`OutputBuffer::into_written`]. Type-erased so that dropping a result
/// does not reach Triton symbols in unit tests.
pub(crate) type WrittenBuffer = Box<dyn std::any::Any + Send + Sync>;

impl OutputBuffer {
    pub(crate) fn raw(&self) -> RawBuffer {
        match self {
            Self::Ram(buffer) => buffer.0.raw(),
            Self::Pinned(buffer) => buffer.0.raw(),
            Self::Gpu(buffer) => buffer.0.raw(),
        }
    }

    /// The buffer once a model has written its output into it:
    /// [`RamBuffer`], [`PinnedRamBuffer`] or [`GpuBuffer`].
    pub(crate) fn into_written(self) -> WrittenBuffer {
        match self {
            Self::Ram(buffer) => Box::new(buffer),
            Self::Pinned(buffer) => Box::new(buffer),
            Self::Gpu(buffer) => Box::new(GpuBuffer(buffer.0)),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn buffers_can_move_to_tasks() {
        fn assert_shared<T: Send + Sync + 'static>() {}
        assert_shared::<super::MemoryManager>();
        assert_shared::<super::RamBuffer>();
        assert_shared::<super::PinnedRamBuffer>();
        assert_shared::<super::GpuBuffer>();
        assert_shared::<super::UninitGpuBuffer>();
    }
}
