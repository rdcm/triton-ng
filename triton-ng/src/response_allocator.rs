use crate::TritonError;
use crate::inference_request::OutputBuffers;
use crate::{CPU, OutputBuffer};
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::ptr;
use std::sync::{Mutex, PoisonError};
use triton_ng_macros::triton_call;

/// What `alloc_fn` stores in `buffer_userp` for an output written into an
/// [`OutputBuffer`]: the buffer, taken out when the response is read. CPU
/// outputs from `malloc` have a null `buffer_userp`.
pub(crate) type OutputTag = Option<OutputBuffer>;

/// Buffers of the requested outputs of one request, passed to the allocator
/// callbacks as `response_allocator_userp`. Triton may call them from
/// several threads, hence the mutex.
pub(crate) struct OutputPlan {
    buffers: Mutex<OutputBuffers>,
}

impl OutputPlan {
    pub(crate) fn new(buffers: OutputBuffers) -> Self {
        Self {
            buffers: Mutex::new(buffers),
        }
    }

    /// Memory type and id of the buffer of output `name`, `None` for outputs
    /// without a buffer.
    fn placement(&self, name: &str) -> Option<(triton_ng_sys::TRITONSERVER_MemoryType, i64)> {
        let buffers = self.buffers.lock().unwrap_or_else(PoisonError::into_inner);
        let raw = buffers.get(name)?.as_ref()?.raw();
        Some((raw.memory_type, raw.memory_type_id))
    }

    /// Hands over the buffer of output `name` for `byte_size` bytes,
    /// `Ok(None)` for outputs without a buffer. Fails when the buffer was
    /// already handed over or has another size; the buffer is dropped then.
    fn take(&self, name: &str, byte_size: usize) -> Result<Option<OutputBuffer>, String> {
        let mut buffers = self.buffers.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(slot) = buffers.get_mut(name) else {
            return Ok(None);
        };
        let buffer = slot
            .take()
            .ok_or_else(|| format!("output '{name}' is returned twice, its buffer is taken"))?;
        let len = buffer.raw().len;
        if len != byte_size {
            return Err(format!(
                "output '{name}' has {byte_size} bytes, its buffer has {len}"
            ));
        }
        Ok(Some(buffer))
    }
}

/// Allocator of BLS outputs: outputs with an [`OutputBuffer`] go to that
/// buffer, the others to CPU memory from `malloc`.
///
/// Owns the `TRITONSERVER_ResponseAllocator` and deletes it on drop, after
/// the last response of its request. `Send + Sync`.
pub(crate) struct ResponseAllocator {
    ptr: *mut triton_ng_sys::TRITONSERVER_ResponseAllocator,
}

// SAFETY: the allocator is an immutable table of callbacks, which Triton
// calls from its own threads, several at once; the callbacks share no state
// but the request's `OutputPlan`, which is behind a mutex.
unsafe impl Send for ResponseAllocator {}
unsafe impl Sync for ResponseAllocator {}

impl ResponseAllocator {
    pub(crate) fn new() -> Result<Self, TritonError> {
        let ptr = triton_call!(triton_ng_sys::TRITONSERVER_ResponseAllocatorNew(
            &mut _,
            Some(alloc_fn),
            Some(release_fn),
            None,
        ))?;
        let allocator = Self { ptr };

        // Without a query function TRITONBACKEND_RequestOutputBufferProperties
        // fails, and backends that ignore that error (onnxruntime) leak it on
        // every request.
        triton_call!(
            triton_ng_sys::TRITONSERVER_ResponseAllocatorSetQueryFunction(ptr, Some(query_fn),)
        )?;

        Ok(allocator)
    }

    pub(crate) fn as_ptr(&self) -> *mut triton_ng_sys::TRITONSERVER_ResponseAllocator {
        self.ptr
    }
}

impl Drop for ResponseAllocator {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                triton_ng_sys::TRITONSERVER_ResponseAllocatorDelete(self.ptr);
            }
        }
    }
}

fn error(
    code: triton_ng_sys::TRITONSERVER_Error_Code,
    message: &str,
) -> *mut triton_ng_sys::TRITONSERVER_Error {
    let message = std::ffi::CString::new(message.replace('\0', "?")).unwrap_or_default();
    unsafe { triton_ng_sys::TRITONSERVER_ErrorNew(code, message.as_ptr()) }
}

fn internal(message: &str) -> *mut triton_ng_sys::TRITONSERVER_Error {
    error(
        triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INTERNAL,
        message,
    )
}

fn invalid_arg(message: &str) -> *mut triton_ng_sys::TRITONSERVER_Error {
    error(
        triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INVALID_ARG,
        message,
    )
}

/// The output plan behind `userp` and the output name, `None` when either is
/// missing.
unsafe fn plan_and_name<'a>(
    userp: *mut c_void,
    tensor_name: *const c_char,
) -> Option<(&'a OutputPlan, String)> {
    if userp.is_null() || tensor_name.is_null() {
        return None;
    }
    // SAFETY: `userp` is the `OutputPlan` of the request, owned by its
    // inference context, which outlives every allocator callback.
    let plan = unsafe { &*(userp as *const OutputPlan) };
    let name = unsafe { CStr::from_ptr(tensor_name) }
        .to_string_lossy()
        .into_owned();
    Some((plan, name))
}

/// Triton calls this to allocate a buffer for each output tensor.
unsafe extern "C" fn alloc_fn(
    _allocator: *mut triton_ng_sys::TRITONSERVER_ResponseAllocator,
    tensor_name: *const c_char,
    byte_size: usize,
    _memory_type: triton_ng_sys::TRITONSERVER_MemoryType,
    _memory_type_id: i64,
    userp: *mut c_void,
    buffer: *mut *mut c_void,
    buffer_userp: *mut *mut c_void,
    actual_memory_type: *mut triton_ng_sys::TRITONSERVER_MemoryType,
    actual_memory_type_id: *mut i64,
) -> *mut triton_ng_sys::TRITONSERVER_Error {
    if buffer.is_null()
        || buffer_userp.is_null()
        || actual_memory_type.is_null()
        || actual_memory_type_id.is_null()
    {
        return invalid_arg("null output slot passed to the allocator");
    }

    let planned = match unsafe { plan_and_name(userp, tensor_name) } {
        Some((plan, name)) => match plan.take(&name, byte_size) {
            Ok(planned) => planned,
            Err(message) => return internal(&message),
        },
        None => None,
    };

    if let Some(output) = planned {
        let raw = output.raw();
        let tag: Box<OutputTag> = Box::new(Some(output));
        unsafe {
            *buffer = raw.ptr;
            *buffer_userp = Box::into_raw(tag) as *mut c_void;
            *actual_memory_type = raw.memory_type;
            *actual_memory_type_id = raw.memory_type_id;
        }
        return ptr::null_mut();
    }

    // Zero-size outputs are valid; return a null buffer.
    let buf = if byte_size == 0 {
        ptr::null_mut()
    } else {
        let buf = unsafe { libc::malloc(byte_size) };
        if buf.is_null() {
            return internal("failed to allocate output buffer");
        }
        buf
    };

    unsafe {
        *buffer = buf;
        *buffer_userp = ptr::null_mut();
        *actual_memory_type = CPU;
        *actual_memory_type_id = 0;
    }

    ptr::null_mut()
}

/// Triton calls this to ask where `alloc_fn` would place an output tensor.
unsafe extern "C" fn query_fn(
    _allocator: *mut triton_ng_sys::TRITONSERVER_ResponseAllocator,
    userp: *mut c_void,
    tensor_name: *const c_char,
    _byte_size: *mut usize,
    memory_type: *mut triton_ng_sys::TRITONSERVER_MemoryType,
    memory_type_id: *mut i64,
) -> *mut triton_ng_sys::TRITONSERVER_Error {
    if memory_type.is_null() || memory_type_id.is_null() {
        return invalid_arg("null memory type passed to the allocator query");
    }

    let (kind, id) = unsafe { plan_and_name(userp, tensor_name) }
        .and_then(|(plan, name)| plan.placement(&name))
        .unwrap_or((CPU, 0));
    unsafe {
        *memory_type = kind;
        *memory_type_id = id;
    }
    ptr::null_mut()
}

/// Triton calls this when it is done with a buffer allocated by `alloc_fn`.
unsafe extern "C" fn release_fn(
    _allocator: *mut triton_ng_sys::TRITONSERVER_ResponseAllocator,
    buffer: *mut c_void,
    buffer_userp: *mut c_void,
    _byte_size: usize,
    _memory_type: triton_ng_sys::TRITONSERVER_MemoryType,
    _memory_type_id: i64,
) -> *mut triton_ng_sys::TRITONSERVER_Error {
    if buffer_userp.is_null() {
        unsafe { libc::free(buffer) };
    } else {
        // The buffer is still in the tag when the response was not read.
        drop(unsafe { Box::from_raw(buffer_userp as *mut OutputTag) });
    }
    ptr::null_mut()
}

#[cfg(test)]
mod tests {
    use super::OutputPlan;
    use crate::MemoryManager;
    use std::collections::HashMap;

    fn plan() -> OutputPlan {
        let memory = MemoryManager::heap();
        OutputPlan::new(HashMap::from([
            ("ho".to_string(), Some(memory.gpu(1, 40).unwrap().into())),
            ("logits".to_string(), Some(memory.pinned(8).unwrap().into())),
        ]))
    }

    #[test]
    fn query_reports_the_memory_of_the_buffer() {
        let plan = plan();

        assert_eq!(plan.placement("ho"), Some((crate::GPU, 1)));
        assert_eq!(plan.placement("dec"), None);
    }

    #[test]
    fn buffer_is_handed_over_once() {
        let plan = plan();

        assert!(plan.take("ho", 40).unwrap().is_some());
        assert_eq!(plan.placement("ho"), None);
        assert!(plan.take("ho", 40).err().unwrap().contains("twice"));
    }

    #[test]
    fn buffer_of_another_size_is_refused() {
        let plan = plan();

        let error = plan.take("logits", 4).err().unwrap();

        assert!(error.contains("4 bytes"), "{error}");
        assert!(plan.take("logits", 8).is_err());
    }

    #[test]
    fn outputs_without_buffer_are_left_to_malloc() {
        assert!(plan().take("dec", 1280).unwrap().is_none());
    }
}
