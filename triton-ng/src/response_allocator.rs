use crate::TritonError;
use std::os::raw::{c_char, c_void};
use std::ptr;
use triton_ng_macros::triton_call;

const CPU: triton_ng_sys::TRITONSERVER_MemoryType =
    triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;

/// Allocator of BLS outputs: every output goes to CPU memory from `malloc`.
pub struct ResponseAllocator {
    ptr: *mut triton_ng_sys::TRITONSERVER_ResponseAllocator,
}

impl ResponseAllocator {
    pub fn new() -> Result<Self, TritonError> {
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

/// Triton calls this to allocate a buffer for each output tensor.
unsafe extern "C" fn alloc_fn(
    _allocator: *mut triton_ng_sys::TRITONSERVER_ResponseAllocator,
    _tensor_name: *const c_char,
    byte_size: usize,
    _memory_type: triton_ng_sys::TRITONSERVER_MemoryType,
    _memory_type_id: i64,
    _userp: *mut c_void,
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
        return unsafe {
            triton_ng_sys::TRITONSERVER_ErrorNew(
                triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INVALID_ARG,
                c"null output slot passed to the allocator".as_ptr() as *const c_char,
            )
        };
    }

    // Zero-size outputs are valid; return a null buffer.
    let buf = if byte_size == 0 {
        ptr::null_mut()
    } else {
        let buf = unsafe { libc::malloc(byte_size) };
        if buf.is_null() {
            return unsafe {
                triton_ng_sys::TRITONSERVER_ErrorNew(
                    triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INTERNAL,
                    c"failed to allocate output buffer".as_ptr() as *const c_char,
                )
            };
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
    _userp: *mut c_void,
    _tensor_name: *const c_char,
    _byte_size: *mut usize,
    memory_type: *mut triton_ng_sys::TRITONSERVER_MemoryType,
    memory_type_id: *mut i64,
) -> *mut triton_ng_sys::TRITONSERVER_Error {
    if memory_type.is_null() || memory_type_id.is_null() {
        return unsafe {
            triton_ng_sys::TRITONSERVER_ErrorNew(
                triton_ng_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INVALID_ARG,
                c"null memory type passed to the allocator query".as_ptr() as *const c_char,
            )
        };
    }

    unsafe {
        *memory_type = CPU;
        *memory_type_id = 0;
    }
    ptr::null_mut()
}

/// Triton calls this when it is done with a buffer allocated by `alloc_fn`.
unsafe extern "C" fn release_fn(
    _allocator: *mut triton_ng_sys::TRITONSERVER_ResponseAllocator,
    buffer: *mut c_void,
    _buffer_userp: *mut c_void,
    _byte_size: usize,
    _memory_type: triton_ng_sys::TRITONSERVER_MemoryType,
    _memory_type_id: i64,
) -> *mut triton_ng_sys::TRITONSERVER_Error {
    unsafe { libc::free(buffer) };
    ptr::null_mut()
}
