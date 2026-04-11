use crate::TritonError;
use std::os::raw::{c_char, c_void};
use std::ptr;
use triton_ng_macros::triton_call;

// CUDA runtime functions — available when `cuda` feature is enabled.
// libcudart is linked via build.rs.
#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFreeHost(ptr: *mut c_void) -> i32;
}

/// Tag stored in `buffer_userp` so the release callback knows how to free.
type MemoryKind = triton_sys::TRITONSERVER_MemoryType;

pub struct ResponseAllocator {
    ptr: *mut triton_sys::TRITONSERVER_ResponseAllocator,
}

impl ResponseAllocator {
    pub fn new() -> Result<Self, TritonError> {
        let ptr = triton_call!(triton_sys::TRITONSERVER_ResponseAllocatorNew(
            &mut _,
            Some(alloc_fn),
            Some(release_fn),
            None,
        ))?;
        Ok(Self { ptr })
    }

    pub(crate) fn as_ptr(&self) -> *mut triton_sys::TRITONSERVER_ResponseAllocator {
        self.ptr
    }
}

impl Drop for ResponseAllocator {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                triton_sys::TRITONSERVER_ResponseAllocatorDelete(self.ptr);
            }
        }
    }
}

/// Triton calls this to allocate a buffer for each output tensor.
///
/// We store the *actual* memory kind in `buffer_userp` (as a `Box<MemoryKind>`)
/// so that `release_fn` knows which allocator to use when freeing.
unsafe extern "C" fn alloc_fn(
    _allocator: *mut triton_sys::TRITONSERVER_ResponseAllocator,
    _tensor_name: *const c_char,
    byte_size: usize,
    memory_type: triton_sys::TRITONSERVER_MemoryType,
    memory_type_id: i64,
    _userp: *mut c_void,
    buffer: *mut *mut c_void,
    buffer_userp: *mut *mut c_void,
    actual_memory_type: *mut triton_sys::TRITONSERVER_MemoryType,
    actual_memory_type_id: *mut i64,
) -> *mut triton_sys::TRITONSERVER_Error {
    // Zero-size outputs are valid; return a null buffer with no userp.
    if byte_size == 0 {
        unsafe {
            *buffer = ptr::null_mut();
            *buffer_userp = ptr::null_mut();
            *actual_memory_type = triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
            *actual_memory_type_id = 0;
        }
        return ptr::null_mut();
    }

    let (buf, actual_kind, actual_id) = allocate(byte_size, memory_type, memory_type_id);

    if buf.is_null() {
        return unsafe {
            triton_sys::TRITONSERVER_ErrorNew(
                triton_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INTERNAL,
                c"failed to allocate output buffer".as_ptr() as *const c_char,
            )
        };
    }

    let kind_ptr = Box::into_raw(Box::new(actual_kind)) as *mut c_void;

    unsafe {
        *buffer = buf;
        *buffer_userp = kind_ptr;
        *actual_memory_type = actual_kind;
        *actual_memory_type_id = actual_id;
    }

    ptr::null_mut()
}

/// Triton calls this when it is done with a buffer allocated by `alloc_fn`.
unsafe extern "C" fn release_fn(
    _allocator: *mut triton_sys::TRITONSERVER_ResponseAllocator,
    buffer: *mut c_void,
    buffer_userp: *mut c_void,
    _byte_size: usize,
    _memory_type: triton_sys::TRITONSERVER_MemoryType,
    _memory_type_id: i64,
) -> *mut triton_sys::TRITONSERVER_Error {
    // Null buffer means it was a zero-size allocation — nothing to free.
    if buffer.is_null() {
        return ptr::null_mut();
    }

    // Recover and drop the metadata tag.
    let kind = unsafe { *Box::from_raw(buffer_userp as *mut MemoryKind) };

    free_buffer(buffer, kind);

    ptr::null_mut()
}

// ── allocation helpers ────────────────────────────────────────────────────────

/// Returns `(ptr, actual_kind, actual_device_id)`.
/// Falls back to CPU when the requested kind is unavailable.
fn allocate(
    byte_size: usize,
    requested: triton_sys::TRITONSERVER_MemoryType,
    device_id: i64,
) -> (*mut c_void, triton_sys::TRITONSERVER_MemoryType, i64) {
    use triton_sys::{
        TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU as CPU,
        TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU_PINNED as CPU_PINNED,
        TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_GPU as GPU,
    };

    match requested {
        CPU => (cpu_alloc(byte_size), CPU, 0),

        CPU_PINNED => {
            #[cfg(feature = "cuda")]
            {
                let ptr = pinned_alloc(byte_size);
                if !ptr.is_null() {
                    return (ptr, CPU_PINNED, 0);
                }
            }
            (cpu_alloc(byte_size), CPU, 0)
        }

        GPU => {
            #[cfg(feature = "cuda")]
            {
                let ptr = gpu_alloc(byte_size);
                if !ptr.is_null() {
                    return (ptr, GPU, device_id);
                }
            }
            let _ = device_id;
            (cpu_alloc(byte_size), CPU, 0)
        }

        _ => (cpu_alloc(byte_size), CPU, 0),
    }
}

fn free_buffer(buf: *mut c_void, kind: triton_sys::TRITONSERVER_MemoryType) {
    use triton_sys::{
        TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU_PINNED as CPU_PINNED,
        TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_GPU as GPU,
    };

    match kind {
        CPU_PINNED => {
            #[cfg(feature = "cuda")]
            {
                pinned_free(buf);
                return;
            }
            #[cfg(not(feature = "cuda"))]
            cpu_free(buf);
        }

        GPU => {
            #[cfg(feature = "cuda")]
            {
                gpu_free(buf);
                return;
            }
            #[cfg(not(feature = "cuda"))]
            cpu_free(buf);
        }

        _ => cpu_free(buf),
    }
}

// ── low-level allocators ──────────────────────────────────────────────────────

#[inline]
fn cpu_alloc(size: usize) -> *mut c_void {
    unsafe { libc::malloc(size) }
}

#[inline]
fn cpu_free(ptr: *mut c_void) {
    unsafe { libc::free(ptr) }
}

#[cfg(feature = "cuda")]
#[inline]
fn gpu_alloc(size: usize) -> *mut c_void {
    let mut ptr: *mut c_void = ptr::null_mut();
    let rc = unsafe { cudaMalloc(&mut ptr, size) };
    if rc != 0 { ptr::null_mut() } else { ptr }
}

#[cfg(feature = "cuda")]
#[inline]
fn gpu_free(ptr: *mut c_void) {
    unsafe { cudaFree(ptr) };
}

#[cfg(feature = "cuda")]
#[inline]
fn pinned_alloc(size: usize) -> *mut c_void {
    let mut ptr: *mut c_void = ptr::null_mut();
    let rc = unsafe { cudaMallocHost(&mut ptr, size) };
    if rc != 0 { ptr::null_mut() } else { ptr }
}

#[cfg(feature = "cuda")]
#[inline]
fn pinned_free(ptr: *mut c_void) {
    unsafe { cudaFreeHost(ptr) };
}
