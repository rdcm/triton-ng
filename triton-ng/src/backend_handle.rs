use crate::error::TritonError;
use crate::ffi_call;
use crate::utils::cstr_to_string;
use libc::c_char;

pub struct BackendHandle {
    ptr: *mut triton_sys::TRITONBACKEND_Backend,
}

impl BackendHandle {
    pub fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_Backend) -> Self {
        Self { ptr }
    }

    pub fn name(&self) -> Result<String, TritonError> {
        let mut name: *const c_char = std::ptr::null();
        ffi_call!(triton_sys::TRITONBACKEND_BackendName(self.ptr, &mut name))?;
        Ok(unsafe { cstr_to_string(name) })
    }

    pub fn as_ptr(&self) -> *mut triton_sys::TRITONBACKEND_Backend {
        self.ptr
    }
}
