use crate::error::TritonError;
use crate::utils::cstr_to_string;
use libc::c_char;
use triton_ng_macros::triton_call;

pub struct BackendHandle {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Backend,
}

impl BackendHandle {
    pub(crate) fn from_ptr(ptr: *mut triton_ng_sys::TRITONBACKEND_Backend) -> Self {
        Self { ptr }
    }

    pub fn name(&self) -> Result<String, TritonError> {
        let mut name: *const c_char = std::ptr::null();
        triton_call!(triton_ng_sys::TRITONBACKEND_BackendName(
            self.ptr, &mut name
        ))?;
        Ok(unsafe { cstr_to_string(name) })
    }
}
