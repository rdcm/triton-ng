use crate::error::TritonError;
use crate::utils::cstr_to_string;
use libc::c_char;
use triton_ng_macros::triton_call;

/// The backend, given to [`crate::CustomBackend::initialize`] and
/// [`crate::CustomBackend::finalize`].
///
/// Borrows the `TRITONBACKEND_Backend`: Triton owns it and keeps it alive
/// for the hook, so there is nothing to drop. Neither `Send` nor `Sync`.
///
/// # Examples
///
/// ```ignore
/// fn initialize(backend: &Backend) -> Result<(), Error> {
///     log::info(&format!("loading backend {}", backend.name()?));
///     Ok(())
/// }
/// ```
pub struct Backend {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Backend,
}

impl Backend {
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
