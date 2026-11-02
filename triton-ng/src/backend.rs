use crate::backend_config::BackendConfig;
use crate::error::{Error, TritonError};
use crate::message::serialize_to_json;
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

    /// The backend config given to Triton with
    /// `--backend-config=<backend>,<key>=<value>`, see [`BackendConfig`].
    ///
    /// Fails when Triton gives no config or it is not the JSON Triton
    /// documents.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn initialize(backend: &Backend) -> Result<(), Error> {
    ///     let pool_size: usize = backend.config()?.parameter_or("pool-size", 4)?;
    ///     Ok(())
    /// }
    /// ```
    pub fn config(&self) -> Result<BackendConfig, Error> {
        let message = triton_call!(triton_ng_sys::TRITONBACKEND_BackendConfig(self.ptr, &mut _))?;
        if message.is_null() {
            return Err("TRITONBACKEND_BackendConfig returned null".into());
        }

        // SAFETY: the message is a member of the TritonBackend (core
        // backend_manager.cc, `&tb->BackendConfig()`), which outlives this
        // borrowed `Backend`. Triton keeps owning it: it is only read here,
        // never deleted.
        let json = unsafe { serialize_to_json(message) }?;
        Ok(BackendConfig::from_json(&json)?)
    }
}
