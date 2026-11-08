use crate::backend_config::BackendConfig;
use crate::error::{Error, TritonError};
use crate::message::serialize_to_json;
use crate::model::SharedState;
use crate::types::{ArtifactType, ExecutionPolicy};
use crate::utils::cstr_to_string;
use libc::c_char;
use std::any::Any;
use std::ffi::c_void;
use std::path::PathBuf;
use std::ptr;
use std::sync::Arc;
use triton_ng_macros::triton_call;

type BoxedState = SharedState;

/// The backend, given to [`crate::CustomBackend::initialize`],
/// [`crate::CustomBackend::attributes`] and [`crate::CustomBackend::finalize`].
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

    /// The name of the backend, as in `backend: "<name>"` of the model configs.
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

    /// The artifact type and the directory containing the backend shared
    /// library and the files next to it.
    ///
    /// The returned path is a copy; Triton keeps the original string for as
    /// long as the backend lives. Fails when Triton returns an artifact type
    /// this version of triton-ng does not know or a null location.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn initialize(backend: &Backend) -> Result<(), Error> {
    ///     let (artifact_type, directory) = backend.artifacts()?;
    ///     log::info(&format!("backend files: {directory:?} ({artifact_type:?})"));
    ///     Ok(())
    /// }
    /// ```
    pub fn artifacts(&self) -> Result<(ArtifactType, PathBuf), TritonError> {
        let mut artifact =
            triton_ng_sys::TRITONBACKEND_artifacttype_enum_TRITONBACKEND_ARTIFACT_FILESYSTEM;
        let mut directory = ptr::null();
        triton_call!(triton_ng_sys::TRITONBACKEND_BackendArtifacts(
            self.ptr,
            &mut artifact,
            &mut directory
        ))?;
        let artifact = ArtifactType::from_sys(artifact).map_err(TritonError::from_message)?;
        if directory.is_null() {
            return Err(TritonError::from_message(
                "TRITONBACKEND_BackendArtifacts returned a null location",
            ));
        }
        // SAFETY: the core returns `tb->Directory().c_str()`, owned by the
        // backend. It is copied here and never modified or freed.
        Ok((
            artifact,
            PathBuf::from(unsafe { cstr_to_string(directory) }),
        ))
    }

    /// Sets how Triton schedules execution on the instances of this backend.
    ///
    /// Call it from [`crate::CustomBackend::initialize`]: Triton reads the
    /// policy immediately after that hook. Models using the sequence batcher
    /// always use [`ExecutionPolicy::Blocking`], regardless of this setting.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn initialize(backend: &Backend) -> Result<(), Error> {
    ///     backend.set_execution_policy(ExecutionPolicy::DeviceBlocking)?;
    ///     Ok(())
    /// }
    /// ```
    pub fn set_execution_policy(&self, policy: ExecutionPolicy) -> Result<(), TritonError> {
        triton_call!(triton_ng_sys::TRITONBACKEND_BackendSetExecutionPolicy(
            self.ptr,
            policy.to_sys()
        ))
    }

    /// Stores backend-defined state shared by every model of this backend.
    ///
    /// Call it once, from [`crate::CustomBackend::initialize`]. Later backend
    /// hooks read it through [`Backend::state`] and every batch through
    /// [`crate::Context::backend_state`]. The state is dropped after
    /// [`crate::CustomBackend::finalize`], once no batch holds it. A second
    /// call fails: replacing the state would free it under existing references.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// struct State {
    ///     pool_size: usize,
    /// }
    ///
    /// fn initialize(backend: &Backend) -> Result<(), Error> {
    ///     let pool_size = backend.config()?.parameter_or("pool-size", 4)?;
    ///     backend.set_state(State { pool_size })?;
    ///     Ok(())
    /// }
    /// ```
    pub fn set_state<T: Any + Send + Sync>(&self, state: T) -> Result<(), TritonError> {
        if !self.raw_state()?.is_null() {
            return Err(TritonError::from_message("backend state is already set"));
        }

        let boxed: Box<BoxedState> = Box::new(Arc::new(state));
        let state_ptr = Box::into_raw(boxed) as *mut c_void;

        if let Err(e) = triton_call!(triton_ng_sys::TRITONBACKEND_BackendSetState(
            self.ptr, state_ptr
        )) {
            unsafe { drop(Box::from_raw(state_ptr as *mut BoxedState)) };
            return Err(e);
        }

        Ok(())
    }

    /// Returns the state previously stored with [`Backend::set_state`].
    ///
    /// Fails when no state is stored or it is not a `T`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn finalize(backend: &Backend) -> Result<(), Error> {
    ///     let state = backend.state::<State>()?;
    ///     log::info(&format!("closing the pool of {} workers", state.pool_size));
    ///     Ok(())
    /// }
    /// ```
    pub fn state<T: Any>(&self) -> Result<&T, Error> {
        let state = self
            .stored_state()?
            .ok_or("backend state is not initialized")?;
        state
            .downcast_ref::<T>()
            .ok_or_else(|| "backend state has unexpected type".into())
    }

    /// A new reference to the state stored with [`Backend::set_state`], if any.
    pub(crate) fn shared_state(&self) -> Result<Option<SharedState>, Error> {
        Ok(self.stored_state()?.map(Arc::clone))
    }

    fn stored_state(&self) -> Result<Option<&SharedState>, Error> {
        let state_ptr = self.raw_state()?;
        if state_ptr.is_null() {
            return Ok(None);
        }

        // SAFETY: the pointer was produced by `set_state` from a `Box<BoxedState>`
        // and stays valid until `clear_state`, which only runs on backend finalize.
        Ok(Some(unsafe { &*(state_ptr as *const BoxedState) }))
    }

    /// Drops the state stored with [`Backend::set_state`], if any.
    pub(crate) fn clear_state(&self) -> Result<(), TritonError> {
        let state_ptr = self.raw_state()?;
        if state_ptr.is_null() {
            return Ok(());
        }

        triton_call!(triton_ng_sys::TRITONBACKEND_BackendSetState(
            self.ptr,
            ptr::null_mut()
        ))?;
        unsafe { drop(Box::from_raw(state_ptr as *mut BoxedState)) };

        Ok(())
    }

    fn raw_state(&self) -> Result<*mut c_void, TritonError> {
        let mut state_ptr: *mut c_void = ptr::null_mut();
        triton_call!(triton_ng_sys::TRITONBACKEND_BackendState(
            self.ptr,
            &mut state_ptr
        ))?;
        Ok(state_ptr)
    }
}
