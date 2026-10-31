use crate::error::{Error, TritonError};
use crate::server::Server;
use crate::utils::{Message, cstr_to_string};
use std::any::Any;
use std::ffi::c_void;
use std::path::PathBuf;
use std::ptr;
use std::str::FromStr;
use std::sync::Arc;
use triton_ng_macros::triton_call;

/// Type-erased model state, shared by the model and the batches in flight.
pub(crate) type SharedState = Arc<dyn Any + Send + Sync>;

/// Model state stored behind `TRITONBACKEND_ModelSetState`.
type BoxedState = SharedState;

pub struct Model {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Model,
}

impl Model {
    pub(crate) fn from_ptr(ptr: *mut triton_ng_sys::TRITONBACKEND_Model) -> Self {
        Self { ptr }
    }

    /// Stores backend-defined state for this model.
    ///
    /// Call it once, from `Backend::model_initialize`; the state is shared by
    /// all instances of the model, reaches every batch through
    /// [`crate::Context::state`] and is dropped when the model is finalized
    /// and no batch holds it. A second call fails: replacing the state would
    /// free it under the references returned by [`Model::state`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// struct State {
    ///     encoder: String,
    /// }
    ///
    /// fn model_initialize(model: &Model) -> Result<(), Error> {
    ///     let encoder = model.config_parameter_or("encoder", "encoder".to_string())?;
    ///     model.set_state(State { encoder })?;
    ///     Ok(())
    /// }
    /// ```
    pub fn set_state<T: Any + Send + Sync>(&self, state: T) -> Result<(), TritonError> {
        if !self.raw_state()?.is_null() {
            return Err(TritonError::from_message("model state is already set"));
        }

        let boxed: Box<BoxedState> = Box::new(Arc::new(state));
        let state_ptr = Box::into_raw(boxed) as *mut c_void;

        if let Err(e) = triton_call!(triton_ng_sys::TRITONBACKEND_ModelSetState(
            self.ptr, state_ptr
        )) {
            unsafe { drop(Box::from_raw(state_ptr as *mut BoxedState)) };
            return Err(e);
        }

        Ok(())
    }

    /// Returns the state previously stored with [`Model::set_state`].
    ///
    /// Fails when no state is stored or it is not a `T`.
    pub fn state<T: Any>(&self) -> Result<&T, Error> {
        let state = self
            .stored_state()?
            .ok_or("model state is not initialized")?;
        state
            .downcast_ref::<T>()
            .ok_or_else(|| "model state has unexpected type".into())
    }

    /// A new reference to the state stored with [`Model::set_state`], if any.
    pub(crate) fn shared_state(&self) -> Result<Option<SharedState>, Error> {
        Ok(self.stored_state()?.map(Arc::clone))
    }

    fn stored_state(&self) -> Result<Option<&SharedState>, Error> {
        let state_ptr = self.raw_state()?;
        if state_ptr.is_null() {
            return Ok(None);
        }

        // SAFETY: the pointer was produced by `set_state` from a `Box<BoxedState>`
        // and stays valid until `clear_state`, which only runs on model finalize.
        Ok(Some(unsafe { &*(state_ptr as *const BoxedState) }))
    }

    /// Drops the state stored with [`Model::set_state`], if any.
    pub(crate) fn clear_state(&self) -> Result<(), TritonError> {
        let state_ptr = self.raw_state()?;
        if state_ptr.is_null() {
            return Ok(());
        }

        triton_call!(triton_ng_sys::TRITONBACKEND_ModelSetState(
            self.ptr,
            ptr::null_mut()
        ))?;
        unsafe { drop(Box::from_raw(state_ptr as *mut BoxedState)) };

        Ok(())
    }

    fn raw_state(&self) -> Result<*mut c_void, TritonError> {
        let mut state_ptr: *mut c_void = ptr::null_mut();
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelState(
            self.ptr,
            &mut state_ptr
        ))?;
        Ok(state_ptr)
    }

    /// Name of the model, as in its config.
    pub fn name(&self) -> Result<String, TritonError> {
        let mut name = ptr::null();
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelName(self.ptr, &mut name))?;
        Ok(unsafe { cstr_to_string(name) })
    }

    /// Version of the model that is being served.
    pub fn version(&self) -> Result<u64, TritonError> {
        let mut version = 0;
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelVersion(
            self.ptr,
            &mut version
        ))?;
        Ok(version)
    }

    /// Directory of the model in the model repository, the one that holds its
    /// `config.pbtxt` and version directories.
    ///
    /// Fails when Triton serves the model from somewhere other than the local
    /// file system.
    pub fn directory(&self) -> Result<PathBuf, TritonError> {
        let mut artifact =
            triton_ng_sys::TRITONBACKEND_artifacttype_enum_TRITONBACKEND_ARTIFACT_FILESYSTEM;
        let mut directory = ptr::null();
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelRepository(
            self.ptr,
            &mut artifact,
            &mut directory
        ))?;

        if artifact
            != triton_ng_sys::TRITONBACKEND_artifacttype_enum_TRITONBACKEND_ARTIFACT_FILESYSTEM
        {
            return Err(TritonError::from_message(
                "the model repository is not on the local file system",
            ));
        }
        Ok(PathBuf::from(unsafe { cstr_to_string(directory) }))
    }

    /// Path of `file_name` in the directory of the served version of the model.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // <repository>/punctuation/1/tokenizer.json
    /// let tokenizer = model.path("tokenizer.json")?;
    /// ```
    pub fn path(&self, file_name: &str) -> Result<PathBuf, Error> {
        Ok(self
            .directory()?
            .join(self.version()?.to_string())
            .join(file_name))
    }

    /// Reads `file_name` from the directory of the served version of the
    /// model, see [`Model::path`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let vocabulary = String::from_utf8(model.load_file("vocab.txt")?)?;
    /// ```
    pub fn load_file(&self, file_name: &str) -> Result<Vec<u8>, Error> {
        Ok(std::fs::read(self.path(file_name)?)?)
    }

    pub fn get_server(&self) -> Result<Server, TritonError> {
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_ModelServer(self.ptr, &mut _))?;
        Server::from_ptr(ptr)
    }

    /// Returns the model configuration as a JSON string (ModelConfig protobuf format).
    pub fn config_json(&self) -> Result<String, TritonError> {
        let config = Message::from_ptr(triton_call!(triton_ng_sys::TRITONBACKEND_ModelConfig(
            self.ptr,
            1,
            &mut _
        ))?);

        config.to_json()
    }

    /// Reads a string parameter from the model configuration `parameters` map.
    pub fn config_parameter(&self, key: &str) -> Result<Option<String>, TritonError> {
        let json = self.config_json()?;
        let value = serde_json::from_str::<serde_json::Value>(&json)
            .map_err(|e| TritonError::from_message(e.to_string()))?;
        Ok(value["parameters"][key]["string_value"]
            .as_str()
            .map(str::to_owned))
    }

    /// Reads and parses a parameter from the model configuration, falling back
    /// to `default` when the parameter is absent.
    pub fn config_parameter_or<T>(&self, key: &str, default: T) -> Result<T, Error>
    where
        T: FromStr,
        T::Err: std::fmt::Display,
    {
        match self.config_parameter(key)? {
            None => Ok(default),
            Some(raw) => raw.trim().parse::<T>().map_err(|e| {
                format!("invalid value '{raw}' of config parameter '{key}': {e}").into()
            }),
        }
    }

    /// Reads a required string parameter from the model configuration.
    pub fn required_config_parameter(&self, key: &str) -> Result<String, Error> {
        self.config_parameter(key)?
            .ok_or_else(|| format!("missing '{key}' config parameter").into())
    }
}
