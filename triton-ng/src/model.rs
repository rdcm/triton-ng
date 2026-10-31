use crate::error::{Error, TritonError};
use crate::server::Server;
use crate::utils::{Message, cstr_to_string};
use libc::c_char;
use std::any::Any;
use std::ffi::c_void;
use std::fs::File;
use std::io::prelude::*;
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

    pub fn name(&self) -> Result<String, TritonError> {
        let mut model_name: *const c_char = std::ptr::null_mut();
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelName(
            self.ptr,
            &mut model_name
        ))?;

        Ok(unsafe { cstr_to_string(model_name) })
    }

    pub fn version(&self) -> Result<u64, TritonError> {
        let mut version = 0u64;
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelVersion(
            self.ptr,
            &mut version
        ))?;

        Ok(version)
    }

    pub fn location(&self) -> Result<String, TritonError> {
        let mut artifact_type: triton_ng_sys::TRITONBACKEND_ArtifactType = 0u32;
        let mut location: *const c_char = std::ptr::null_mut();
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelRepository(
            self.ptr,
            &mut artifact_type,
            &mut location,
        ))?;

        Ok(unsafe { cstr_to_string(location) })
    }

    pub fn path(&self, filename: &str) -> Result<PathBuf, Error> {
        Ok(PathBuf::from(format!(
            "{}/{}/{}",
            self.location()?,
            self.version()?,
            filename
        )))
    }

    pub fn load_file(&self, filename: &str) -> Result<Vec<u8>, Error> {
        let path = self.path(filename)?;
        let mut f = File::open(path)?;

        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer)?;

        Ok(buffer)
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
