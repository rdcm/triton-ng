use crate::error::{Error, TritonError};
use crate::server::Server;
use crate::utils::cstr_to_string;
use libc::c_char;
use std::any::Any;
use std::ffi::c_void;
use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;
use std::ptr;
use std::str::FromStr;
use triton_ng_macros::triton_call;

/// Type-erased model state stored behind `TRITONBACKEND_ModelSetState`.
type BoxedState = Box<dyn Any + Send + Sync>;

pub struct Model {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Model,
}

impl Model {
    pub(crate) fn from_ptr(ptr: *mut triton_ng_sys::TRITONBACKEND_Model) -> Self {
        Self { ptr }
    }

    /// Stores backend-defined state for this model, replacing the previous one.
    ///
    /// Typically called from `Backend::model_initialize`; the state is shared
    /// by all instances of the model and dropped when the model is finalized.
    pub fn set_state<T: Any + Send + Sync>(&self, state: T) -> Result<(), TritonError> {
        self.clear_state()?;

        let boxed: Box<BoxedState> = Box::new(Box::new(state));
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
    pub fn state<T: Any>(&self) -> Result<&T, Error> {
        let state_ptr = self.raw_state()?;
        if state_ptr.is_null() {
            return Err("model state is not initialized".into());
        }

        // SAFETY: the pointer was produced by `set_state` from a `Box<BoxedState>`
        // and stays valid until `clear_state`, which only runs on model finalize.
        let state = unsafe { &*(state_ptr as *const BoxedState) };

        state
            .downcast_ref::<T>()
            .ok_or_else(|| "model state has unexpected type".into())
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
        let config_ptr = triton_call!(triton_ng_sys::TRITONBACKEND_ModelConfig(
            self.ptr,
            1,
            &mut _
        ))?;

        let mut base: *const i8 = ptr::null();
        let mut byte_size: usize = 0;
        triton_call!(triton_ng_sys::TRITONSERVER_MessageSerializeToJson(
            config_ptr,
            &mut base,
            &mut byte_size,
        ))?;

        let json = unsafe {
            let bytes = std::slice::from_raw_parts(base as *const u8, byte_size);
            String::from_utf8_lossy(bytes).into_owned()
        };

        unsafe { triton_ng_sys::TRITONSERVER_MessageDelete(config_ptr) };

        Ok(json)
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
