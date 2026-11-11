use crate::backend::Backend;
use crate::backend_state::SharedState;
use crate::buffer_attributes::BufferAttributes;
use crate::error::{Error, TritonError};
use crate::memory_manager::MemoryManager;
use crate::message::Message;
use crate::server::Server;
use crate::utils::{cstr_to_path, cstr_to_string, parse_parameter};
use std::any::Any;
use std::ffi::c_void;
use std::path::PathBuf;
use std::ptr;
use std::str::FromStr;
use std::sync::Arc;
use triton_ng_macros::triton_call;

/// Model state stored behind `TRITONBACKEND_ModelSetState`.
type BoxedState = SharedState;

/// The `TRITONBACKEND_Model` a value came from, as the address Triton keeps it
/// at and the version it is served as.
///
/// `Copy` so that it travels with everything a backend reads from a model; the
/// address is only ever compared, never read from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ModelAddress {
    pub(crate) address: usize,
    pub(crate) version: u64,
}

/// The model a value came from: its [`ModelAddress`] and the name it is served
/// under, which is what an error says when the value is used with a request of
/// another model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelIdentity {
    pub(crate) address: ModelAddress,
    pub(crate) name: String,
}

/// A model that uses the backend, given to the model and instance hooks.
///
/// Borrows the `TRITONBACKEND_Model`: Triton owns it and keeps it alive for
/// the hook, so there is nothing to drop. Neither `Send` nor `Sync`: use it
/// only inside the hook; batches get what they need through
/// [`crate::Context`].
///
/// # Examples
///
/// ```ignore
/// fn model_initialize(model: &Model) -> Result<(), Error> {
///     let vocab = model.load_file("vocab.txt")?;
///     model.set_state(State { vocab })?;
///     Ok(())
/// }
/// ```
pub struct Model {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Model,
}

impl Model {
    pub(crate) fn from_ptr(ptr: *mut triton_ng_sys::TRITONBACKEND_Model) -> Self {
        Self { ptr }
    }

    pub(crate) fn as_ptr(&self) -> *mut triton_ng_sys::TRITONBACKEND_Model {
        self.ptr
    }

    /// Stores backend-defined state for this model.
    ///
    /// Call it once, from `CustomBackend::model_initialize`; the state is shared by
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

    /// A new reference to the state of the backend this model uses, if any.
    pub(crate) fn backend_state(&self) -> Result<Option<SharedState>, Error> {
        let backend = triton_call!(triton_ng_sys::TRITONBACKEND_ModelBackend(
            self.as_ptr(),
            &mut _
        ))?;
        if backend.is_null() {
            return Err("TRITONBACKEND_ModelBackend returned null".into());
        }
        // SAFETY: the core returns the backend the model was loaded with, which
        // outlives the model; the wrapper only borrows it here.
        Backend::from_ptr(backend).shared_state()
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

    /// Where Triton keeps this model and which version it serves, see
    /// [`ModelAddress`].
    pub(crate) fn address(&self) -> Result<ModelAddress, TritonError> {
        Ok(ModelAddress {
            address: self.ptr as usize,
            version: self.version()?,
        })
    }

    /// This model as the value a [`crate::StateConfig`] remembers, see
    /// [`ModelIdentity`].
    pub(crate) fn identity(&self) -> Result<ModelIdentity, TritonError> {
        Ok(ModelIdentity {
            address: self.address()?,
            name: self.name()?,
        })
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
        // SAFETY: the core returns the path of the repository, owned by the
        // model. It is copied here and never modified or freed.
        unsafe { cstr_to_path(directory) }
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

    /// Reports the memory the model itself holds, for the `memory_usage` of
    /// `/v2/models/<model>/stats`.
    ///
    /// Pass one [`BufferAttributes`] per memory type and id, each holding the
    /// total of that kind; entries of the same memory type and id are summed
    /// before they are reported, since Triton keeps one entry per memory type
    /// and id. Call it from `CustomBackend::model_initialize` or any later hook
    /// of the model: each call replaces what the previous one reported and an
    /// empty slice clears it. Triton copies the entries while the call runs, so
    /// the slice can be dropped afterwards. Instances report their own memory
    /// through [`crate::ModelInstance::report_memory_usage`].
    ///
    /// # Errors
    ///
    /// Fails when the entries of one memory type and id add up to more than a
    /// `u64`, when `usage` holds more entries than Triton can count, or when
    /// Triton rejects the report.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn model_initialize(model: &Model) -> Result<(), Error> {
    ///     let weights = 64 << 20;
    ///     model.report_memory_usage(&[BufferAttributes::new(MemoryType::Gpu { device_id: 0 }, weights)?])?;
    ///     Ok(())
    /// }
    /// ```
    pub fn report_memory_usage(&self, usage: &[BufferAttributes]) -> Result<(), TritonError> {
        let (entries, count) = BufferAttributes::report(usage)?;
        let mut pointers: Vec<_> = entries.iter().map(BufferAttributes::as_ptr).collect();
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelReportMemoryUsage(
            self.ptr,
            pointers.as_mut_ptr(),
            count
        ))
    }

    /// Triton's memory manager, the one batches get from
    /// [`crate::Context::memory`].
    ///
    /// Usable in every hook of the model and of its instances, for buffers that
    /// live in the state of the model or of an instance instead of being
    /// allocated per batch.
    ///
    /// # Errors
    ///
    /// Fails when Triton cannot hand out the memory manager.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn model_instance_initialize(instance: &ModelInstance) -> Result<(), Error> {
    ///     let pinned = instance.model()?.memory_manager()?.pinned(1 << 20)?;
    ///     instance.set_state(Staging(pinned))?;
    ///     Ok(())
    /// }
    /// ```
    pub fn memory_manager(&self) -> Result<MemoryManager, TritonError> {
        MemoryManager::of_model(self)
    }

    /// The server the model is loaded in, for BLS calls from the hooks that have a `Model`; batches get it from [`crate::Context::server`].
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

    /// Whether Triton asks the backend to complete the config of the model:
    /// on unless Triton runs with `--disable-auto-complete-config`.
    pub(crate) fn auto_complete_config_enabled(&self) -> Result<bool, TritonError> {
        let mut enabled = false;
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelAutoCompleteConfig(
            self.ptr,
            &mut enabled
        ))?;
        Ok(enabled)
    }

    /// Replaces the config of the model with `json`, a model config of
    /// version 1. Triton takes only some parts of it, see
    /// [`crate::AutoCompleteConfig`], and validates the result after
    /// `model_initialize`.
    pub(crate) fn set_config(&self, json: &str) -> Result<(), TritonError> {
        let message = Message::from_json(json)?;
        // The core only serializes the message (TritonModel::UpdateModelConfig)
        // and does not keep it: `message` deletes it afterwards.
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelSetConfig(
            self.ptr,
            1,
            message.as_ptr()
        ))
    }

    /// Whether the model is decoupled
    /// (`model_transaction_policy { decoupled: true }`): its requests can get
    /// any number of responses, see [`crate::Request::into_sender`].
    ///
    /// Fails when the configuration cannot be read or `decoupled` is not a
    /// boolean.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn model_initialize(model: &Model) -> Result<(), Error> {
    ///     if !model.is_decoupled()? {
    ///         return Err("the model streams its responses, set decoupled: true".into());
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn is_decoupled(&self) -> Result<bool, TritonError> {
        let json = self.config_json()?;
        let value = serde_json::from_str::<serde_json::Value>(&json)
            .map_err(|e| TritonError::from_message(e.to_string()))?;
        match &value["model_transaction_policy"]["decoupled"] {
            serde_json::Value::Null => Ok(false),
            serde_json::Value::Bool(decoupled) => Ok(*decoupled),
            other => Err(TritonError::from_message(format!(
                "model_transaction_policy.decoupled is not a boolean: {other}"
            ))),
        }
    }

    /// `max_batch_size` of the model config, 0 for a model whose inputs have
    /// no batch dimension.
    ///
    /// Fails when the configuration cannot be read or `max_batch_size` is not
    /// a number that fits an `u32`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn model_initialize(model: &Model) -> Result<(), Error> {
    ///     if model.max_batch_size()? == 0 {
    ///         return Err("the model batches its requests, set max_batch_size".into());
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn max_batch_size(&self) -> Result<u32, TritonError> {
        let json = self.config_json()?;
        let value = serde_json::from_str::<serde_json::Value>(&json)
            .map_err(|e| TritonError::from_message(e.to_string()))?;
        match &value["max_batch_size"] {
            serde_json::Value::Null => Ok(0),
            serde_json::Value::Number(number) => number
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| {
                    TritonError::from_message(format!(
                        "max_batch_size does not fit an u32: {number}"
                    ))
                }),
            other => Err(TritonError::from_message(format!(
                "max_batch_size is not a number: {other}"
            ))),
        }
    }

    /// Whether the model reschedules its requests
    /// (`sequence_batching { iterative_sequence: true }`), see
    /// [`crate::Request::reschedule`].
    ///
    /// Fails when the configuration cannot be read or `iterative_sequence`
    /// is not a boolean.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn model_initialize(model: &Model) -> Result<(), Error> {
    ///     if !model.is_iterative_sequence()? {
    ///         return Err("the model generates step by step, set iterative_sequence: true".into());
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn is_iterative_sequence(&self) -> Result<bool, TritonError> {
        let json = self.config_json()?;
        let value = serde_json::from_str::<serde_json::Value>(&json)
            .map_err(|e| TritonError::from_message(e.to_string()))?;
        match &value["sequence_batching"]["iterative_sequence"] {
            serde_json::Value::Null => Ok(false),
            serde_json::Value::Bool(iterative) => Ok(*iterative),
            other => Err(TritonError::from_message(format!(
                "sequence_batching.iterative_sequence is not a boolean: {other}"
            ))),
        }
    }

    /// The states the `state` section of the `sequence_batching` configuration
    /// declares, in the order the configuration lists them.
    ///
    /// A [`crate::State`] is created from one of them, which is how the
    /// datatype and the fixed dimensions of the configured state reach the code
    /// that writes it. A backend asks for them once, when it initializes the
    /// model, and keeps them.
    ///
    /// Fails when the configuration cannot be read, an entry has no
    /// `output_name` or `input_name`, or names a datatype this version does not
    /// know.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn model_initialize(model: &Model, state: &mut State) -> Result<(), Error> {
    ///     state.states = model.sequence_states()?;
    ///     Ok(())
    /// }
    /// ```
    pub fn sequence_states(&self) -> Result<Vec<crate::StateConfig>, TritonError> {
        let json = self.config_json()?;
        let value = serde_json::from_str::<serde_json::Value>(&json)
            .map_err(|e| TritonError::from_message(e.to_string()))?;
        let Some(states) = value["sequence_batching"]["state"].as_array() else {
            return Ok(Vec::new());
        };
        let model = self.identity()?;
        states
            .iter()
            .map(|entry| crate::StateConfig::from_json(entry, model.clone()))
            .collect()
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
            Some(raw) => Ok(parse_parameter("config", key, &raw)?),
        }
    }

    /// Reads a required string parameter from the model configuration.
    pub fn required_config_parameter(&self, key: &str) -> Result<String, Error> {
        self.config_parameter(key)?
            .ok_or_else(|| format!("missing '{key}' config parameter").into())
    }
}
