use std::fmt;

use crate::generated::inference::{log_settings_request, log_settings_response};

use crate::datatype::Datatype;
use crate::error::{Error, Result};

/// Server name, version, and supported extensions.
#[derive(Debug, Clone)]
pub struct ServerInfo {
    /// Server name (`triton`).
    pub name: String,
    /// Server version.
    pub version: String,
    /// Extensions of the inference protocol the server supports.
    pub extensions: Vec<String>,
}

/// Metadata for a single tensor (input or output).
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Name of the tensor.
    pub name: String,
    /// Element type of the tensor.
    pub datatype: Datatype,
    /// Tensor shape. A value of `-1` indicates a dynamic (variable-size) axis.
    /// Replace dynamic axes with actual sizes when constructing [`crate::InferInput`].
    pub shape: Vec<i64>,
}

/// Model metadata returned by the server.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Name of the model.
    pub name: String,
    /// Versions of the model the server knows.
    pub versions: Vec<String>,
    /// Platform or backend of the model (`onnxruntime_onnx`, `tensorrt_plan`, a custom backend name).
    pub platform: String,
    /// Inputs of the model.
    pub inputs: Vec<TensorMetadata>,
    /// Outputs of the model.
    pub outputs: Vec<TensorMetadata>,
}

/// Index entry for a model in the model repository.
#[derive(Debug, Clone)]
pub struct ModelIndex {
    /// Name of the model.
    pub name: String,
    /// Version of the model, empty when it is not loaded.
    pub version: String,
    /// State of the model (`READY`, `UNAVAILABLE`, ...), empty when it is not loaded.
    pub state: String,
    /// Why the model is in its state, empty when there is nothing to say.
    pub reason: String,
}

/// Full model configuration returned by Triton.
///
/// The underlying structure is a proto message from the Triton gRPC schema.
/// Call [`into_proto`](Self::into_proto) for detailed field access.
#[derive(Clone)]
pub struct ModelConfig(pub(crate) crate::generated::inference::ModelConfig);

impl ModelConfig {
    /// Returns the model name.
    pub fn name(&self) -> &str {
        &self.0.name
    }

    /// Returns the model platform (e.g. `"onnxruntime_onnx"`).
    pub fn platform(&self) -> &str {
        &self.0.platform
    }

    /// Consumes the wrapper and returns the underlying proto value.
    pub fn into_proto(self) -> crate::generated::inference::ModelConfig {
        self.0
    }
}

impl fmt::Debug for ModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ModelConfig {{ name: {:?}, platform: {:?} }}",
            self.0.name, self.0.platform
        )
    }
}

/// Inference statistics for a specific model version.
///
/// Call [`into_proto`](Self::into_proto) for detailed timing breakdowns.
#[derive(Clone)]
pub struct ModelStatistics(pub(crate) crate::generated::inference::ModelStatistics);

impl ModelStatistics {
    /// Returns the model name.
    pub fn name(&self) -> &str {
        &self.0.name
    }

    /// Returns the model version.
    pub fn version(&self) -> &str {
        &self.0.version
    }

    /// Timestamp of the last inference request in milliseconds since the epoch.
    pub fn last_inference_ms(&self) -> u64 {
        self.0.last_inference
    }

    /// Cumulative count of successful inference requests.
    pub fn inference_count(&self) -> u64 {
        self.0.inference_count
    }

    /// Cumulative count of model executions.
    pub fn execution_count(&self) -> u64 {
        self.0.execution_count
    }

    /// The memory the model and its instances reported, one entry per memory
    /// type and id. Empty when the backend has not reported any memory usage.
    pub fn memory_usage(&self) -> Vec<MemoryUsage> {
        self.0
            .memory_usage
            .iter()
            .map(|usage| MemoryUsage {
                memory_type: usage.r#type.clone(),
                id: usage.id,
                byte_size: usage.byte_size,
            })
            .collect()
    }

    /// Consumes the wrapper and returns the underlying proto value.
    pub fn into_proto(self) -> crate::generated::inference::ModelStatistics {
        self.0
    }
}

impl fmt::Debug for ModelStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ModelStatistics {{ name: {:?}, version: {:?}, inference_count: {} }}",
            self.0.name, self.0.version, self.0.inference_count
        )
    }
}

/// Memory of one memory type and id, as a model reports it for
/// [`ModelStatistics::memory_usage`].
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// The memory type Triton names: `"CPU"`, `"CPU_PINNED"` or `"GPU"`.
    pub memory_type: String,
    /// The id of the memory type: the CUDA device for `"GPU"`, zero otherwise.
    pub id: i64,
    /// Bytes held on that memory type and id.
    pub byte_size: u64,
}

/// A registered system shared-memory region.
#[derive(Debug, Clone)]
pub struct SharedMemoryRegion {
    /// Key of the region, as registered.
    pub key: String,
    /// Offset of the region in the shared memory, in bytes.
    pub offset: u64,
    /// Size of the region, in bytes.
    /// Size of the region, in bytes.
    pub byte_size: u64,
}

/// A registered CUDA shared-memory region.
#[derive(Debug, Clone)]
pub struct CudaMemoryRegion {
    /// CUDA device of the region.
    pub device_id: u64,
    /// Size of the region, in bytes.
    pub byte_size: u64,
}

/// A single log setting value.
#[derive(Debug, Clone)]
pub enum LogSettingValue {
    /// A boolean setting.
    Bool(bool),
    /// An unsigned integer setting.
    Uint32(u32),
    /// Freeform text setting.
    Text(String),
}

impl From<LogSettingValue> for log_settings_request::SettingValue {
    fn from(v: LogSettingValue) -> Self {
        use log_settings_request::setting_value::ParameterChoice;
        let choice = match v {
            LogSettingValue::Bool(b) => ParameterChoice::BoolParam(b),
            LogSettingValue::Uint32(u) => ParameterChoice::Uint32Param(u),
            LogSettingValue::Text(s) => ParameterChoice::StringParam(s),
        };
        Self {
            parameter_choice: Some(choice),
        }
    }
}

impl TryFrom<log_settings_response::SettingValue> for LogSettingValue {
    type Error = Error;

    fn try_from(v: log_settings_response::SettingValue) -> Result<Self> {
        use log_settings_response::setting_value::ParameterChoice;
        match v
            .parameter_choice
            .ok_or_else(|| Error::Decode("empty log setting value".into()))?
        {
            ParameterChoice::BoolParam(b) => Ok(Self::Bool(b)),
            ParameterChoice::Uint32Param(u) => Ok(Self::Uint32(u)),
            ParameterChoice::StringParam(s) => Ok(Self::Text(s)),
        }
    }
}
