use std::fmt;

use triton_grpc_client::inference::{log_settings_request, log_settings_response};

use crate::datatype::Datatype;
use crate::error::{Error, Result};

// ── Server ────────────────────────────────────────────────────────────────────

/// Server name, version, and supported extensions.
#[derive(Debug, Clone)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
    pub extensions: Vec<String>,
}

// ── Model ─────────────────────────────────────────────────────────────────────

/// Metadata for a single tensor (input or output).
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub name: String,
    pub datatype: Datatype,
    /// Tensor shape. A value of `-1` indicates a dynamic (variable-size) axis.
    /// Replace dynamic axes with actual sizes when constructing [`crate::InferInput`].
    pub shape: Vec<i64>,
}

/// Model metadata returned by the server.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub versions: Vec<String>,
    pub platform: String,
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
}

/// Index entry for a model in the model repository.
#[derive(Debug, Clone)]
pub struct ModelIndex {
    pub name: String,
    pub version: String,
    pub state: String,
    pub reason: String,
}

/// Full model configuration returned by Triton.
///
/// The underlying structure is a proto message from the Triton gRPC schema.
/// Call [`into_proto`](Self::into_proto) for detailed field access.
#[derive(Clone)]
pub struct ModelConfig(pub(crate) triton_grpc_client::inference::ModelConfig);

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
    pub fn into_proto(self) -> triton_grpc_client::inference::ModelConfig {
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
pub struct ModelStatistics(pub(crate) triton_grpc_client::inference::ModelStatistics);

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

    /// Consumes the wrapper and returns the underlying proto value.
    pub fn into_proto(self) -> triton_grpc_client::inference::ModelStatistics {
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

// ── Shared memory ─────────────────────────────────────────────────────────────

/// A registered system shared-memory region.
#[derive(Debug, Clone)]
pub struct SharedMemoryRegion {
    pub key: String,
    pub offset: u64,
    pub byte_size: u64,
}

/// A registered CUDA shared-memory region.
#[derive(Debug, Clone)]
pub struct CudaMemoryRegion {
    pub device_id: u64,
    pub byte_size: u64,
}

// ── Log settings ──────────────────────────────────────────────────────────────

/// A single log setting value.
#[derive(Debug, Clone)]
pub enum LogSettingValue {
    Bool(bool),
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
