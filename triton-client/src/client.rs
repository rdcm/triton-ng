use std::collections::HashMap;
use std::time::Duration;

use tonic::Request;
use tonic::transport::{Channel, ClientTlsConfig};
use triton_grpc_client::inference::grpc_inference_service_client::GrpcInferenceServiceClient;
use triton_grpc_client::inference::model_infer_request::{
    InferInputTensor, InferRequestedOutputTensor,
};
use triton_grpc_client::inference::{
    CudaSharedMemoryRegisterRequest, CudaSharedMemoryStatusRequest,
    CudaSharedMemoryUnregisterRequest, LogSettingsRequest, ModelConfigRequest, ModelInferRequest,
    ModelMetadataRequest, ModelReadyRequest, ModelStatisticsRequest, RepositoryIndexRequest,
    RepositoryModelLoadRequest, RepositoryModelUnloadRequest, ServerLiveRequest,
    ServerMetadataRequest, ServerReadyRequest, SystemSharedMemoryRegisterRequest,
    SystemSharedMemoryStatusRequest, SystemSharedMemoryUnregisterRequest, TraceSettingRequest,
    trace_setting_request,
};

use crate::datatype::Datatype;
use crate::error::{Error, Result};
use crate::infer::{InferInput, InferOptions, InferResponse};
use crate::types::{
    CudaMemoryRegion, LogSettingValue, ModelConfig, ModelIndex, ModelInfo, ModelStatistics,
    ServerInfo, SharedMemoryRegion, TensorMetadata,
};

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for [`TritonClient`].
#[derive(Debug, Clone)]
pub struct TritonClientConfig {
    url: String,
    tls: Option<ClientTlsConfig>,
}

impl TritonClientConfig {
    pub fn new(url: impl Into<String>) -> Self {
        Self { url: url.into(), tls: None }
    }

    /// Enables TLS. Pass `ClientTlsConfig::new()` for system roots, or configure
    /// a custom CA / client identity for mTLS.
    pub fn with_tls(mut self, tls: ClientTlsConfig) -> Self {
        self.tls = Some(tls);
        self
    }

    pub fn url(&self) -> &str {
        &self.url
    }
}

// ── Client ────────────────────────────────────────────────────────────────────

fn ver(v: Option<&str>) -> String {
    v.unwrap_or("").to_string()
}

fn apply_timeout<T>(mut req: Request<T>, timeout: Option<Duration>) -> Request<T> {
    if let Some(d) = timeout {
        req.set_timeout(d);
    }
    req
}

/// Async client for the Triton Inference Server gRPC API.
///
/// Cheaply [`Clone`]able — each clone shares the same underlying connection
/// and can be used concurrently from multiple tasks.
#[derive(Clone)]
pub struct TritonClient {
    config: TritonClientConfig,
    client: GrpcInferenceServiceClient<Channel>,
}

impl TritonClient {
    /// Connects to the Triton server. Returns an error if the server is unreachable.
    pub async fn new(config: TritonClientConfig) -> Result<Self> {
        let mut endpoint = Channel::from_shared(config.url.clone())
            .map_err(|e| Error::InvalidUrl(e.to_string()))?;

        if let Some(tls) = config.tls.clone() {
            endpoint = endpoint.tls_config(tls)?;
        }

        let channel = endpoint.connect().await?;
        Ok(Self {
            config,
            client: GrpcInferenceServiceClient::new(channel),
        })
    }

    /// Returns the configuration this client was created with.
    pub fn config(&self) -> &TritonClientConfig {
        &self.config
    }

    // ── Server ────────────────────────────────────────────────────────────────

    /// Returns `true` if the server is live (able to receive requests).
    pub async fn server_live(&self, timeout: Option<Duration>) -> Result<bool> {
        let r = self
            .client
            .clone()
            .server_live(apply_timeout(Request::new(ServerLiveRequest {}), timeout))
            .await?;
        Ok(r.into_inner().live)
    }

    /// Returns `true` if the server is ready (all models loaded and operational).
    pub async fn server_ready(&self, timeout: Option<Duration>) -> Result<bool> {
        let r = self
            .client
            .clone()
            .server_ready(apply_timeout(Request::new(ServerReadyRequest {}), timeout))
            .await?;
        Ok(r.into_inner().ready)
    }

    /// Returns server name, version, and supported extensions.
    pub async fn server_metadata(&self, timeout: Option<Duration>) -> Result<ServerInfo> {
        let r = self
            .client
            .clone()
            .server_metadata(apply_timeout(
                Request::new(ServerMetadataRequest {}),
                timeout,
            ))
            .await?;
        let m = r.into_inner();
        Ok(ServerInfo {
            name: m.name,
            version: m.version,
            extensions: m.extensions,
        })
    }

    // ── Model ─────────────────────────────────────────────────────────────────

    /// Returns `true` if the specified model is ready for inference.
    ///
    /// `version`: `None` checks the latest version.
    pub async fn model_ready(
        &self,
        model: &str,
        version: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<bool> {
        let r = self
            .client
            .clone()
            .model_ready(apply_timeout(
                Request::new(ModelReadyRequest {
                    name: model.to_string(),
                    version: ver(version),
                }),
                timeout,
            ))
            .await?;
        Ok(r.into_inner().ready)
    }

    /// Returns model metadata: tensor names, datatypes, and shapes.
    ///
    /// `version`: `None` returns metadata for the latest version.
    pub async fn model_metadata(
        &self,
        model: &str,
        version: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<ModelInfo> {
        let r = self
            .client
            .clone()
            .model_metadata(apply_timeout(
                Request::new(ModelMetadataRequest {
                    name: model.to_string(),
                    version: ver(version),
                }),
                timeout,
            ))
            .await?;
        let m = r.into_inner();

        let parse_tensors =
            |tensors: Vec<_>| {
                tensors
                .into_iter()
                .map(|t: triton_grpc_client::inference::model_metadata_response::TensorMetadata| {
                    Ok(TensorMetadata {
                        name: t.name,
                        datatype: Datatype::try_from(t.datatype.as_str())?,
                        shape: t.shape,
                    })
                })
                .collect::<Result<Vec<_>>>()
            };

        Ok(ModelInfo {
            name: m.name,
            versions: m.versions,
            platform: m.platform,
            inputs: parse_tensors(m.inputs)?,
            outputs: parse_tensors(m.outputs)?,
        })
    }

    /// Returns the full configuration for the specified model.
    ///
    /// `version`: `None` returns the configuration for the latest version.
    pub async fn model_config(
        &self,
        model: &str,
        version: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<Option<ModelConfig>> {
        let r = self
            .client
            .clone()
            .model_config(apply_timeout(
                Request::new(ModelConfigRequest {
                    name: model.to_string(),
                    version: ver(version),
                }),
                timeout,
            ))
            .await?;
        Ok(r.into_inner().config.map(ModelConfig))
    }

    /// Returns inference statistics.
    ///
    /// `model`: `None` returns statistics for all models.
    /// `version`: `None` returns statistics for all versions.
    pub async fn model_statistics(
        &self,
        model: Option<&str>,
        version: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<Vec<ModelStatistics>> {
        let r = self
            .client
            .clone()
            .model_statistics(apply_timeout(
                Request::new(ModelStatisticsRequest {
                    name: model.unwrap_or("").to_string(),
                    version: ver(version),
                }),
                timeout,
            ))
            .await?;
        Ok(r.into_inner()
            .model_stats
            .into_iter()
            .map(ModelStatistics)
            .collect())
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /// Runs inference on the specified model.
    ///
    /// `version`: `None` uses the latest version.
    /// `output_names`: pass an empty iterator to request all model outputs.
    pub async fn infer(
        &self,
        model: &str,
        version: Option<&str>,
        inputs: impl IntoIterator<Item = InferInput>,
        output_names: impl IntoIterator<Item = impl AsRef<str>>,
        options: InferOptions,
    ) -> Result<InferResponse> {
        let (proto_inputs, raw_contents): (Vec<_>, Vec<_>) = inputs
            .into_iter()
            .map(|t| {
                let proto = InferInputTensor {
                    name: t.name,
                    datatype: t.data.datatype().as_str().to_string(),
                    shape: t.shape,
                    contents: None,
                    parameters: Default::default(),
                };
                (proto, t.data.to_raw_bytes())
            })
            .unzip();

        let request = ModelInferRequest {
            model_name: model.to_string(),
            model_version: ver(version),
            id: options.id.unwrap_or_default(),
            inputs: proto_inputs,
            outputs: output_names
                .into_iter()
                .map(|name| InferRequestedOutputTensor {
                    name: name.as_ref().to_string(),
                    parameters: Default::default(),
                })
                .collect(),
            raw_input_contents: raw_contents,
            ..Default::default()
        };

        let r = self
            .client
            .clone()
            .model_infer(apply_timeout(Request::new(request), options.timeout))
            .await?;
        InferResponse::from_proto(r.into_inner())
    }

    // ── Repository ────────────────────────────────────────────────────────────

    /// Returns the index of all models across all repositories.
    ///
    /// `ready_only`: if `true`, only models currently ready for inference are returned.
    pub async fn repository_index(
        &self,
        ready_only: bool,
        timeout: Option<Duration>,
    ) -> Result<Vec<ModelIndex>> {
        let r = self
            .client
            .clone()
            .repository_index(apply_timeout(
                Request::new(RepositoryIndexRequest {
                    repository_name: String::new(),
                    ready: ready_only,
                }),
                timeout,
            ))
            .await?;
        Ok(r.into_inner()
            .models
            .into_iter()
            .map(|m| ModelIndex {
                name: m.name,
                version: m.version,
                state: m.state,
                reason: m.reason,
            })
            .collect())
    }

    /// Loads or reloads a model from the model repository.
    pub async fn repository_model_load(
        &self,
        model: &str,
        timeout: Option<Duration>,
    ) -> Result<()> {
        self.client
            .clone()
            .repository_model_load(apply_timeout(
                Request::new(RepositoryModelLoadRequest {
                    repository_name: String::new(),
                    model_name: model.to_string(),
                    parameters: Default::default(),
                }),
                timeout,
            ))
            .await?;
        Ok(())
    }

    /// Unloads a model from the server.
    pub async fn repository_model_unload(
        &self,
        model: &str,
        timeout: Option<Duration>,
    ) -> Result<()> {
        self.client
            .clone()
            .repository_model_unload(apply_timeout(
                Request::new(RepositoryModelUnloadRequest {
                    repository_name: String::new(),
                    model_name: model.to_string(),
                    parameters: Default::default(),
                }),
                timeout,
            ))
            .await?;
        Ok(())
    }

    // ── System shared memory ──────────────────────────────────────────────────

    /// Returns the status of registered system shared-memory regions, keyed by region name.
    ///
    /// `name`: `None` returns status for all registered regions.
    pub async fn system_shared_memory_status(
        &self,
        name: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<HashMap<String, SharedMemoryRegion>> {
        let r = self
            .client
            .clone()
            .system_shared_memory_status(apply_timeout(
                Request::new(SystemSharedMemoryStatusRequest {
                    name: name.unwrap_or("").to_string(),
                }),
                timeout,
            ))
            .await?;
        Ok(r.into_inner()
            .regions
            .into_iter()
            .map(|(k, s)| {
                (
                    k,
                    SharedMemoryRegion {
                        key: s.key,
                        offset: s.offset,
                        byte_size: s.byte_size,
                    },
                )
            })
            .collect())
    }

    /// Registers a system shared-memory region.
    pub async fn system_shared_memory_register(
        &self,
        name: &str,
        key: &str,
        offset: u64,
        byte_size: u64,
        timeout: Option<Duration>,
    ) -> Result<()> {
        self.client
            .clone()
            .system_shared_memory_register(apply_timeout(
                Request::new(SystemSharedMemoryRegisterRequest {
                    name: name.to_string(),
                    key: key.to_string(),
                    offset,
                    byte_size,
                }),
                timeout,
            ))
            .await?;
        Ok(())
    }

    /// Unregisters system shared-memory regions.
    ///
    /// `name`: `None` unregisters all system shared-memory regions.
    pub async fn system_shared_memory_unregister(
        &self,
        name: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<()> {
        self.client
            .clone()
            .system_shared_memory_unregister(apply_timeout(
                Request::new(SystemSharedMemoryUnregisterRequest {
                    name: name.unwrap_or("").to_string(),
                }),
                timeout,
            ))
            .await?;
        Ok(())
    }

    // ── CUDA shared memory ────────────────────────────────────────────────────

    /// Returns the status of registered CUDA shared-memory regions, keyed by region name.
    ///
    /// `name`: `None` returns status for all registered regions.
    pub async fn cuda_shared_memory_status(
        &self,
        name: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<HashMap<String, CudaMemoryRegion>> {
        let r = self
            .client
            .clone()
            .cuda_shared_memory_status(apply_timeout(
                Request::new(CudaSharedMemoryStatusRequest {
                    name: name.unwrap_or("").to_string(),
                }),
                timeout,
            ))
            .await?;
        Ok(r.into_inner()
            .regions
            .into_iter()
            .map(|(k, s)| {
                (
                    k,
                    CudaMemoryRegion {
                        device_id: s.device_id,
                        byte_size: s.byte_size,
                    },
                )
            })
            .collect())
    }

    /// Registers a CUDA shared-memory region.
    pub async fn cuda_shared_memory_register(
        &self,
        name: &str,
        raw_handle: Vec<u8>,
        device_id: i64,
        byte_size: u64,
        timeout: Option<Duration>,
    ) -> Result<()> {
        self.client
            .clone()
            .cuda_shared_memory_register(apply_timeout(
                Request::new(CudaSharedMemoryRegisterRequest {
                    name: name.to_string(),
                    raw_handle,
                    device_id,
                    byte_size,
                }),
                timeout,
            ))
            .await?;
        Ok(())
    }

    /// Unregisters CUDA shared-memory regions.
    ///
    /// `name`: `None` unregisters all CUDA shared-memory regions.
    pub async fn cuda_shared_memory_unregister(
        &self,
        name: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<()> {
        self.client
            .clone()
            .cuda_shared_memory_unregister(apply_timeout(
                Request::new(CudaSharedMemoryUnregisterRequest {
                    name: name.unwrap_or("").to_string(),
                }),
                timeout,
            ))
            .await?;
        Ok(())
    }

    // ── Trace settings ────────────────────────────────────────────────────────

    /// Gets or updates trace settings. Pass an empty map to only read current settings.
    ///
    /// `model`: `None` applies settings globally.
    pub async fn trace_settings(
        &self,
        settings: HashMap<String, Vec<String>>,
        model: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<HashMap<String, Vec<String>>> {
        let r = self
            .client
            .clone()
            .trace_setting(apply_timeout(
                Request::new(TraceSettingRequest {
                    model_name: model.unwrap_or("").to_string(),
                    settings: settings
                        .into_iter()
                        .map(|(k, v)| (k, trace_setting_request::SettingValue { value: v }))
                        .collect(),
                }),
                timeout,
            ))
            .await?;
        Ok(r.into_inner()
            .settings
            .into_iter()
            .map(|(k, v)| (k, v.value))
            .collect())
    }

    // ── Log settings ──────────────────────────────────────────────────────────

    /// Gets or updates log settings. Pass an empty map to only read current settings.
    pub async fn log_settings(
        &self,
        settings: HashMap<String, LogSettingValue>,
        timeout: Option<Duration>,
    ) -> Result<HashMap<String, LogSettingValue>> {
        let r = self
            .client
            .clone()
            .log_settings(apply_timeout(
                Request::new(LogSettingsRequest {
                    settings: settings.into_iter().map(|(k, v)| (k, v.into())).collect(),
                }),
                timeout,
            ))
            .await?;
        r.into_inner()
            .settings
            .into_iter()
            .map(|(k, v)| Ok((k, LogSettingValue::try_from(v)?)))
            .collect()
    }
}
