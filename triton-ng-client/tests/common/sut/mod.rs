#[path = "statistics.rs"]
mod statistics;

#[path = "model_configuration.rs"]
mod model_configuration;

#[path = "concurrency.rs"]
mod concurrency;

#[path = "cancellation.rs"]
mod cancellation;

#[path = "memory_buffers.rs"]
mod memory_buffers;

#[path = "shared_inputs.rs"]
mod shared_inputs;

#[path = "errors.rs"]
mod errors;

#[path = "tracing.rs"]
mod tracing;

#[path = "metadata.rs"]
mod metadata;

#[path = "model_control.rs"]
mod model_control;

#[path = "iterative_sequence.rs"]
mod iterative_sequence;

#[path = "bls_streaming.rs"]
mod bls_streaming;

#[path = "sequence_state.rs"]
mod sequence_state;

#[path = "custom_batching.rs"]
mod custom_batching;

#[path = "inference.rs"]
mod inference;

#[path = "streaming.rs"]
mod streaming;

use std::collections::HashMap;
use std::ffi::CString;

use super::config::Config;
use anyhow::{Context, Result};
use triton_ng_client as client;

use super::dto::{
    CudaRegion, InferInput, InferResult, LogValue, ModelConfig, ModelEntry, ModelInfo, ModelStats,
    OutputTensor, ServerInfo, ShmRegion, TensorMeta,
};

#[derive(Clone)]
pub struct Sut {
    inner: client::TritonClient,
    config: Config,
    models: std::sync::Arc<super::models::Models>,
}

impl Sut {
    pub async fn new(backend: &str, dependencies: &[&str]) -> Result<Self> {
        Self::prepare(backend, dependencies, true).await
    }

    pub async fn unloaded(backend: &str, dependencies: &[&str]) -> Result<Self> {
        Self::prepare(backend, dependencies, false).await
    }

    async fn prepare(backend: &str, dependencies: &[&str], load_backend: bool) -> Result<Self> {
        let mut config = Config::load(backend)?;
        let fixtures: Vec<_> = std::iter::once(backend)
            .chain(dependencies.iter().copied())
            .collect();
        let models =
            std::sync::Arc::new(super::models::Models::new(&config.triton_url, &fixtures)?);
        config.mnist_model = models.name(&config.mnist_model).to_string();
        config.backend_model = models.name(&config.backend_model).to_string();
        config.stream_model = models.name(&config.stream_model).to_string();
        config.sequence_state_model = models.name(&config.sequence_state_model).to_string();
        config.sequence_state_other_model =
            models.name(&config.sequence_state_other_model).to_string();
        config.batch_strategy_model = models.name(&config.batch_strategy_model).to_string();
        let triton_config = client::TritonClientConfig::new(&config.triton_url);
        let inner = client::TritonClient::new(triton_config)
            .await
            .context("Failed to connect to Triton")?;
        let sut = Self {
            inner,
            config,
            models,
        };
        for dependency in dependencies {
            sut.repository_model_load_at(sut.model(dependency)).await?;
        }
        if load_backend {
            sut.repository_model_load_at(&sut.config.backend_model)
                .await?;
        }
        Ok(sut)
    }

    pub fn model<'a>(&'a self, name: &'a str) -> &'a str {
        self.models.name(name)
    }

    pub async fn infer_with_parameters(
        &self,
        parameters: HashMap<String, client::Parameter>,
    ) -> Result<InferResult> {
        self.infer_model_with_parameters(&self.config.backend_model, parameters)
            .await
    }

    pub async fn infer_model_with_parameters(
        &self,
        model: &str,
        parameters: HashMap<String, client::Parameter>,
    ) -> Result<InferResult> {
        let model = self.model(model);
        let (inputs, outputs) = self.request_tensors(model, None).await?;
        let names: Vec<_> = outputs.iter().map(String::as_str).collect();
        self.infer_raw_with_parameters(model, inputs, &names, parameters)
            .await
    }

    pub async fn infer_with_input(
        &self,
        data: Vec<f32>,
        parameters: HashMap<String, client::Parameter>,
    ) -> Result<InferResult> {
        let model = &self.config.backend_model;
        let (inputs, outputs) = self.request_tensors(model, Some(data)).await?;
        let names: Vec<_> = outputs.iter().map(String::as_str).collect();
        self.infer_raw_with_parameters(model, inputs, &names, parameters)
            .await
    }

    pub async fn infer_with_id_and_parameters(
        &self,
        id: &str,
        parameters: HashMap<String, client::Parameter>,
    ) -> Result<InferResult> {
        let model = &self.config.backend_model;
        let (inputs, outputs) = self.request_tensors(model, None).await?;
        let names: Vec<_> = outputs.iter().map(String::as_str).collect();
        self.infer_raw_with_id_and_parameters(model, inputs, &names, id, parameters)
            .await
    }

    pub async fn infer_stream(
        &self,
        parameters: HashMap<String, client::Parameter>,
    ) -> Result<client::InferStream> {
        let model = &self.config.backend_model;
        let (inputs, outputs) = self.request_tensors(model, None).await?;
        let names: Vec<_> = outputs.iter().map(String::as_str).collect();
        self.infer_raw_stream(model, inputs, &names, parameters)
            .await
    }

    async fn request_tensors(
        &self,
        model: &str,
        data: Option<Vec<f32>>,
    ) -> Result<(Vec<InferInput>, Vec<String>)> {
        let metadata = self.model_metadata_at(model, None).await?;
        let input = metadata.inputs.first().context("fixture has no input")?;
        if metadata.inputs.len() != 1 || input.datatype != "FP32" {
            return Err(anyhow::anyhow!("fixture must have one FP32 input"));
        }
        let shape: Vec<_> = input
            .shape
            .iter()
            .map(|&dimension| if dimension == -1 { 1 } else { dimension })
            .collect();
        let size = shape.iter().try_fold(1_usize, |size, &dimension| {
            size.checked_mul(usize::try_from(dimension)?)
                .context("fixture input size overflow")
        })?;
        let data = data.unwrap_or_else(|| vec![0.0; size]);
        if data.len() != size {
            return Err(anyhow::anyhow!(
                "fixture input has {} values, expected {size}",
                data.len()
            ));
        }
        let outputs = metadata
            .outputs
            .into_iter()
            .map(|output| output.name)
            .collect();
        Ok((
            vec![InferInput {
                name: input.name.clone(),
                shape,
                data,
            }],
            outputs,
        ))
    }

    pub async fn model_ready(&self) -> Result<bool> {
        self.model_ready_at(&self.config.backend_model, None).await
    }
    pub async fn repository_model_ready(&self, model: &str) -> Result<bool> {
        self.model_ready_at(self.model(model), None).await
    }
    pub async fn model_metadata(&self) -> Result<ModelInfo> {
        self.model_metadata_at(&self.config.backend_model, None)
            .await
    }
    pub async fn model_config(&self) -> Result<Option<ModelConfig>> {
        self.model_config_at(&self.config.backend_model, None).await
    }
    pub async fn model_statistics(&self) -> Result<Vec<ModelStats>> {
        self.model_statistics_at(Some(&self.config.backend_model), None)
            .await
    }
    pub async fn repository_statistics(&self) -> Result<Vec<ModelStats>> {
        self.model_statistics_at(None, None).await
    }
    pub async fn model_memory_usage(&self) -> Result<Vec<client::MemoryUsage>> {
        self.model_memory_usage_at(&self.config.backend_model).await
    }
    pub async fn response_statistics(&self) -> Result<u64> {
        self.response_statistics_at(&self.config.backend_model)
            .await
    }
    pub async fn failed_and_cancelled_responses(&self) -> Result<(u64, u64)> {
        self.failed_and_cancelled_responses_at(&self.config.backend_model)
            .await
    }
    pub async fn successful_response_ns(&self) -> Result<u64> {
        self.successful_response_ns_at(&self.config.backend_model)
            .await
    }
    pub async fn repository_model_load(&self) -> Result<()> {
        self.repository_model_load_at(&self.config.backend_model)
            .await
    }
    pub async fn repository_model_unload(&self) -> Result<()> {
        self.repository_model_unload_at(&self.config.backend_model)
            .await
    }
    pub async fn trace_settings(
        &self,
        settings: HashMap<String, Vec<String>>,
    ) -> Result<HashMap<String, Vec<String>>> {
        self.trace_settings_at(settings, None).await
    }

    pub async fn server_live(&self) -> Result<bool> {
        self.inner
            .server_live(None)
            .await
            .context("server_live failed")
    }

    pub async fn server_ready(&self) -> Result<bool> {
        self.inner
            .server_ready(None)
            .await
            .context("server_ready failed")
    }

    pub async fn server_metadata(&self) -> Result<ServerInfo> {
        let info = self.inner.server_metadata(None).await?;
        Ok(ServerInfo {
            name: info.name,
            version: info.version,
            extensions: info.extensions,
        })
    }

    async fn model_ready_at(&self, model: &str, version: Option<&str>) -> Result<bool> {
        self.inner
            .model_ready(model, version, None)
            .await
            .context("model_ready failed")
    }

    async fn model_metadata_at(&self, model: &str, version: Option<&str>) -> Result<ModelInfo> {
        let info = self.inner.model_metadata(model, version, None).await?;
        Ok(ModelInfo {
            name: info.name,
            versions: info.versions,
            platform: info.platform,
            inputs: info
                .inputs
                .into_iter()
                .map(|t| TensorMeta {
                    name: t.name,
                    datatype: t.datatype.to_string(),
                    shape: t.shape,
                })
                .collect(),
            outputs: info
                .outputs
                .into_iter()
                .map(|t| TensorMeta {
                    name: t.name,
                    datatype: t.datatype.to_string(),
                    shape: t.shape,
                })
                .collect(),
        })
    }

    async fn model_config_at(
        &self,
        model: &str,
        version: Option<&str>,
    ) -> Result<Option<ModelConfig>> {
        let cfg = self.inner.model_config(model, version, None).await?;
        Ok(cfg.map(|c| {
            let proto = c.into_proto();
            ModelConfig {
                instance_groups: proto
                    .instance_group
                    .iter()
                    .map(|group| (group.kind, group.count))
                    .collect(),
                name: proto.name,
                platform: proto.platform,
            }
        }))
    }

    async fn model_statistics_at(
        &self,
        model: Option<&str>,
        version: Option<&str>,
    ) -> Result<Vec<ModelStats>> {
        let stats = self.inner.model_statistics(model, version, None).await?;
        Ok(stats
            .into_iter()
            .map(|s| ModelStats {
                name: s.name().to_string(),
                version: s.version().to_string(),
                inference_count: s.inference_count(),
                execution_count: s.execution_count(),
            })
            .collect())
    }

    async fn model_memory_usage_at(&self, model: &str) -> Result<Vec<client::MemoryUsage>> {
        let stats = self.inner.model_statistics(Some(model), None, None).await?;
        let stats = stats
            .into_iter()
            .find(|stats| stats.name() == model)
            .with_context(|| format!("{model} is missing from the statistics"))?;
        Ok(stats.memory_usage())
    }

    /// The number of responses the model reported per-response statistics for,
    /// the sum over every bucket Triton keeps for a response.
    async fn response_statistics_at(&self, model: &str) -> Result<u64> {
        let stats = self.inner.model_statistics(Some(model), None, None).await?;
        let stats = stats
            .into_iter()
            .find(|stats| stats.name() == model)
            .with_context(|| format!("{model} is missing from the statistics"))?;
        Ok(stats
            .into_proto()
            .response_stats
            .values()
            .map(|statistics| {
                [
                    statistics.success.as_ref(),
                    statistics.fail.as_ref(),
                    statistics.empty_response.as_ref(),
                    statistics.cancel.as_ref(),
                ]
                .into_iter()
                .flatten()
                .map(|bucket| bucket.count)
                .sum::<u64>()
            })
            .sum())
    }

    /// How many responses of the model the statistics count as failed and how
    /// many as cancelled, summed over the responses Triton keeps.
    async fn failed_and_cancelled_responses_at(&self, model: &str) -> Result<(u64, u64)> {
        let stats = self.inner.model_statistics(Some(model), None, None).await?;
        let stats = stats
            .into_iter()
            .find(|stats| stats.name() == model)
            .with_context(|| format!("{model} is missing from the statistics"))?;
        let mut counts = (0, 0);
        for statistics in stats.into_proto().response_stats.values() {
            counts.0 += statistics.fail.as_ref().map_or(0, |bucket| bucket.count);
            counts.1 += statistics.cancel.as_ref().map_or(0, |bucket| bucket.count);
        }
        Ok(counts)
    }

    /// The nanoseconds of the responses the statistics count as successful,
    /// summed over the responses Triton keeps.
    async fn successful_response_ns_at(&self, model: &str) -> Result<u64> {
        let stats = self.inner.model_statistics(Some(model), None, None).await?;
        let stats = stats
            .into_iter()
            .find(|stats| stats.name() == model)
            .with_context(|| format!("{model} is missing from the statistics"))?;
        Ok(stats
            .into_proto()
            .response_stats
            .values()
            .map(|statistics| statistics.success.as_ref().map_or(0, |bucket| bucket.ns))
            .sum())
    }

    async fn infer_raw(
        &self,
        model: &str,
        version: Option<&str>,
        inputs: Vec<InferInput>,
        output_names: &[&str],
        id: Option<String>,
    ) -> Result<InferResult> {
        let options = client::InferOptions {
            id,
            ..Default::default()
        };
        self.infer_with_options(model, version, inputs, output_names, options)
            .await
    }

    async fn infer_raw_with_parameters(
        &self,
        model: &str,
        inputs: Vec<InferInput>,
        output_names: &[&str],
        parameters: HashMap<String, client::Parameter>,
    ) -> Result<InferResult> {
        let options = client::InferOptions {
            parameters,
            ..Default::default()
        };
        self.infer_with_options(model, None, inputs, output_names, options)
            .await
    }

    async fn infer_raw_with_id_and_parameters(
        &self,
        model: &str,
        inputs: Vec<InferInput>,
        output_names: &[&str],
        id: &str,
        parameters: HashMap<String, client::Parameter>,
    ) -> Result<InferResult> {
        let options = client::InferOptions {
            id: Some(id.to_string()),
            parameters,
            ..Default::default()
        };
        self.infer_with_options(model, None, inputs, output_names, options)
            .await
    }

    async fn infer_raw_stream(
        &self,
        model: &str,
        inputs: Vec<InferInput>,
        output_names: &[&str],
        parameters: HashMap<String, client::Parameter>,
    ) -> Result<client::InferStream> {
        let options = client::InferOptions {
            parameters,
            ..Default::default()
        };
        Ok(self
            .inner
            .infer_stream(
                model,
                None,
                proto_inputs(inputs),
                output_names.iter().copied(),
                options,
            )
            .await?)
    }

    async fn infer_with_options(
        &self,
        model: &str,
        version: Option<&str>,
        inputs: Vec<InferInput>,
        output_names: &[&str],
        options: client::InferOptions,
    ) -> Result<InferResult> {
        let response = self
            .inner
            .infer(
                model,
                version,
                proto_inputs(inputs),
                output_names.iter().copied(),
                options,
            )
            .await?;
        Ok(infer_result(response))
    }

    pub async fn repository_index(&self, ready_only: bool) -> Result<Vec<ModelEntry>> {
        let idx = self.inner.repository_index(ready_only, None).await?;
        Ok(idx
            .into_iter()
            .map(|m| ModelEntry {
                name: m.name,
                state: m.state,
            })
            .collect())
    }

    async fn repository_model_load_at(&self, model: &str) -> Result<()> {
        self.inner
            .repository_model_load(model, None)
            .await
            .context("repository_model_load failed")
    }

    async fn repository_model_unload_at(&self, model: &str) -> Result<()> {
        self.inner
            .repository_model_unload(model, None)
            .await
            .context("repository_model_unload failed")
    }

    pub async fn system_shared_memory_status(
        &self,
        name: Option<&str>,
    ) -> Result<HashMap<String, ShmRegion>> {
        let regions = self.inner.system_shared_memory_status(name, None).await?;
        Ok(regions
            .into_iter()
            .map(|(k, r)| {
                (
                    k,
                    ShmRegion {
                        key: r.key,
                        offset: r.offset,
                        byte_size: r.byte_size,
                    },
                )
            })
            .collect())
    }

    pub async fn system_shared_memory_register(
        &self,
        name: &str,
        key: &str,
        offset: u64,
        byte_size: u64,
    ) -> Result<()> {
        self.inner
            .system_shared_memory_register(name, key, offset, byte_size, None)
            .await
            .context("system_shared_memory_register failed")
    }

    pub async fn system_shared_memory_unregister(&self, name: Option<&str>) -> Result<()> {
        self.inner
            .system_shared_memory_unregister(name, None)
            .await
            .context("system_shared_memory_unregister failed")
    }

    pub async fn cuda_shared_memory_status(
        &self,
        name: Option<&str>,
    ) -> Result<HashMap<String, CudaRegion>> {
        let regions = self.inner.cuda_shared_memory_status(name, None).await?;
        Ok(regions
            .into_iter()
            .map(|(k, r)| {
                (
                    k,
                    CudaRegion {
                        device_id: r.device_id,
                        byte_size: r.byte_size,
                    },
                )
            })
            .collect())
    }

    pub async fn cuda_shared_memory_register(
        &self,
        name: &str,
        raw_handle: Vec<u8>,
        device_id: i64,
        byte_size: u64,
    ) -> Result<()> {
        self.inner
            .cuda_shared_memory_register(name, raw_handle, device_id, byte_size, None)
            .await
            .context("cuda_shared_memory_register failed")
    }

    pub async fn cuda_shared_memory_unregister(&self, name: Option<&str>) -> Result<()> {
        self.inner
            .cuda_shared_memory_unregister(name, None)
            .await
            .context("cuda_shared_memory_unregister failed")
    }

    async fn trace_settings_at(
        &self,
        settings: HashMap<String, Vec<String>>,
        model: Option<&str>,
    ) -> Result<HashMap<String, Vec<String>>> {
        self.inner
            .trace_settings(settings, model, None)
            .await
            .context("trace_settings failed")
    }

    pub async fn log_settings(
        &self,
        settings: HashMap<String, LogValue>,
    ) -> Result<HashMap<String, LogValue>> {
        let proto = settings
            .into_iter()
            .map(|(k, v)| (k, log_value_into(v)))
            .collect();
        let result = self.inner.log_settings(proto, None).await?;
        Ok(result
            .into_iter()
            .map(|(k, v)| (k, log_value_from(v)))
            .collect())
    }

    pub fn scrape_metrics(&self) -> Result<String> {
        let address = &self.config.metrics_address;
        use std::io::{Read, Write};

        let mut stream = std::net::TcpStream::connect(address)
            .context("failed to connect to the metrics port")?;
        write!(
            stream,
            "GET /metrics HTTP/1.0\r\nHost: {address}\r\nConnection: close\r\n\r\n"
        )?;
        let mut response = String::new();
        stream.read_to_string(&mut response)?;
        let (head, body) = response
            .split_once("\r\n\r\n")
            .context("metrics response has no body")?;
        if !head.starts_with("HTTP/1.1 200") && !head.starts_with("HTTP/1.0 200") {
            return Err(anyhow::anyhow!("metrics endpoint answered {head}"));
        }
        Ok(body.to_string())
    }
}

fn proto_inputs(inputs: Vec<InferInput>) -> Vec<client::InferInput> {
    inputs
        .into_iter()
        .map(|i| client::InferInput::fp32(i.name, i.shape, i.data))
        .collect()
}

pub fn infer_result(response: client::InferResponse) -> InferResult {
    InferResult {
        id: response.id,
        parameters: response.parameters,
        outputs: response
            .outputs
            .into_iter()
            .map(|o| {
                let (data_f32, data_i32) = match o.data {
                    client::OutputData::Fp32(v) => (v, vec![]),
                    client::OutputData::Int32(v) => (vec![], v),
                    _ => (vec![], vec![]),
                };
                OutputTensor {
                    name: o.name,
                    shape: o.shape,
                    data_f32,
                    data_i32,
                }
            })
            .collect(),
    }
}

pub fn create_posix_shm(name: &str, size: i64) -> Result<()> {
    let cname = CString::new(name).context("invalid shared memory name")?;
    // SAFETY: cname is a live, NUL-terminated CString; shm_open does not retain it.
    let fd = unsafe { libc::shm_open(cname.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o600_u32) };
    if fd < 0 {
        return Err(anyhow::anyhow!(
            "shm_open failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: shm_open succeeded and fd has not been closed.
    let ret = unsafe { libc::ftruncate(fd, size) };
    // SAFETY: this function owns fd and closes it exactly once.
    unsafe { libc::close(fd) };
    if ret < 0 {
        return Err(anyhow::anyhow!(
            "ftruncate failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

pub fn remove_posix_shm(name: &str) -> Result<()> {
    let cname = CString::new(name).context("invalid shared memory name")?;
    // SAFETY: cname is a live, NUL-terminated CString; shm_unlink does not retain it.
    let ret = unsafe { libc::shm_unlink(cname.as_ptr()) };
    if ret < 0 {
        return Err(anyhow::anyhow!(
            "shm_unlink failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

fn log_value_into(v: LogValue) -> client::LogSettingValue {
    match v {
        LogValue::Bool(b) => client::LogSettingValue::Bool(b),
        LogValue::Uint32(u) => client::LogSettingValue::Uint32(u),
        LogValue::Text(s) => client::LogSettingValue::Text(s),
    }
}

fn log_value_from(v: client::LogSettingValue) -> LogValue {
    match v {
        client::LogSettingValue::Bool(b) => LogValue::Bool(b),
        client::LogSettingValue::Uint32(u) => LogValue::Uint32(u),
        client::LogSettingValue::Text(s) => LogValue::Text(s),
    }
}
