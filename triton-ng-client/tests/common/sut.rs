use std::collections::HashMap;
use std::ffi::CString;

use anyhow::{Context, Result, bail};
use triton_ng_client as client;

use super::dto::{
    CudaRegion, InferInput, InferResult, LogValue, ModelConfig, ModelEntry, ModelInfo, ModelStats,
    OutputTensor, ServerInfo, ShmRegion, TensorMeta,
};

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct Config {
    pub triton_url: String,
    pub mnist_model: String,
    pub input_name: String,
    pub output_name: String,
}

fn load_config() -> Result<Config> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("config.json");
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read test config from {}", path.display()))?;
    serde_json::from_str(&content).context("Failed to parse test config")
}

// ── Sut ───────────────────────────────────────────────────────────────────────

pub struct Sut {
    inner: client::TritonClient,
    pub config: Config,
}

impl Sut {
    pub async fn new() -> Result<Sut> {
        let config = load_config()?;
        let triton_config = client::TritonClientConfig::new(&config.triton_url);
        let inner = client::TritonClient::new(triton_config)
            .await
            .context("Failed to connect to Triton")?;
        Ok(Sut { inner, config })
    }

    // ── Server ────────────────────────────────────────────────────────────────

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

    // ── Model ─────────────────────────────────────────────────────────────────

    pub async fn model_ready(&self, model: &str, version: Option<&str>) -> Result<bool> {
        self.inner
            .model_ready(model, version, None)
            .await
            .context("model_ready failed")
    }

    pub async fn model_metadata(&self, model: &str, version: Option<&str>) -> Result<ModelInfo> {
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

    pub async fn model_config(
        &self,
        model: &str,
        version: Option<&str>,
    ) -> Result<Option<ModelConfig>> {
        let cfg = self.inner.model_config(model, version, None).await?;
        Ok(cfg.map(|c| ModelConfig {
            name: c.name().to_string(),
            platform: c.platform().to_string(),
        }))
    }

    pub async fn model_statistics(
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
            })
            .collect())
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    pub async fn infer(
        &self,
        model: &str,
        version: Option<&str>,
        inputs: Vec<InferInput>,
        output_names: &[&str],
        id: Option<String>,
    ) -> Result<InferResult> {
        let proto_inputs: Vec<client::InferInput> = inputs
            .into_iter()
            .map(|i| client::InferInput::fp32(i.name, i.shape, i.data))
            .collect();

        let options = client::InferOptions { id, timeout: None };

        let response = self
            .inner
            .infer(
                model,
                version,
                proto_inputs,
                output_names.iter().copied(),
                options,
            )
            .await?;

        Ok(InferResult {
            id: response.id,
            outputs: response
                .outputs
                .into_iter()
                .map(|o| {
                    let data_f32 = match o.data {
                        client::OutputData::Fp32(v) => v,
                        _ => vec![],
                    };
                    OutputTensor {
                        name: o.name,
                        shape: o.shape,
                        data_f32,
                    }
                })
                .collect(),
        })
    }

    // ── Repository ────────────────────────────────────────────────────────────

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

    pub async fn repository_model_load(&self, model: &str) -> Result<()> {
        self.inner
            .repository_model_load(model, None)
            .await
            .context("repository_model_load failed")
    }

    pub async fn repository_model_unload(&self, model: &str) -> Result<()> {
        self.inner
            .repository_model_unload(model, None)
            .await
            .context("repository_model_unload failed")
    }

    // ── System shared memory ──────────────────────────────────────────────────

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

    // ── CUDA shared memory ────────────────────────────────────────────────────

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

    // ── Settings ──────────────────────────────────────────────────────────────

    pub async fn trace_settings(
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

    pub fn mnist_blank_input(&self) -> InferInput {
        InferInput {
            name: self.config.input_name.clone(),
            shape: vec![1, 1, 28, 28],
            data: vec![0.0_f32; 784],
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub fn create_posix_shm(name: &str, size: i64) -> Result<()> {
    let cname = CString::new(name).context("invalid shared memory name")?;
    // SAFETY: standard POSIX shm_open call
    let fd = unsafe { libc::shm_open(cname.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o600_u32) };
    if fd < 0 {
        bail!("shm_open failed: {}", std::io::Error::last_os_error());
    }
    // SAFETY: fd is valid
    let ret = unsafe { libc::ftruncate(fd, size) };
    // SAFETY: fd is valid
    unsafe { libc::close(fd) };
    if ret < 0 {
        bail!("ftruncate failed: {}", std::io::Error::last_os_error());
    }
    Ok(())
}

pub fn remove_posix_shm(name: &str) -> Result<()> {
    let cname = CString::new(name).context("invalid shared memory name")?;
    // SAFETY: standard POSIX shm_unlink call
    let ret = unsafe { libc::shm_unlink(cname.as_ptr()) };
    if ret < 0 {
        bail!("shm_unlink failed: {}", std::io::Error::last_os_error());
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
