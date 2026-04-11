use std::collections::HashMap;

use app::app_config::AppConfig;
use app::error::{Error, Result};
use tracing::info;
use triton_ng_client::{InferInput, InferOptions, TritonClient, TritonClientConfig};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::init();

    let config = AppConfig::load()?;
    let model = config.model_name.as_str();
    let version = config.model_version.as_deref();

    let client = TritonClient::new(TritonClientConfig::new(&config.triton_url)).await?;

    // ── Server ────────────────────────────────────────────────────────────────

    let live = client.server_live(None).await?;
    let ready = client.server_ready(None).await?;
    let server = client.server_metadata(None).await?;
    info!(
        "server: {} v{} | live={live} ready={ready} | extensions: {}",
        server.name,
        server.version,
        server.extensions.join(", ")
    );

    // ── Repository ────────────────────────────────────────────────────────────

    let index = client.repository_index(false, None).await?;
    info!("model repository ({} entries):", index.len());
    for m in &index {
        info!("  {} v{} — {}", m.name, m.version, m.state);
    }

    // ── Model ─────────────────────────────────────────────────────────────────

    let model_ready = client.model_ready(model, version, None).await?;
    info!("{model} ready={model_ready}");

    let meta = client.model_metadata(model, version, None).await?;
    info!(
        "{} versions={:?} platform={}",
        meta.name, meta.versions, meta.platform
    );
    for t in &meta.inputs {
        info!("  input  {} {:?} {:?}", t.name, t.datatype, t.shape);
    }
    for t in &meta.outputs {
        info!("  output {} {:?} {:?}", t.name, t.datatype, t.shape);
    }

    if let Some(cfg) = client.model_config(model, version, None).await? {
        info!("config: name={} platform={}", cfg.name(), cfg.platform());
    }

    // ── Statistics before inference ───────────────────────────────────────────

    let stats_before = client.model_statistics(Some(model), version, None).await?;
    for s in &stats_before {
        info!(
            "stats before: {} v{} — inferences={} executions={}",
            s.name(),
            s.version(),
            s.inference_count(),
            s.execution_count()
        );
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    let input = meta.inputs.first().ok_or(Error::ModelHasNoInputs)?;
    let output = meta.outputs.first().ok_or(Error::ModelHasNoOutputs)?;
    let n_elements: usize = input.shape.iter().map(|&d| d as usize).product();

    let response = client
        .infer(
            model,
            version,
            [InferInput::fp32(
                &input.name,
                input.shape.clone(),
                vec![0.123f32; n_elements],
            )],
            [output.name.as_str()],
            InferOptions {
                id: Some("test-request-1".into()),
                timeout: None,
            },
        )
        .await?;

    info!("infer response id={:?}", response.id);
    for out in &response.outputs {
        info!("  {} {:?} shape={:?}", out.name, out.datatype, out.shape);
    }

    // ── Statistics after inference ────────────────────────────────────────────

    let stats_after = client.model_statistics(Some(model), version, None).await?;
    for s in &stats_after {
        info!(
            "stats after:  {} v{} — inferences={} executions={}",
            s.name(),
            s.version(),
            s.inference_count(),
            s.execution_count()
        );
    }

    // ── Shared memory status ──────────────────────────────────────────────────

    let sys_shm = client.system_shared_memory_status(None, None).await?;
    info!("system shared memory regions: {}", sys_shm.len());

    let cuda_shm = client.cuda_shared_memory_status(None, None).await?;
    info!("cuda shared memory regions: {}", cuda_shm.len());

    // ── Settings (read-only) ──────────────────────────────────────────────────

    let log = client.log_settings(HashMap::new(), None).await?;
    info!("log settings ({} entries):", log.len());
    for (k, v) in &log {
        info!("  {k} = {v:?}");
    }

    let trace = client.trace_settings(HashMap::new(), None, None).await?;
    info!("trace settings ({} entries):", trace.len());
    for (k, v) in &trace {
        info!("  {k} = {v:?}");
    }

    Ok(())
}
