mod common;

use std::collections::HashMap;

use anyhow::{Context, Result};
use common::config::Config;
use common::sut::{Sut, create_posix_shm, infer_result, remove_posix_shm, scrape_metrics};
use tonic::Code;
use triton_ng_client::Parameter;

#[tokio::test]
async fn server_is_live() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let live = sut.server_live().await?;
    assert!(live);
    Ok(())
}

#[tokio::test]
async fn server_is_ready() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let ready = sut.server_ready().await?;
    assert!(ready);
    Ok(())
}

#[tokio::test]
async fn server_metadata_contains_triton_name() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let info = sut.server_metadata().await?;
    assert_eq!(info.name, "triton");
    assert!(!info.version.is_empty());
    assert!(!info.extensions.is_empty());
    Ok(())
}

#[tokio::test]
async fn mnist_onnx_model_is_ready() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let ready = sut.model_ready(&config.mnist_model, None).await?;
    assert!(ready);
    Ok(())
}

#[tokio::test]
async fn mnist_onnx_model_metadata_describes_fp32_input_and_output() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let info = sut.model_metadata(&config.mnist_model, None).await?;

    assert_eq!(info.name, config.mnist_model);
    assert!(info.versions.contains(&"1".to_string()));
    assert_eq!(info.platform, "onnxruntime_onnx");

    let input = info
        .inputs
        .iter()
        .find(|t| t.name == config.input_name)
        .context("input tensor not found")?;
    assert_eq!(input.datatype, "FP32");
    assert_eq!(input.shape, vec![1, 1, 28, 28]);

    let output = info
        .outputs
        .iter()
        .find(|t| t.name == config.output_name)
        .context("output tensor not found")?;
    assert_eq!(output.datatype, "FP32");
    assert_eq!(output.shape, vec![1, 10]);

    Ok(())
}

#[tokio::test]
async fn mnist_onnx_model_config_identifies_onnxruntime_platform() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let cfg = sut.model_config(&config.mnist_model, None).await?;
    let cfg = cfg.context("model config was None")?;
    assert_eq!(cfg.name, config.mnist_model);
    assert_eq!(cfg.platform, "onnxruntime_onnx");
    Ok(())
}

#[tokio::test]
async fn model_statistics_are_returned_for_all_models() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let stats = sut.model_statistics(None, None).await?;
    assert!(!stats.is_empty());
    let mnist_stats = stats
        .iter()
        .find(|s| s.name == config.mnist_model)
        .context("mnist_onnx not found in statistics")?;
    assert_eq!(mnist_stats.version, "1");
    Ok(())
}

#[tokio::test]
async fn repository_index_lists_all_available_models() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let models = sut.repository_index(false).await?;
    let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
    assert!(
        names.contains(&config.mnist_model.as_str()),
        "{} missing from index",
        config.mnist_model
    );
    assert!(names.contains(&"mnist"), "mnist missing from index");
    Ok(())
}

#[tokio::test]
async fn ready_repository_index_contains_only_ready_models() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let models = sut.repository_index(true).await?;
    for model in &models {
        assert_eq!(
            model.state, "READY",
            "model {} has state {} in ready-only index",
            model.name, model.state
        );
    }
    assert!(!models.is_empty());
    Ok(())
}

#[tokio::test]
async fn model_becomes_ready_after_unload_and_reload() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    sut.repository_model_unload(&config.mnist_model).await?;
    sut.repository_model_load(&config.mnist_model).await?;
    let ready = sut.model_ready(&config.mnist_model, None).await?;
    assert!(ready);
    Ok(())
}

#[tokio::test]
async fn mnist_onnx_inference_returns_ten_class_scores() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let output_name = config.output_name.clone();
    let result = sut
        .infer(
            &config.mnist_model,
            None,
            vec![config.mnist_blank_input()],
            &[&output_name],
            None,
        )
        .await?;
    let output = result
        .outputs
        .iter()
        .find(|o| o.name == output_name)
        .context("output tensor not found in response")?;
    assert_eq!(output.shape, vec![1, 10]);
    assert_eq!(output.data_f32.len(), 10);
    Ok(())
}

#[tokio::test]
async fn custom_backend_returns_the_digit_with_the_highest_target_model_score() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let scores = sut
        .infer(
            &config.mnist_model,
            None,
            vec![config.mnist_blank_input()],
            &[&config.output_name],
            None,
        )
        .await?
        .outputs
        .into_iter()
        .find(|o| o.name == config.output_name)
        .context("scores not found in response")?
        .data_f32;
    let expected = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(digit, _)| digit as i32)
        .context("no scores")?;

    let digit = sut
        .infer(
            &config.backend_model,
            None,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            None,
        )
        .await?
        .outputs
        .into_iter()
        .find(|o| o.name == config.backend_output_name)
        .context("digit not found in response")?;
    assert_eq!(digit.shape, vec![1, 1]);
    assert_eq!(digit.data_i32, vec![expected]);
    Ok(())
}

#[tokio::test]
async fn custom_backend_answers_concurrent_requests() -> Result<()> {
    let mut tasks = tokio::task::JoinSet::new();
    for _ in 0..200 {
        tasks.spawn(async {
            let config = Config::load()?;
            let sut = Sut::new(&config.triton_url).await?;
            let output_name = config.backend_output_name.clone();
            let result = sut
                .infer(
                    &config.backend_model,
                    None,
                    vec![config.mnist_blank_input()],
                    &[&output_name],
                    None,
                )
                .await?;
            anyhow::ensure!(result.outputs.len() == 1, "expected one output");
            Ok(())
        });
    }
    while let Some(result) = tasks.join_next().await {
        result??;
    }
    Ok(())
}

/// `held_requests` and `cancelled_requests` reported by the custom backend.
async fn backend_counters(sut: &Sut, config: &Config) -> Result<(i64, i64)> {
    let output_name = config.backend_output_name.clone();
    let result = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&output_name],
            HashMap::new(),
        )
        .await?;
    let counter = |name: &str| match result.parameters.get(name) {
        Some(Parameter::Int(value)) => Ok(*value),
        other => anyhow::bail!("response parameter {name} is {other:?}"),
    };
    Ok((counter("held_requests")?, counter("cancelled_requests")?))
}

async fn wait_for_backend_counters(
    sut: &Sut,
    config: &Config,
    expected: impl Fn(i64, i64) -> bool,
) -> Result<()> {
    for _ in 0..10_000 {
        let (held, cancelled) = backend_counters(sut, config).await?;
        if expected(held, cancelled) {
            return Ok(());
        }
    }
    anyhow::bail!("backend counters did not reach the expected values")
}

#[tokio::test]
async fn custom_backend_answers_cancelled_request_as_cancelled() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let (_, cancelled_before) = backend_counters(&sut, &config).await?;

    let held = tokio::spawn(hold_request_until_cancelled());

    wait_for_backend_counters(&sut, &config, |held, _| held == 1).await?;
    held.abort();
    wait_for_backend_counters(&sut, &config, |held, cancelled| {
        held == 0 && cancelled == cancelled_before + 1
    })
    .await
}

async fn hold_request_until_cancelled() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    sut.infer_with_parameters(
        &config.backend_model,
        vec![config.mnist_blank_input()],
        &[&config.backend_output_name],
        HashMap::from([("hold_until_cancelled".to_string(), Parameter::Bool(true))]),
    )
    .await?;
    Ok(())
}

async fn infer_digit_with_memory(sut: &Sut, config: &Config, memory: &str) -> Result<(i32, i64)> {
    let result = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("memory".to_string(), Parameter::String(memory.to_string()))]),
        )
        .await?;
    let digit = result
        .outputs
        .into_iter()
        .find(|o| o.name == config.backend_output_name)
        .context("digit not found in response")?
        .data_i32;
    let allocated = match result
        .parameters
        .get(&format!("{memory}_buffers_allocated"))
    {
        Some(Parameter::Int(value)) => *value,
        other => anyhow::bail!("{memory}_buffers_allocated is {other:?}"),
    };
    Ok((digit[0], allocated))
}

async fn infer_digit(sut: &Sut, config: &Config) -> Result<i32> {
    let digit = sut
        .infer(
            &config.backend_model,
            None,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            None,
        )
        .await?
        .outputs
        .into_iter()
        .find(|o| o.name == config.backend_output_name)
        .context("digit not found in response")?
        .data_i32;
    Ok(digit[0])
}

#[tokio::test]
async fn custom_backend_answers_from_cached_ram_buffers() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let expected = infer_digit(&sut, &config).await?;

    let (digit, allocated) = infer_digit_with_memory(&sut, &config, "ram").await?;
    let (_, allocated_after_reuse) = infer_digit_with_memory(&sut, &config, "ram").await?;
    let (_, allocated_at_last) = infer_digit_with_memory(&sut, &config, "ram").await?;

    assert_eq!(digit, expected);
    assert!(allocated >= 1);
    assert_eq!(allocated_after_reuse, allocated);
    assert_eq!(allocated_at_last, allocated);
    Ok(())
}

#[tokio::test]
async fn custom_backend_answers_from_cached_pinned_buffers() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let expected = infer_digit(&sut, &config).await?;

    let (digit, allocated) = infer_digit_with_memory(&sut, &config, "pinned").await?;
    let (_, allocated_after_reuse) = infer_digit_with_memory(&sut, &config, "pinned").await?;
    let (_, allocated_at_last) = infer_digit_with_memory(&sut, &config, "pinned").await?;

    assert_eq!(digit, expected);
    assert!(allocated >= 1);
    assert_eq!(allocated_after_reuse, allocated);
    assert_eq!(allocated_at_last, allocated);
    Ok(())
}

#[tokio::test]
async fn custom_backend_reuses_cached_gpu_output_buffers() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let expected = infer_digit(&sut, &config).await?;

    let (digit, allocated) = infer_digit_with_memory(&sut, &config, "gpu").await?;
    let (_, allocated_after_reuse) = infer_digit_with_memory(&sut, &config, "gpu").await?;
    let (_, allocated_at_last) = infer_digit_with_memory(&sut, &config, "gpu").await?;

    assert_eq!(digit, expected);
    assert!(allocated >= 1);
    assert_eq!(allocated_after_reuse, allocated);
    assert_eq!(allocated_at_last, allocated);
    Ok(())
}

#[tokio::test]
async fn custom_backend_passes_a_gpu_output_as_the_input_of_two_requests() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let expected = infer_digit(&sut, &config).await?;

    let digit = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([(
                "memory".to_string(),
                Parameter::String("gpu_input".to_string()),
            )]),
        )
        .await?
        .outputs
        .into_iter()
        .find(|o| o.name == config.backend_output_name)
        .context("digit not found in response")?
        .data_i32;

    assert_eq!(digit, vec![expected]);
    Ok(())
}

#[tokio::test]
async fn custom_backend_fails_output_into_buffer_of_wrong_size() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let error = infer_digit_with_memory(&sut, &config, "wrong_size")
        .await
        .err()
        .context("output into a buffer of the wrong size succeeded")?;

    assert!(
        format!("{error:#}").contains("has 40 bytes, its buffer has 4"),
        "{error:#}"
    );
    Ok(())
}

async fn grpc_code_of_failed_inference(
    sut: &Sut,
    config: &Config,
    parameter: &str,
    value: &str,
) -> Result<Code> {
    let error = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([(parameter.to_string(), Parameter::String(value.to_string()))]),
        )
        .await
        .err()
        .context("the inference succeeded")?;
    match error.downcast_ref::<triton_ng_client::Error>() {
        Some(triton_ng_client::Error::Grpc(status)) => Ok(status.code()),
        _ => anyhow::bail!("not a gRPC error: {error:#}"),
    }
}

#[tokio::test]
async fn custom_backend_answers_with_the_requested_error_code() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    for (code, expected) in [
        ("INTERNAL", Code::Internal),
        ("NOT_FOUND", Code::NotFound),
        ("INVALID_ARG", Code::InvalidArgument),
        ("UNAVAILABLE", Code::Unavailable),
        ("UNSUPPORTED", Code::Unimplemented),
        ("ALREADY_EXISTS", Code::AlreadyExists),
        ("CANCELLED", Code::Cancelled),
    ] {
        let actual = grpc_code_of_failed_inference(&sut, &config, "error_code", code).await?;
        assert_eq!(actual, expected, "{code}");
    }
    Ok(())
}

#[tokio::test]
async fn custom_backend_passes_on_the_code_of_a_failed_bls_call() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let code = grpc_code_of_failed_inference(&sut, &config, "infer_model", "missing_model").await?;

    assert_eq!(code, Code::NotFound);
    Ok(())
}

async fn wait_for_model_state(sut: &Sut, model: &str, state: &str) -> Result<()> {
    for _ in 0..10_000 {
        let models = sut.repository_index(false).await?;
        if models.iter().any(|m| m.name == model && m.state == state) {
            return Ok(());
        }
    }
    anyhow::bail!("model {model} did not reach the {state} state")
}

#[tokio::test]
async fn custom_backend_model_unloads_only_after_its_held_request_is_answered() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let held = tokio::spawn(hold_request_until_cancelled());
    wait_for_backend_counters(&sut, &config, |held, _| held == 1).await?;

    sut.repository_model_unload(&config.backend_model).await?;
    let unloading = wait_for_model_state(&sut, &config.backend_model, "UNLOADING").await;
    held.abort();
    let unloaded = wait_for_model_state(&sut, &config.backend_model, "UNAVAILABLE").await;
    sut.repository_model_load(&config.backend_model).await?;

    unloading?;
    unloaded?;
    assert!(sut.model_ready(&config.backend_model, None).await?);
    Ok(())
}

#[tokio::test]
async fn inference_request_id_is_echoed_in_response() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let output_name = config.output_name.clone();
    let request_id = "test-echo-id-42".to_string();
    let result = sut
        .infer(
            &config.mnist_model,
            None,
            vec![config.mnist_blank_input()],
            &[&output_name],
            Some(request_id.clone()),
        )
        .await?;
    assert_eq!(result.id, request_id);
    Ok(())
}

#[tokio::test]
async fn inference_count_increases_after_successful_inference() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let output_name = config.output_name.clone();
    let model = config.mnist_model.clone();

    let before = sut.model_statistics(Some(&model), None).await?;
    let count_before = before
        .iter()
        .find(|s| s.name == model)
        .map(|s| s.inference_count)
        .unwrap_or(0);

    sut.infer(
        &model,
        None,
        vec![config.mnist_blank_input()],
        &[&output_name],
        None,
    )
    .await?;

    let after = sut.model_statistics(Some(&model), None).await?;
    let count_after = after
        .iter()
        .find(|s| s.name == model)
        .map(|s| s.inference_count)
        .unwrap_or(0);

    assert!(
        count_after > count_before,
        "inference_count did not increase: before={count_before}, after={count_after}"
    );
    Ok(())
}

#[tokio::test]
async fn system_shared_memory_status_is_empty_when_no_regions_are_registered() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    sut.system_shared_memory_unregister(None).await?;
    let regions = sut.system_shared_memory_status(None).await?;
    assert!(regions.is_empty());
    Ok(())
}

#[tokio::test]
async fn system_shared_memory_region_can_be_registered_and_unregistered() -> Result<()> {
    const SHM_KEY: &str = "/triton_ng_test_shm";
    const REGION_NAME: &str = "triton_ng_test_region";
    const BYTE_SIZE: u64 = 64;

    create_posix_shm(SHM_KEY, BYTE_SIZE as i64)?;

    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    sut.system_shared_memory_register(REGION_NAME, SHM_KEY, 0, BYTE_SIZE)
        .await?;

    let regions = sut.system_shared_memory_status(None).await?;
    let region = regions
        .get(REGION_NAME)
        .context("registered region not found in status")?;
    assert_eq!(region.key, SHM_KEY);
    assert_eq!(region.offset, 0);
    assert_eq!(region.byte_size, BYTE_SIZE);

    sut.system_shared_memory_unregister(Some(REGION_NAME))
        .await?;

    let regions_after = sut.system_shared_memory_status(None).await?;
    assert!(!regions_after.contains_key(REGION_NAME));

    remove_posix_shm(SHM_KEY)?;
    Ok(())
}

#[tokio::test]
async fn cuda_shared_memory_status_is_empty_when_no_regions_are_registered() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    sut.cuda_shared_memory_unregister(None).await?;
    let regions = sut.cuda_shared_memory_status(None).await?;
    assert!(regions.is_empty());
    Ok(())
}

#[tokio::test]
async fn cuda_shared_memory_register_returns_error_for_invalid_handle() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let result = sut
        .cuda_shared_memory_register("invalid_region", vec![0u8; 64], 0, 64)
        .await;
    assert!(
        result.is_err(),
        "expected an error for an invalid CUDA handle, but got Ok"
    );
    Ok(())
}

#[tokio::test]
async fn trace_settings_are_readable() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let settings = sut.trace_settings(HashMap::new(), None).await?;
    assert!(!settings.is_empty());
    Ok(())
}

#[tokio::test]
async fn log_settings_are_readable() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let settings = sut.log_settings(HashMap::new()).await?;
    assert!(!settings.is_empty());
    Ok(())
}

async fn echoed_metadata(
    sut: &Sut,
    config: &Config,
    id: &str,
    parameters: &[(&str, Parameter)],
) -> Result<HashMap<String, Parameter>> {
    let result = sut
        .infer_with_id_and_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            id,
            parameters
                .iter()
                .map(|(name, value)| (name.to_string(), value.clone()))
                .collect(),
        )
        .await?;
    anyhow::ensure!(result.id == id, "response id is {}", result.id);
    Ok(result.parameters)
}

fn string(value: &str) -> Parameter {
    Parameter::String(value.to_string())
}

#[tokio::test]
async fn custom_backend_reads_the_metadata_of_a_request() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let echoed = echoed_metadata(
        &sut,
        &config,
        "metadata-1",
        &[
            ("echo_metadata", Parameter::Bool(true)),
            ("sequence_id", Parameter::Int(42)),
            ("sequence_start", Parameter::Bool(true)),
            ("sequence_end", Parameter::Bool(false)),
            ("timeout", Parameter::Int(5_000_000)),
        ],
    )
    .await?;

    assert_eq!(echoed.get("request_id"), Some(&string("metadata-1")));
    assert_eq!(echoed.get("correlation_id"), Some(&string("42")));
    assert_eq!(echoed.get("sequence_start"), Some(&Parameter::Bool(true)));
    assert_eq!(echoed.get("sequence_end"), Some(&Parameter::Bool(false)));
    assert_eq!(echoed.get("timeout_us"), Some(&Parameter::Int(5_000_000)));
    Ok(())
}

#[tokio::test]
async fn custom_backend_reads_a_string_correlation_id_and_no_metadata() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let echoed = echoed_metadata(
        &sut,
        &config,
        "metadata-2",
        &[
            ("echo_metadata", Parameter::Bool(true)),
            ("sequence_id", string("call-7")),
        ],
    )
    .await?;
    assert_eq!(echoed.get("correlation_id"), Some(&string("call-7")));
    assert_eq!(echoed.get("sequence_start"), Some(&Parameter::Bool(false)));
    assert_eq!(echoed.get("timeout_us"), None);

    let bare = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("echo_metadata".to_string(), Parameter::Bool(true))]),
        )
        .await?
        .parameters;
    assert_eq!(bare.get("request_id"), None);
    assert_eq!(bare.get("correlation_id"), None);
    Ok(())
}

#[tokio::test]
async fn custom_backend_reads_its_backend_config() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let echoed = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("echo_backend_config".to_string(), Parameter::Bool(true))]),
        )
        .await?
        .parameters;
    assert_eq!(echoed.get("greeting"), Some(&string("hello")));
    assert!(echoed.contains_key("backend-directory"), "{echoed:?}");
    Ok(())
}

#[tokio::test]
async fn model_without_instance_group_gets_the_one_the_backend_prefers() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let cfg = sut
        .model_config(&config.stream_model, None)
        .await?
        .context("model config was None")?;
    assert_eq!(
        cfg.instance_groups,
        vec![(2, 2)],
        "expected 2 instances of KIND_CPU (2)"
    );
    Ok(())
}

#[tokio::test]
async fn custom_backend_passes_metadata_and_parameters_to_a_bls_request() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let echoed = echoed_metadata(
        &sut,
        &config,
        "metadata-3",
        &[
            ("forward_metadata", Parameter::Bool(true)),
            ("sequence_id", Parameter::Int(41)),
            ("sequence_start", Parameter::Bool(true)),
            ("timeout", Parameter::Int(3_000_000)),
        ],
    )
    .await?;

    assert_eq!(echoed.get("request_id"), Some(&string("metadata-3/bls")));
    assert_eq!(echoed.get("correlation_id"), Some(&string("42")));
    assert_eq!(echoed.get("sequence_start"), Some(&Parameter::Bool(false)));
    assert_eq!(echoed.get("sequence_end"), Some(&Parameter::Bool(true)));
    assert_eq!(echoed.get("timeout_us"), Some(&Parameter::Int(6_000_000)));
    assert_eq!(echoed.get("forwarded"), Some(&string("by the backend")));
    Ok(())
}

/// The value of the series `name{model="mnist"}`, `None` when it is not
/// exported.
fn mnist_metric(metrics: &str, name: &str) -> Result<Option<f64>> {
    let prefix = format!("{name}{{model=\"mnist\"}} ");
    metrics
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .map(|value| {
            value
                .trim()
                .parse::<f64>()
                .context(format!("{name} is {value}"))
        })
        .transpose()
}

#[tokio::test]
async fn custom_backend_metrics_are_exported_by_triton() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    infer_digit(&sut, &config).await?;
    let before = scrape_metrics(&config.metrics_address)?;

    infer_digit(&sut, &config).await?;
    let after = scrape_metrics(&config.metrics_address)?;

    let answered = "triton_ng_example_answered_total";
    let observed = "triton_ng_example_bls_seconds_count";
    let held = "triton_ng_example_held_requests";
    for name in [answered, observed, held] {
        assert!(
            mnist_metric(&after, name)?.is_some(),
            "{name} is missing:\n{after}"
        );
    }
    assert!(mnist_metric(&after, answered)? > mnist_metric(&before, answered)?);
    assert!(mnist_metric(&after, observed)? > mnist_metric(&before, observed)?);
    assert!(
        after.contains("# TYPE triton_ng_example_bls_seconds histogram"),
        "{after}"
    );
    assert!(
        after.contains("triton_ng_example_bls_seconds_bucket{model=\"mnist\",le=\"0.01\"}"),
        "{after}"
    );
    Ok(())
}

async fn set_mnist_tracing(sut: &Sut, level: &str) -> Result<()> {
    let settings = [
        ("trace_level", level),
        ("trace_rate", "1"),
        ("log_frequency", "1"),
    ];
    sut.trace_settings(
        settings
            .into_iter()
            .map(|(name, value)| (name.to_string(), vec![value.to_string()]))
            .collect(),
        Some("mnist"),
    )
    .await?;
    Ok(())
}

/// Every trace entry written into the trace files so far.
fn trace_entries(config: &Config) -> Result<Vec<serde_json::Value>> {
    let mut entries = Vec::new();
    let Ok(dir) = std::fs::read_dir(config.trace_dir()) else {
        return Ok(entries);
    };
    for file in dir {
        let path = file?.path();
        let content = std::fs::read_to_string(&path)?;
        if let Ok(serde_json::Value::Array(values)) = serde_json::from_str(&content) {
            entries.extend(values);
        }
    }
    Ok(entries)
}

fn has_traced_child(entries: &[serde_json::Value], request_id: &str) -> bool {
    let parents: Vec<_> = entries
        .iter()
        .filter(|entry| entry["request_id"] == request_id && entry["model_name"] == "mnist")
        .filter_map(|entry| entry["id"].as_u64())
        .collect();
    entries.iter().any(|entry| {
        entry["model_name"] == "mnist_onnx"
            && entry["parent_id"]
                .as_u64()
                .is_some_and(|parent| parents.contains(&parent))
    })
}

#[tokio::test]
async fn bls_call_is_traced_as_a_child_of_its_client_request() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let request_id = format!(
        "traced-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    );

    set_mnist_tracing(&sut, "TIMESTAMPS").await?;
    let inferred = sut
        .infer_with_id_and_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            &request_id,
            HashMap::new(),
        )
        .await;
    set_mnist_tracing(&sut, "OFF").await?;
    inferred?;

    for _ in 0..100 {
        if has_traced_child(&trace_entries(&config)?, &request_id) {
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    anyhow::bail!("no trace of mnist_onnx with request {request_id} as the parent")
}

#[tokio::test]
async fn request_answered_before_its_traced_bls_call_is_released_after_it() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let request_id = format!(
        "answered-first-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    );

    set_mnist_tracing(&sut, "TIMESTAMPS").await?;
    let mut tasks = tokio::task::JoinSet::new();
    for index in 0..50 {
        let request_id = format!("{request_id}-{index}");
        tasks.spawn(async move {
            let config = Config::load()?;
            let sut = Sut::new(&config.triton_url).await?;
            sut.infer_with_id_and_parameters(
                &config.backend_model,
                vec![config.mnist_blank_input()],
                &[&config.backend_output_name],
                &request_id,
                HashMap::from([("answer_before_bls".to_string(), Parameter::Bool(true))]),
            )
            .await
        });
    }
    let mut answered = Ok(());
    while let Some(result) = tasks.join_next().await {
        if let Err(e) = result.map_err(anyhow::Error::from).and_then(|r| r) {
            answered = Err(e);
        }
    }
    set_mnist_tracing(&sut, "OFF").await?;
    answered?;

    assert!(sut.server_live().await?);
    let first = format!("{request_id}-0");
    for _ in 0..100 {
        if has_traced_child(&trace_entries(&config)?, &first) {
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    anyhow::bail!("no trace of mnist_onnx with request {first} as the parent")
}

/// Digits streamed by the decoupled model for `images` blank images, and the
/// error the stream ended with, if any.
async fn streamed_digits(
    sut: &Sut,
    config: &Config,
    images: usize,
    parameter: Option<&str>,
) -> Result<(Vec<i32>, Option<triton_ng_client::Error>)> {
    let parameters = parameter
        .map(|name| HashMap::from([(name.to_string(), Parameter::Bool(true))]))
        .unwrap_or_default();
    let mut stream = sut
        .infer_stream(
            &config.stream_model,
            vec![config.mnist_blank_images(images)],
            &[&config.backend_output_name],
            parameters,
        )
        .await?;
    let mut digits = Vec::new();
    while let Some(response) = stream.next().await {
        match response {
            Ok(response) => {
                let output = infer_result(response)
                    .outputs
                    .into_iter()
                    .find(|o| o.name == config.backend_output_name)
                    .context("digit not found in response")?;
                anyhow::ensure!(output.shape == vec![1, 1], "shape {:?}", output.shape);
                digits.extend(output.data_i32);
            }
            Err(error) => return Ok((digits, Some(error))),
        }
    }
    Ok((digits, None))
}

#[tokio::test]
async fn decoupled_model_streams_a_digit_per_image() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let expected = infer_digit(&sut, &config).await?;

    let (digits, error) = streamed_digits(&sut, &config, 3, None).await?;

    assert!(error.is_none(), "{error:?}");
    assert_eq!(digits, vec![expected; 3]);
    Ok(())
}

#[tokio::test]
async fn decoupled_model_finishes_a_stream_without_responses() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let (digits, error) = streamed_digits(&sut, &config, 0, None).await?;

    assert!(error.is_none(), "{error:?}");
    assert!(digits.is_empty());
    Ok(())
}

#[tokio::test]
async fn stream_whose_sender_is_dropped_ends_with_an_error() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let (digits, error) = streamed_digits(&sut, &config, 2, Some("unfinished")).await?;

    assert_eq!(digits.len(), 2);
    let error = error.context("the stream ended without an error")?;
    assert!(
        error
            .to_string()
            .contains("did not produce a final response"),
        "{error}"
    );
    Ok(())
}

#[tokio::test]
async fn request_of_a_model_that_is_not_decoupled_gets_no_sender() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let error = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("into_sender".to_string(), Parameter::Bool(true))]),
        )
        .await
        .err()
        .context("the inference succeeded")?;

    assert!(
        format!("{error:#}").contains("only requests of a decoupled model"),
        "{error:#}"
    );
    Ok(())
}

/// `held_requests` and `cancelled_requests` reported by the decoupled model.
async fn stream_counters(sut: &Sut, config: &Config) -> Result<(i64, i64)> {
    let mut stream = sut
        .infer_stream(
            &config.stream_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::new(),
        )
        .await?;
    let response = stream.next().await.context("the stream is empty")??;
    let counter = |name: &str| match response.parameters.get(name) {
        Some(Parameter::Int(value)) => Ok(*value),
        other => anyhow::bail!("response parameter {name} is {other:?}"),
    };
    Ok((counter("held_requests")?, counter("cancelled_requests")?))
}

#[tokio::test]
async fn stream_cancelled_by_the_client_is_answered_as_cancelled() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let (_, cancelled_before) = stream_counters(&sut, &config).await?;

    let mut stream = sut
        .infer_stream(
            &config.stream_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("hold_until_cancelled".to_string(), Parameter::Bool(true))]),
        )
        .await?;
    stream.next().await.context("the stream is empty")??;
    for _ in 0..10_000 {
        if stream_counters(&sut, &config).await?.0 == 1 {
            break;
        }
    }
    drop(stream);

    for _ in 0..10_000 {
        let (held, cancelled) = stream_counters(&sut, &config).await?;
        if held == 0 && cancelled == cancelled_before + 1 {
            return Ok(());
        }
    }
    anyhow::bail!("the backend did not see the cancellation")
}

async fn infer_backend(
    sut: &Sut,
    config: &Config,
    parameters: &[(&str, Parameter)],
) -> Result<common::dto::InferResult> {
    sut.infer_with_parameters(
        &config.backend_model,
        vec![config.mnist_blank_input()],
        &[&config.backend_output_name],
        parameters
            .iter()
            .map(|(name, value)| (name.to_string(), value.clone()))
            .collect(),
    )
    .await
}

fn response_parameter(result: &common::dto::InferResult, name: &str) -> Result<Parameter> {
    result
        .parameters
        .get(name)
        .cloned()
        .with_context(|| format!("response parameter {name} is missing"))
}

#[tokio::test]
async fn bls_stream_returns_every_response_of_a_decoupled_model() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let expected = infer_digit(&sut, &config).await?;

    let result = infer_backend(&sut, &config, &[("stream_images", Parameter::Int(3))]).await?;

    assert_eq!(
        response_parameter(&result, "streamed_digits")?,
        Parameter::Int(3)
    );
    let digit = result
        .outputs
        .into_iter()
        .find(|o| o.name == config.backend_output_name)
        .context("digit not found in response")?;
    assert_eq!(digit.data_i32, vec![expected]);
    Ok(())
}

#[tokio::test]
async fn bls_stream_of_a_decoupled_model_without_responses_is_empty() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let result = infer_backend(&sut, &config, &[("stream_images", Parameter::Int(0))]).await?;

    assert_eq!(
        response_parameter(&result, "streamed_digits")?,
        Parameter::Int(0)
    );
    Ok(())
}

async fn infer_async_error(sut: &Sut, config: &Config, images: i64) -> Result<String> {
    let error = infer_backend(
        sut,
        config,
        &[
            ("stream_images", Parameter::Int(images)),
            ("via_infer_async", Parameter::Bool(true)),
        ],
    )
    .await
    .err()
    .context("the inference succeeded")?;
    Ok(format!("{error:#}"))
}

#[tokio::test]
async fn infer_async_fails_when_a_decoupled_model_sends_several_responses() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let error = infer_async_error(&sut, &config, 2).await?;

    assert!(error.contains("more than one response"), "{error}");
    Ok(())
}

#[tokio::test]
async fn infer_async_fails_when_a_decoupled_model_sends_no_response() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let error = infer_async_error(&sut, &config, 0).await?;

    assert!(error.contains("sent no response"), "{error}");
    Ok(())
}

async fn wait_for_stream_cancellation(sut: &Sut, config: &Config, before: i64) -> Result<()> {
    for _ in 0..10_000 {
        let (held, cancelled) = stream_counters(sut, config).await?;
        if held == 0 && cancelled == before + 1 {
            return Ok(());
        }
    }
    anyhow::bail!("the decoupled model did not see the cancellation")
}

#[tokio::test]
async fn cancelled_bls_stream_ends_with_the_cancellation_of_the_model() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let (_, before) = stream_counters(&sut, &config).await?;

    let result = infer_backend(&sut, &config, &[("cancel_stream", string("cancel"))]).await?;

    assert_eq!(
        response_parameter(&result, "stream_ended_with")?,
        string("CANCELLED")
    );
    wait_for_stream_cancellation(&sut, &config, before).await
}

#[tokio::test]
async fn dropped_bls_stream_cancels_its_request() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let (_, before) = stream_counters(&sut, &config).await?;

    let result = infer_backend(&sut, &config, &[("cancel_stream", string("drop"))]).await?;

    assert_eq!(
        response_parameter(&result, "stream_ended_with")?,
        string("dropped")
    );
    wait_for_stream_cancellation(&sut, &config, before).await
}

#[tokio::test]
async fn cancelled_bls_request_resolves_to_cancelled() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let error = infer_backend(&sut, &config, &[("cancel_bls", Parameter::Bool(true))])
        .await
        .err()
        .context("the inference succeeded")?;

    match error.downcast_ref::<triton_ng_client::Error>() {
        Some(triton_ng_client::Error::Grpc(status)) => {
            assert_eq!(status.code(), Code::Cancelled, "{status}")
        }
        _ => anyhow::bail!("not a gRPC error: {error:#}"),
    }
    wait_for_backend_counters(&sut, &config, |held, _| held == 0).await
}

#[tokio::test]
async fn unary_inference_of_a_decoupled_model_is_rejected() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let error = sut
        .infer(
            &config.stream_model,
            None,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            None,
        )
        .await
        .err()
        .context("the inference succeeded")?;

    assert!(format!("{error:#}").contains("decoupled"), "{error:#}");
    Ok(())
}
