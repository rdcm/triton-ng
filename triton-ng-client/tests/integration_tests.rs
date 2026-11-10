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

/// Successful requests and executions of the custom backend model.
async fn backend_statistics(sut: &Sut, config: &Config) -> Result<(u64, u64)> {
    let stats = sut
        .model_statistics(Some(&config.backend_model), None)
        .await?;
    let stats = stats
        .iter()
        .find(|s| s.name == config.backend_model)
        .context("backend model not found in statistics")?;
    Ok((stats.inference_count, stats.execution_count))
}

#[tokio::test]
async fn custom_backend_reports_its_requests_and_batches_to_the_statistics() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let (requests_before, batches_before) = backend_statistics(&sut, &config).await?;

    infer_digit(&sut, &config).await?;

    let (requests, batches) = backend_statistics(&sut, &config).await?;
    assert!(
        requests > requests_before,
        "{requests} <= {requests_before}"
    );
    assert!(batches > batches_before, "{batches} <= {batches_before}");
    Ok(())
}

#[tokio::test]
async fn custom_backend_reports_the_memory_the_model_and_its_instances_hold() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let usage = sut.model_memory_usage(&config.backend_model).await?;

    let cpu: Vec<_> = usage
        .iter()
        .filter(|usage| usage.memory_type == "CPU" && usage.id == 0)
        .collect();
    assert_eq!(
        cpu.len(),
        1,
        "the model and its instances report one CPU entry"
    );
    assert!(cpu[0].byte_size > 0);
    Ok(())
}

/// The fixture of `models/memory_usage` reports two entries of one memory type
/// and id and a pinned buffer that it really allocates; its sizes are the
/// constants of the example backend.
#[tokio::test]
async fn memory_reports_are_summed_per_memory_type_and_keep_their_kind() -> Result<()> {
    let model = "memory_usage";
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    sut.repository_model_load(model).await?;
    let usage = sut.model_memory_usage(model).await;
    let unloaded = sut.repository_model_unload(model).await;
    let usage = usage?;
    unloaded?;

    let cpu: Vec<_> = usage
        .iter()
        .filter(|usage| usage.memory_type == "CPU" && usage.id == 0)
        .collect();
    assert_eq!(
        cpu.len(),
        1,
        "the two reported CPU entries are summed into one"
    );
    assert_eq!(cpu[0].byte_size, 111 + 222);

    let pinned: u64 = usage
        .iter()
        .filter(|usage| usage.memory_type == "CPU_PINNED")
        .map(|usage| usage.byte_size)
        .sum();
    assert_eq!(
        pinned, 4096,
        "the instance reports the pinned buffer it holds"
    );
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
async fn custom_backend_reports_the_response_statistics_it_enables() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    // A response the backend does not opt in for is not recorded.
    let before = sut.response_statistics(&config.backend_model).await?;
    infer_backend(&sut, &config, &[]).await?;
    let unchanged = sut.response_statistics(&config.backend_model).await?;
    anyhow::ensure!(unchanged == before, "{unchanged} != {before}");

    sut.infer_with_parameters(
        &config.backend_model,
        vec![config.mnist_blank_input()],
        &[&config.backend_output_name],
        HashMap::from([("enable_statistics".to_string(), Parameter::Bool(true))]),
    )
    .await?;
    let after = sut.response_statistics(&config.backend_model).await?;
    anyhow::ensure!(after == before + 1, "{after} != {before} + 1");

    // The same opt-in on a response of a stream: the decoupled model reports
    // one entry per response it sends.
    let before = sut.response_statistics(&config.stream_model).await?;
    let (digits, error) = streamed_digits(&sut, &config, 2, None).await?;
    anyhow::ensure!(error.is_none(), "{error:?}");
    anyhow::ensure!(digits.len() == 2, "two responses: {digits:?}");
    let unchanged = sut.response_statistics(&config.stream_model).await?;
    anyhow::ensure!(unchanged == before, "{unchanged} != {before}");

    let (digits, error) = streamed_digits(&sut, &config, 2, Some("enable_statistics")).await?;
    anyhow::ensure!(error.is_none(), "{error:?}");
    anyhow::ensure!(digits.len() == 2, "two responses: {digits:?}");
    let after = sut.response_statistics(&config.stream_model).await?;
    anyhow::ensure!(after == before + 2, "{after} != {before} + 2");
    Ok(())
}

#[tokio::test]
async fn response_statistics_count_a_response_enabled_twice_once() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let before = sut.response_statistics(&config.backend_model).await?;
    let ns_before = sut.successful_response_ns(&config.backend_model).await?;
    let work_micros = 50_000_i64;
    sut.infer_with_parameters(
        &config.backend_model,
        vec![config.mnist_blank_input()],
        &[&config.backend_output_name],
        HashMap::from([
            ("enable_statistics".to_string(), Parameter::Bool(true)),
            ("hold_micros".to_string(), Parameter::Int(work_micros)),
        ]),
    )
    .await?;

    let after = sut.response_statistics(&config.backend_model).await?;
    anyhow::ensure!(after == before + 1, "{after} != {before} + 1");
    // The interval starts when the response was created, so the work done before
    // the opt-in is inside it.
    let ns_after = sut.successful_response_ns(&config.backend_model).await?;
    anyhow::ensure!(
        ns_after >= ns_before + u64::try_from(work_micros)? * 1_000,
        "{ns_after} - {ns_before} is shorter than {work_micros}us"
    );
    Ok(())
}

#[tokio::test]
async fn response_statistics_count_a_failed_response() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let (failed_before, _) = sut
        .failed_and_cancelled_responses(&config.backend_model)
        .await?;
    // The client asks for the statistics of the response, and the backend opts in
    // for the error response it answers with.
    let error = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([
                ("error_code".to_string(), string("INVALID_ARG")),
                ("enable_statistics".to_string(), Parameter::Bool(true)),
            ]),
        )
        .await
        .err()
        .context("the inference succeeded")?;
    anyhow::ensure!(
        format!("{error:#}").contains("error requested by the client"),
        "{error:#}"
    );

    let (failed_after, _) = sut
        .failed_and_cancelled_responses(&config.backend_model)
        .await?;
    anyhow::ensure!(
        failed_after == failed_before + 1,
        "{failed_after} != {failed_before} + 1"
    );
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
async fn model_with_a_config_completed_by_the_backend_answers() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let digit = sut
        .infer(
            "mnist_auto",
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
    assert_eq!(digit.data_i32.len(), 1);
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
    let (_, statistics_before) = sut
        .failed_and_cancelled_responses(&config.backend_model)
        .await?;

    let held = tokio::spawn(hold_request_until_cancelled());

    wait_for_backend_counters(&sut, &config, |held, _| held == 1).await?;
    held.abort();
    wait_for_backend_counters(&sut, &config, |held, cancelled| {
        held == 0 && cancelled == cancelled_before + 1
    })
    .await?;

    // The cancelled request is answered with an error the statistics record as a
    // cancellation, see `FinalResponse::send_error`.
    let (_, statistics_after) = sut
        .failed_and_cancelled_responses(&config.backend_model)
        .await?;
    anyhow::ensure!(
        statistics_after == statistics_before + 1,
        "{statistics_after} != {statistics_before} + 1"
    );
    Ok(())
}

async fn hold_request_until_cancelled() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    sut.infer_with_parameters(
        &config.backend_model,
        vec![config.mnist_blank_input()],
        &[&config.backend_output_name],
        HashMap::from([
            ("hold_until_cancelled".to_string(), Parameter::Bool(true)),
            // The backend opts in for the cancelled response it answers with.
            ("enable_statistics".to_string(), Parameter::Bool(true)),
        ]),
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
async fn custom_backend_passes_a_gpu_input_of_its_request_on_without_a_copy() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let expected = infer_digit(&sut, &config).await?;

    let digit = infer_backend(
        &sut,
        &config,
        &[("memory", Parameter::String("gpu_forward".to_string()))],
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

async fn infer_digit_with_shared_input(sut: &Sut, config: &Config, keep: bool) -> Result<i32> {
    let result = infer_backend(sut, config, &[("shared_input", Parameter::Bool(keep))]).await?;
    let digit = result
        .outputs
        .into_iter()
        .find(|o| o.name == config.backend_output_name)
        .context("digit not found in response")?
        .data_i32;
    Ok(digit[0])
}

async fn infer_digit_keeping_shared_input() -> Result<i32> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    infer_digit_with_shared_input(&sut, &config, true).await
}

async fn kept_inputs(sut: &Sut, config: &Config) -> Result<i64> {
    let result = infer_backend(sut, config, &[]).await?;
    match response_parameter(&result, "kept_inputs")? {
        Parameter::Int(kept) => Ok(kept),
        other => anyhow::bail!("kept_inputs is {other:?}"),
    }
}

async fn wait_for_kept_inputs(sut: &Sut, config: &Config) -> Result<()> {
    let mut kept = 0;
    for _ in 0..10_000 {
        kept = kept_inputs(sut, config).await?;
        if kept == 1 {
            return Ok(());
        }
    }
    anyhow::bail!("the backend kept {kept} inputs of its requests")
}

#[tokio::test]
async fn custom_backend_passes_an_input_of_its_request_on_without_a_copy() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let expected = infer_digit(&sut, &config).await?;

    let digit = infer_digit_with_shared_input(&sut, &config, false).await?;

    assert_eq!(digit, expected);
    Ok(())
}

#[tokio::test]
async fn response_is_held_back_until_the_shared_input_of_its_request_is_dropped() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let expected = infer_digit(&sut, &config).await?;

    let held = tokio::spawn(infer_digit_keeping_shared_input());
    wait_for_kept_inputs(&sut, &config).await?;
    assert!(!held.is_finished());

    infer_backend(
        &sut,
        &config,
        &[("release_shared_inputs", Parameter::Bool(true))],
    )
    .await?;

    assert_eq!(held.await??, expected);
    Ok(())
}

async fn infer_digit_keeping_shared_input_with_statistics() -> Result<i32> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let result = infer_backend(
        &sut,
        &config,
        &[
            ("shared_input", Parameter::Bool(true)),
            ("enable_statistics", Parameter::Bool(true)),
        ],
    )
    .await?;
    let digit = result
        .outputs
        .into_iter()
        .find(|o| o.name == config.backend_output_name)
        .context("digit not found in response")?
        .data_i32;
    Ok(digit[0])
}

#[tokio::test]
async fn response_statistics_count_a_held_back_response_when_it_is_sent() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let before = sut.response_statistics(&config.backend_model).await?;
    let held = tokio::spawn(infer_digit_keeping_shared_input_with_statistics());
    wait_for_kept_inputs(&sut, &config).await?;

    // The response waits for the last shared input of its request, so nothing is
    // recorded yet.
    let unchanged = sut.response_statistics(&config.backend_model).await?;
    anyhow::ensure!(unchanged == before, "{unchanged} != {before}");

    infer_backend(
        &sut,
        &config,
        &[("release_shared_inputs", Parameter::Bool(true))],
    )
    .await?;
    held.await??;

    let after = sut.response_statistics(&config.backend_model).await?;
    anyhow::ensure!(after == before + 1, "{after} != {before} + 1");
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
async fn custom_backend_writes_its_activity_into_the_trace_of_a_request() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let activity = format!(
        "triton_ng_activity_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    );
    let parameters = HashMap::from([(
        "report_activity".to_string(),
        Parameter::String(activity.clone()),
    )]);

    // With tracing off the report is a no-op and the request still succeeds.
    sut.infer_with_parameters(
        &config.backend_model,
        vec![config.mnist_blank_input()],
        &[&config.backend_output_name],
        parameters.clone(),
    )
    .await
    .context("reporting an activity of an untraced request")?;

    set_mnist_tracing(&sut, "TIMESTAMPS").await?;
    let inferred = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            parameters,
        )
        .await;
    set_mnist_tracing(&sut, "OFF").await?;
    inferred?;

    // The trace collector writes the file after the request, so poll it.
    for _ in 0..100 {
        let named = trace_entries(&config)?
            .iter()
            .any(|entry| entry.to_string().contains(&activity));
        if named {
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    anyhow::bail!("no trace entry names the activity {activity}")
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
async fn custom_backend_shares_its_state_and_reads_its_artifacts() -> Result<()> {
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
    let backend_directory = echoed
        .get("backend-directory")
        .and_then(|value| match value {
            Parameter::String(directory) => Some(directory),
            _ => None,
        })
        .context("backend-directory is missing or not a string")?;
    assert_eq!(echoed.get("artifact_type"), Some(&string("FileSystem")));
    assert_eq!(
        echoed.get("artifact_directory"),
        Some(&string(&format!("{backend_directory}/custom_backend")))
    );
    Ok(())
}

#[tokio::test]
async fn custom_backend_shares_its_state_between_models() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    // One state counts both models: a state per model or per instance would
    // start over for the second one.
    let mut counted = Vec::new();
    for model in ["mnist", "mnist_auto"] {
        let echoed = sut
            .infer_with_parameters(
                model,
                vec![config.mnist_blank_input()],
                &[&config.backend_output_name],
                HashMap::from([("count_backend_state".to_string(), Parameter::Bool(true))]),
            )
            .await?
            .parameters;
        counted.push(match echoed.get("backend_state_count") {
            Some(Parameter::Int(count)) => *count,
            other => anyhow::bail!("no count from {model}: {other:?}"),
        });
    }
    anyhow::ensure!(
        counted[1] > counted[0],
        "the counter of the second model continues the first: {counted:?}"
    );
    Ok(())
}

#[tokio::test]
async fn custom_backend_reports_the_details_of_its_instances() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let echoed = sut
        .infer_with_parameters(
            "instance_details",
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("echo_instance_details".to_string(), Parameter::Bool(true))]),
        )
        .await?
        .parameters;

    let mut details: Vec<&str> = echoed
        .iter()
        .filter(|(name, _)| name.starts_with("instance_"))
        .filter_map(|(_, value)| match value {
            Parameter::String(detail) => Some(detail.as_str()),
            _ => None,
        })
        .collect();
    details.sort();
    anyhow::ensure!(details.len() == 2, "two instances: {details:?}");

    let passive = details
        .iter()
        .find(|detail| detail.contains("passive=true"))
        .context("no passive instance")?;
    assert!(
        passive.contains("secondary_devices=[SecondaryDevice { kind: \"KIND_NVDLA\", id: 7 }]"),
        "a passive instance keeps its secondary device: {passive}"
    );
    let active = details
        .iter()
        .find(|detail| detail.contains("passive=false"))
        .context("no active instance")?;
    assert!(
        active.contains("secondary_devices=[]"),
        "an active instance without a secondary device: {active}"
    );
    Ok(())
}

#[tokio::test]
async fn custom_backend_serializes_the_gpu_instances_of_a_device() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    // The example backend selects `ExecutionPolicy::DeviceBlocking`, which the
    // core applies to the non-passive GPU instances of one model and device
    // only.
    assert_eq!(
        concurrent_executions(&sut, &config, "device_blocking").await?,
        1
    );
    assert!(
        concurrent_executions(&sut, &config, "cpu_parallel").await? >= 2,
        "two CPU instances keep executing at the same time"
    );

    // The policy the backend asked for is the one Triton holds, read back with
    // `TRITONBACKEND_BackendExecutionPolicy`.
    for (model, policy) in [("mnist", "Blocking"), ("device_blocking", "DeviceBlocking")] {
        let echoed = sut
            .infer_with_parameters(
                model,
                vec![config.mnist_blank_input()],
                &[&config.backend_output_name],
                HashMap::from([("echo_backend_config".to_string(), Parameter::Bool(true))]),
            )
            .await?
            .parameters;
        assert_eq!(
            echoed.get("execution_policy"),
            Some(&string(policy)),
            "{model}"
        );
    }
    Ok(())
}

/// Sends eight overlapping requests to `model` and answers the most execute
/// calls of that model that ran at once.
async fn concurrent_executions(sut: &Sut, config: &Config, model: &str) -> Result<i64> {
    let mut tasks = tokio::task::JoinSet::new();
    for _ in 0..8 {
        let model = model.to_string();
        let input = config.mnist_blank_input();
        let output = config.backend_output_name.clone();
        let url = config.triton_url.clone();
        tasks.spawn(async move {
            let sut = Sut::new(&url).await?;
            sut.infer_with_parameters(
                &model,
                vec![input],
                &[&output],
                HashMap::from([("hold_micros".to_string(), Parameter::Int(50_000))]),
            )
            .await
        });
    }
    while let Some(result) = tasks.join_next().await {
        result??;
    }

    let echoed = sut
        .infer_with_parameters(
            model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("echo_concurrency".to_string(), Parameter::Bool(true))]),
        )
        .await?
        .parameters;
    match echoed.get("max_in_flight") {
        Some(Parameter::Int(max)) => Ok(*max),
        other => anyhow::bail!("no maximum from {model}: {other:?}"),
    }
}

#[tokio::test]
async fn custom_backend_reports_the_host_policy_of_its_instances() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    // `make docker-env-up` starts Triton with `--host-policy=pinned,cpu-cores=0-7`
    // and `--host-policy=cpu,cpu-cores=0-7`.
    for (model, name, settings) in [
        ("host_policy_pinned", "pinned", "cpu-cores=0-7"),
        ("host_policy_unknown", "unset_policy", ""),
        ("mnist_auto", "cpu", "cpu-cores=0-7"),
    ] {
        let echoed = sut
            .infer_with_parameters(
                model,
                vec![config.mnist_blank_input()],
                &[&config.backend_output_name],
                HashMap::from([("echo_host_policy".to_string(), Parameter::Bool(true))]),
            )
            .await?
            .parameters;
        assert_eq!(
            echoed.get("host_policy_name"),
            Some(&string(name)),
            "{model}"
        );
        assert_eq!(
            echoed.get("host_policy_settings"),
            Some(&string(settings)),
            "{model}"
        );
    }
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

async fn model_control(
    sut: &Sut,
    config: &Config,
    parameters: &[(&str, &str)],
) -> Result<std::result::Result<(), Code>> {
    let result = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            parameters
                .iter()
                .map(|(name, value)| (name.to_string(), string(value)))
                .collect(),
        )
        .await;
    match result {
        Ok(_) => Ok(Ok(())),
        Err(error) => match error.downcast_ref::<triton_ng_client::Error>() {
            Some(triton_ng_client::Error::Grpc(status)) => Ok(Err(status.code())),
            _ => anyhow::bail!("not a gRPC error: {error:#}"),
        },
    }
}

#[tokio::test]
async fn custom_backend_loads_a_model_from_files_and_unloads_it() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let model = "mnist_from_files";

    assert_eq!(
        model_control(&sut, &config, &[("load_model", model)]).await?,
        Ok(())
    );
    assert!(sut.model_ready(model, None).await?);
    let result = sut
        .infer(
            model,
            None,
            vec![config.mnist_blank_input()],
            &[&config.output_name],
            None,
        )
        .await?;
    let scores = result
        .outputs
        .iter()
        .find(|o| o.name == config.output_name)
        .context("output tensor not found in response")?;
    assert_eq!(scores.data_f32.len(), 10);

    assert_eq!(
        model_control(&sut, &config, &[("unload_model", model)]).await?,
        Ok(())
    );
    for _ in 0..10_000 {
        if !sut.model_ready(model, None).await? {
            return Ok(());
        }
    }
    anyhow::bail!("model {model} is still ready")
}

#[tokio::test]
async fn custom_backend_cannot_load_a_file_outside_the_model_directory() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let code = model_control(
        &sut,
        &config,
        &[
            ("load_model", "mnist_escaped"),
            ("load_file", "file:../mnist_escaped.onnx"),
        ],
    )
    .await?;
    assert_eq!(code, Err(Code::InvalidArgument));
    Ok(())
}

#[tokio::test]
async fn custom_backend_reports_a_model_missing_from_the_repository() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let code = model_control(
        &sut,
        &config,
        &[("load_model", "no_such_model"), ("load_from", "repository")],
    )
    .await?;
    assert!(code.is_err(), "the model loaded");
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

async fn streamed_steps(
    sut: &Sut,
    config: &Config,
    steps: i64,
) -> Result<(Vec<i32>, Option<triton_ng_client::Error>)> {
    let mut stream = sut
        .infer_stream(
            "mnist_steps",
            vec![config.mnist_blank_images(1)],
            &[&config.backend_output_name],
            HashMap::from([("steps".to_string(), Parameter::Int(steps))]),
        )
        .await?;
    let mut numbers = Vec::new();
    while let Some(response) = stream.next().await {
        match response {
            Ok(response) => {
                let output = infer_result(response)
                    .outputs
                    .into_iter()
                    .find(|o| o.name == config.backend_output_name)
                    .context("step not found in response")?;
                numbers.extend(output.data_i32);
            }
            Err(error) => return Ok((numbers, Some(error))),
        }
    }
    Ok((numbers, None))
}

#[tokio::test]
async fn rescheduled_request_streams_every_step() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let (numbers, error) = streamed_steps(&sut, &config, 1).await?;
    assert_eq!((numbers, error.map(|e| e.to_string())), (vec![1], None));
    let (numbers, error) = streamed_steps(&sut, &config, 2).await?;
    assert_eq!((numbers, error.map(|e| e.to_string())), (vec![1, 2], None));
    Ok(())
}

#[tokio::test]
async fn request_is_rescheduled_only_once() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let (numbers, error) = streamed_steps(&sut, &config, 3).await?;
    assert_eq!(numbers, vec![1, 2]);
    let error = error.context("the third step was rescheduled")?;
    assert!(error.to_string().contains("only once"), "{error}");
    Ok(())
}

#[tokio::test]
async fn request_of_a_model_without_iterative_sequence_is_not_rescheduled() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;

    let error = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("reschedule".to_string(), Parameter::Bool(true))]),
        )
        .await
        .err()
        .context("the request was answered")?;
    assert!(
        format!("{error:#}").contains("iterative_sequence"),
        "{error:#}"
    );
    Ok(())
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
