mod common;

use std::collections::HashMap;

use anyhow::{Context, Result};
use common::sut::{Sut, create_posix_shm, remove_posix_shm};
use triton_ng_client::Parameter;

#[tokio::test]
async fn server_is_live() -> Result<()> {
    let sut = Sut::new().await?;
    let live = sut.server_live().await?;
    assert!(live);
    Ok(())
}

#[tokio::test]
async fn server_is_ready() -> Result<()> {
    let sut = Sut::new().await?;
    let ready = sut.server_ready().await?;
    assert!(ready);
    Ok(())
}

#[tokio::test]
async fn server_metadata_contains_triton_name() -> Result<()> {
    let sut = Sut::new().await?;
    let info = sut.server_metadata().await?;
    assert_eq!(info.name, "triton");
    assert!(!info.version.is_empty());
    assert!(!info.extensions.is_empty());
    Ok(())
}

#[tokio::test]
async fn mnist_onnx_model_is_ready() -> Result<()> {
    let sut = Sut::new().await?;
    let ready = sut.model_ready(&sut.config.mnist_model, None).await?;
    assert!(ready);
    Ok(())
}

#[tokio::test]
async fn mnist_onnx_model_metadata_describes_fp32_input_and_output() -> Result<()> {
    let sut = Sut::new().await?;
    let info = sut.model_metadata(&sut.config.mnist_model, None).await?;

    assert_eq!(info.name, sut.config.mnist_model);
    assert!(info.versions.contains(&"1".to_string()));
    assert_eq!(info.platform, "onnxruntime_onnx");

    let input = info
        .inputs
        .iter()
        .find(|t| t.name == sut.config.input_name)
        .context("input tensor not found")?;
    assert_eq!(input.datatype, "FP32");
    assert_eq!(input.shape, vec![1, 1, 28, 28]);

    let output = info
        .outputs
        .iter()
        .find(|t| t.name == sut.config.output_name)
        .context("output tensor not found")?;
    assert_eq!(output.datatype, "FP32");
    assert_eq!(output.shape, vec![1, 10]);

    Ok(())
}

#[tokio::test]
async fn mnist_onnx_model_config_identifies_onnxruntime_platform() -> Result<()> {
    let sut = Sut::new().await?;
    let cfg = sut.model_config(&sut.config.mnist_model, None).await?;
    let cfg = cfg.context("model config was None")?;
    assert_eq!(cfg.name, sut.config.mnist_model);
    assert_eq!(cfg.platform, "onnxruntime_onnx");
    Ok(())
}

#[tokio::test]
async fn model_statistics_are_returned_for_all_models() -> Result<()> {
    let sut = Sut::new().await?;
    let stats = sut.model_statistics(None, None).await?;
    assert!(!stats.is_empty());
    let mnist_stats = stats
        .iter()
        .find(|s| s.name == sut.config.mnist_model)
        .context("mnist_onnx not found in statistics")?;
    assert_eq!(mnist_stats.version, "1");
    Ok(())
}

#[tokio::test]
async fn repository_index_lists_all_available_models() -> Result<()> {
    let sut = Sut::new().await?;
    let models = sut.repository_index(false).await?;
    let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
    assert!(
        names.contains(&sut.config.mnist_model.as_str()),
        "{} missing from index",
        sut.config.mnist_model
    );
    assert!(names.contains(&"mnist"), "mnist missing from index");
    Ok(())
}

#[tokio::test]
async fn ready_repository_index_contains_only_ready_models() -> Result<()> {
    let sut = Sut::new().await?;
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
    let sut = Sut::new().await?;
    sut.repository_model_unload(&sut.config.mnist_model).await?;
    sut.repository_model_load(&sut.config.mnist_model).await?;
    let ready = sut.model_ready(&sut.config.mnist_model, None).await?;
    assert!(ready);
    Ok(())
}

#[tokio::test]
async fn mnist_onnx_inference_returns_ten_class_scores() -> Result<()> {
    let sut = Sut::new().await?;
    let output_name = sut.config.output_name.clone();
    let result = sut
        .infer(
            &sut.config.mnist_model,
            None,
            vec![sut.mnist_blank_input()],
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
async fn custom_backend_returns_target_model_result() -> Result<()> {
    let sut = Sut::new().await?;
    let output_name = sut.config.output_name.clone();
    let mut scores = Vec::new();
    for model in [&sut.config.backend_model, &sut.config.mnist_model] {
        let result = sut
            .infer(
                model,
                None,
                vec![sut.mnist_blank_input()],
                &[&output_name],
                None,
            )
            .await?;
        let output = result
            .outputs
            .into_iter()
            .find(|o| o.name == output_name)
            .context("output tensor not found in response")?;
        assert_eq!(output.shape, vec![1, 10]);
        scores.push(output.data_f32);
    }
    assert_eq!(scores[0], scores[1]);
    Ok(())
}

#[tokio::test]
async fn custom_backend_answers_concurrent_requests() -> Result<()> {
    let mut tasks = tokio::task::JoinSet::new();
    for _ in 0..200 {
        tasks.spawn(async {
            let sut = Sut::new().await?;
            let output_name = sut.config.output_name.clone();
            let result = sut
                .infer(
                    &sut.config.backend_model,
                    None,
                    vec![sut.mnist_blank_input()],
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
async fn backend_counters(sut: &Sut) -> Result<(i64, i64)> {
    let output_name = sut.config.output_name.clone();
    let result = sut
        .infer_with_parameters(
            &sut.config.backend_model,
            vec![sut.mnist_blank_input()],
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

async fn wait_for_backend_counters(sut: &Sut, expected: impl Fn(i64, i64) -> bool) -> Result<()> {
    for _ in 0..10_000 {
        let (held, cancelled) = backend_counters(sut).await?;
        if expected(held, cancelled) {
            return Ok(());
        }
    }
    anyhow::bail!("backend counters did not reach the expected values")
}

#[tokio::test]
async fn custom_backend_answers_cancelled_request_as_cancelled() -> Result<()> {
    let sut = Sut::new().await?;
    let (_, cancelled_before) = backend_counters(&sut).await?;

    let held = tokio::spawn(async {
        let sut = Sut::new().await?;
        let output_name = sut.config.output_name.clone();
        sut.infer_with_parameters(
            &sut.config.backend_model,
            vec![sut.mnist_blank_input()],
            &[&output_name],
            HashMap::from([("hold_until_cancelled".to_string(), Parameter::Bool(true))]),
        )
        .await
    });

    wait_for_backend_counters(&sut, |held, _| held == 1).await?;
    held.abort();
    wait_for_backend_counters(&sut, |held, cancelled| {
        held == 0 && cancelled == cancelled_before + 1
    })
    .await
}

#[tokio::test]
async fn inference_request_id_is_echoed_in_response() -> Result<()> {
    let sut = Sut::new().await?;
    let output_name = sut.config.output_name.clone();
    let request_id = "test-echo-id-42".to_string();
    let result = sut
        .infer(
            &sut.config.mnist_model,
            None,
            vec![sut.mnist_blank_input()],
            &[&output_name],
            Some(request_id.clone()),
        )
        .await?;
    assert_eq!(result.id, request_id);
    Ok(())
}

#[tokio::test]
async fn inference_count_increases_after_successful_inference() -> Result<()> {
    let sut = Sut::new().await?;
    let output_name = sut.config.output_name.clone();
    let model = sut.config.mnist_model.clone();

    let before = sut.model_statistics(Some(&model), None).await?;
    let count_before = before
        .iter()
        .find(|s| s.name == model)
        .map(|s| s.inference_count)
        .unwrap_or(0);

    sut.infer(
        &model,
        None,
        vec![sut.mnist_blank_input()],
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
    let sut = Sut::new().await?;
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

    let sut = Sut::new().await?;
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
    let sut = Sut::new().await?;
    sut.cuda_shared_memory_unregister(None).await?;
    let regions = sut.cuda_shared_memory_status(None).await?;
    assert!(regions.is_empty());
    Ok(())
}

#[tokio::test]
async fn cuda_shared_memory_register_returns_error_for_invalid_handle() -> Result<()> {
    let sut = Sut::new().await?;
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
    let sut = Sut::new().await?;
    let settings = sut.trace_settings(HashMap::new(), None).await?;
    assert!(!settings.is_empty());
    Ok(())
}

#[tokio::test]
async fn log_settings_are_readable() -> Result<()> {
    let sut = Sut::new().await?;
    let settings = sut.log_settings(HashMap::new()).await?;
    assert!(!settings.is_empty());
    Ok(())
}
