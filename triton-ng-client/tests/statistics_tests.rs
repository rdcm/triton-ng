#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::support::infer_digit;
use crate::common::sut::Sut;
use anyhow::{Context, Result};

#[tokio::test]
async fn model_statistics_include_loaded_models() -> Result<()> {
    let config = Config::isolated(
        "model_statistics_include_loaded_models",
        "mnist",
        &["mnist", "mnist_onnx"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let stats = sut.model_statistics(None, None).await?;
    let actual = !stats.is_empty();
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn model_statistics_preserve_the_model_version() -> Result<()> {
    let config = Config::isolated(
        "model_statistics_preserve_the_model_version",
        "mnist",
        &["mnist", "mnist_onnx"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let stats = sut.model_statistics(None, None).await?;
    let mnist_stats = stats
        .iter()
        .find(|s| s.name == config.mnist_model)
        .context("mnist_onnx not found in statistics")?;
    let actual = mnist_stats.version.clone();
    let expected = "1".to_string();
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_reports_its_requests_and_batches_to_the_statistics() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_reports_its_requests_and_batches_to_the_statistics",
        "mnist",
        &["mnist", "mnist_onnx"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let (requests_before, batches_before) = backend_statistics(&sut, &config).await?;

    infer_digit(&sut, &config).await?;

    // Sending the response can wake the client before execute returns and
    // reports its batch statistics. Wait for the report, not another inference.
    let (mut requests, mut batches) = backend_statistics(&sut, &config).await?;
    for _ in 0..100 {
        if requests > requests_before && batches > batches_before {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        (requests, batches) = backend_statistics(&sut, &config).await?;
    }
    let expected = (true, true);
    let actual = (requests > requests_before, batches > batches_before);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn inference_count_increases_after_successful_inference() -> Result<()> {
    let config = Config::isolated(
        "inference_count_increases_after_successful_inference",
        "mnist",
        &["mnist", "mnist_onnx"],
    )
    .await?;
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

    let actual = count_after > count_before;
    let expected = true;
    assert_eq!(actual, expected);
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
