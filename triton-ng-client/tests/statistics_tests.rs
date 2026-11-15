#[path = "common/mod.rs"]
mod common;

use crate::common::sut::Sut;
use anyhow::{Context, Result};

#[tokio::test]
async fn model_statistics_include_loaded_models() -> Result<()> {
    let sut = Sut::new("mnist_onnx", &[]).await?;
    let stats = sut.repository_statistics().await?;
    let actual = !stats.is_empty();
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn model_statistics_preserve_the_model_version() -> Result<()> {
    let sut = Sut::new("mnist_onnx", &[]).await?;
    let stats = sut.model_statistics().await?;
    let mnist_stats = stats
        .iter()
        .find(|s| s.name == sut.model("mnist_onnx"))
        .context("mnist_onnx not found in statistics")?;
    let actual = mnist_stats.version.clone();
    let expected = "1".to_string();
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_reports_its_requests_and_batches_to_the_statistics() -> Result<()> {
    let sut = Sut::new("mnist", &[]).await?;
    let (requests_before, batches_before) = sut.backend_statistics().await?;

    sut.infer_digit().await?;

    // Sending the response can wake the client before execute returns and
    // reports its batch statistics. Wait for the report, not another inference.
    let (mut requests, mut batches) = sut.backend_statistics().await?;
    for _ in 0..100 {
        if requests > requests_before && batches > batches_before {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        (requests, batches) = sut.backend_statistics().await?;
    }
    let expected = (true, true);
    let actual = (requests > requests_before, batches > batches_before);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn inference_count_increases_after_successful_inference() -> Result<()> {
    let sut = Sut::new("mnist_onnx", &[]).await?;
    let model = sut.model("mnist_onnx").to_string();

    let before = sut.model_statistics().await?;
    let count_before = before
        .iter()
        .find(|s| s.name == model)
        .map(|s| s.inference_count)
        .unwrap_or(0);

    sut.infer_async(&[]).await?;

    let after = sut.model_statistics().await?;
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
