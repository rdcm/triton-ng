#[path = "common/mod.rs"]
mod common;

use crate::common::sut::Sut;
use anyhow::{Context, Result};

#[tokio::test]
async fn mnist_onnx_model_is_ready() -> Result<()> {
    let sut = Sut::new("mnist_onnx", &[]).await?;
    let ready = sut.model_ready().await?;
    let actual = ready;
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn model_metadata_identifies_its_name_version_and_platform() -> Result<()> {
    let sut = Sut::new("mnist_onnx", &[]).await?;
    let info = sut.model_metadata().await?;

    let expected = (
        sut.model("mnist_onnx").to_string(),
        true,
        "onnxruntime_onnx".to_string(),
    );
    let actual = (
        info.name,
        info.versions.contains(&"1".to_string()),
        info.platform,
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn model_metadata_describes_its_fp32_input() -> Result<()> {
    let sut = Sut::new("mnist_onnx", &[]).await?;
    let info = sut.model_metadata().await?;

    let input = info
        .inputs
        .iter()
        .find(|t| t.name == "Input3")
        .context("input tensor not found")?;
    let expected = ("FP32".to_string(), vec![1, 1, 28, 28]);
    let actual = (input.datatype.clone(), input.shape.clone());
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn model_metadata_describes_its_fp32_output() -> Result<()> {
    let sut = Sut::new("mnist_onnx", &[]).await?;
    let info = sut.model_metadata().await?;

    let output = info
        .outputs
        .iter()
        .find(|t| t.name == "Plus214_Output_0")
        .context("output tensor not found")?;
    let expected = ("FP32".to_string(), vec![1, 10]);
    let actual = (output.datatype.clone(), output.shape.clone());
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn mnist_onnx_model_config_identifies_onnxruntime_platform() -> Result<()> {
    let sut = Sut::new("mnist_onnx", &[]).await?;
    let cfg = sut.model_config().await?;
    let cfg = cfg.context("model config was None")?;
    let expected = (
        sut.model("mnist_onnx").to_string(),
        "onnxruntime_onnx".to_string(),
    );
    let actual = (cfg.name, cfg.platform);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn repository_index_lists_all_available_models() -> Result<()> {
    let sut = Sut::new("mnist", &["mnist_onnx"]).await?;
    let models = sut.repository_index(false).await?;
    let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
    let expected = (true, true);
    let actual = (
        names.contains(&sut.model("mnist_onnx")),
        names.contains(&sut.model("mnist")),
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn ready_repository_index_contains_only_ready_models() -> Result<()> {
    let sut = Sut::new("mnist_onnx", &[]).await?;
    let models = sut.repository_index(true).await?;
    let states: Vec<_> = models.iter().map(|model| model.state.as_str()).collect();
    let expected = (false, vec!["READY"; states.len()]);
    let actual = (models.is_empty(), states.clone());
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn model_becomes_ready_after_unload_and_reload() -> Result<()> {
    let sut = Sut::new("mnist_onnx", &[]).await?;
    sut.repository_model_unload().await?;
    sut.repository_model_load().await?;
    let ready = sut.model_ready().await?;
    let actual = ready;
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}
