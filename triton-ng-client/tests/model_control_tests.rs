#[path = "common/mod.rs"]
mod common;

use std::collections::HashMap;

use crate::common::sut::Sut;
use anyhow::{Context, Result};
use tonic::Code;

#[tokio::test]
async fn custom_backend_loads_a_model_from_files() -> Result<()> {
    let sut = Sut::new("fixture_model_control", &[]).await?;
    let model = sut.model("mnist_from_files");
    let loaded = sut.model_control(&[("load_model", model)]).await?;

    let expected = (Ok(()), true);
    let actual = (loaded, sut.repository_model_ready(model).await?);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn a_model_loaded_from_files_returns_class_scores() -> Result<()> {
    let sut = Sut::new("fixture_model_control", &[]).await?;
    let model = sut.model("mnist_from_files");
    let loaded = sut.model_control(&[("load_model", model)]).await?;
    loaded.map_err(|code| anyhow::anyhow!("load failed: {code:?}"))?;
    let result = sut
        .infer_model_with_parameters(model, HashMap::new())
        .await?;
    let output = result
        .outputs
        .iter()
        .find(|o| o.name == "Plus214_Output_0")
        .context("scores not found")?;
    let expected = 10;
    let actual = output.data_f32.len();
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_unloads_a_model_loaded_from_files() -> Result<()> {
    let sut = Sut::new("fixture_model_control", &[]).await?;
    let model = sut.model("mnist_from_files");
    let loaded = sut.model_control(&[("load_model", model)]).await?;
    loaded.map_err(|code| anyhow::anyhow!("load failed: {code:?}"))?;
    let unloaded = sut.model_control(&[("unload_model", model)]).await?;
    let mut ready = true;
    for _ in 0..100 {
        ready = sut.repository_model_ready(model).await?;
        if !ready {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let expected = (Ok(()), false);
    let actual = (unloaded, ready);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_cannot_load_a_file_outside_the_model_directory() -> Result<()> {
    let sut = Sut::new("fixture_model_control", &[]).await?;

    let code = sut
        .model_control(&[
            ("load_model", sut.model("mnist_escaped")),
            ("load_file", "file:../mnist_escaped.onnx"),
        ])
        .await?;
    let actual = code;
    let expected = Err(Code::InvalidArgument);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_reports_a_model_missing_from_the_repository() -> Result<()> {
    let sut = Sut::new("fixture_model_control", &[]).await?;

    let code = sut
        .model_control(&[
            ("load_model", sut.model("no_such_model")),
            ("load_from", "repository"),
        ])
        .await?;
    let actual = code.is_err();
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}
