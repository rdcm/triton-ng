#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::support::string;
use crate::common::sut::Sut;
use anyhow::{Context, Result};
use tonic::Code;

#[tokio::test]
async fn custom_backend_loads_a_model_from_files() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_loads_a_model_from_files",
        "fixture_model_control",
        &["fixture_model_control"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let model = config.model("mnist_from_files");
    let loaded = model_control(&sut, &config, &[("load_model", model)]).await?;

    let expected = (Ok(()), true);
    let actual = (loaded, sut.model_ready(model, None).await?);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn a_model_loaded_from_files_returns_class_scores() -> Result<()> {
    let config = Config::isolated(
        "a_model_loaded_from_files_returns_class_scores",
        "fixture_model_control",
        &["fixture_model_control"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let model = config.model("mnist_from_files");
    let loaded = model_control(&sut, &config, &[("load_model", model)]).await?;
    loaded.map_err(|code| anyhow::anyhow!("load failed: {code:?}"))?;
    let result = sut
        .infer(
            model,
            None,
            vec![config.mnist_blank_input()],
            &[&config.output_name],
            None,
        )
        .await?;
    let output = result
        .outputs
        .iter()
        .find(|o| o.name == config.output_name)
        .context("scores not found")?;
    let expected = 10;
    let actual = output.data_f32.len();
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_unloads_a_model_loaded_from_files() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_unloads_a_model_loaded_from_files",
        "fixture_model_control",
        &["fixture_model_control"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let model = config.model("mnist_from_files");
    let loaded = model_control(&sut, &config, &[("load_model", model)]).await?;
    loaded.map_err(|code| anyhow::anyhow!("load failed: {code:?}"))?;
    let unloaded = model_control(&sut, &config, &[("unload_model", model)]).await?;
    let mut ready = true;
    for _ in 0..100 {
        ready = sut.model_ready(model, None).await?;
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
    let config = Config::isolated(
        "custom_backend_cannot_load_a_file_outside_the_model_directory",
        "fixture_model_control",
        &["fixture_model_control"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let code = model_control(
        &sut,
        &config,
        &[
            ("load_model", config.model("mnist_escaped")),
            ("load_file", "file:../mnist_escaped.onnx"),
        ],
    )
    .await?;
    let actual = code;
    let expected = Err(Code::InvalidArgument);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_reports_a_model_missing_from_the_repository() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_reports_a_model_missing_from_the_repository",
        "fixture_model_control",
        &["fixture_model_control"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let code = model_control(
        &sut,
        &config,
        &[
            ("load_model", config.model("no_such_model")),
            ("load_from", "repository"),
        ],
    )
    .await?;
    let actual = code.is_err();
    let expected = true;
    assert_eq!(actual, expected);
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
            _ => return Err(anyhow::anyhow!("not a gRPC error: {error:#}")),
        },
    }
}
