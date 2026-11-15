#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::support::string;
use crate::common::sut::Sut;
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn backend_state_preserves_its_configuration() -> Result<()> {
    let config = Config::isolated(
        "backend_state_preserves_its_configuration",
        "fixture_backend_config",
        &[
            "fixture_backend_config",
            "fixture_backend_state",
            "fixture_backend_state_other",
        ],
    )
    .await?;
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
    let actual = echoed.get("greeting");
    let expected = Some(&string("hello"));
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn backend_artifacts_describe_the_backend_directory() -> Result<()> {
    let config = Config::isolated(
        "backend_artifacts_describe_the_backend_directory",
        "fixture_backend_config",
        &[
            "fixture_backend_config",
            "fixture_backend_state",
            "fixture_backend_state_other",
        ],
    )
    .await?;
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
    let backend_directory = echoed
        .get("backend-directory")
        .and_then(|value| match value {
            Parameter::String(directory) => Some(directory),
            _ => None,
        })
        .context("backend-directory is missing or not a string")?;
    let expected = (
        Some(&string("FileSystem")),
        Some(&string(&format!("{backend_directory}/custom_backend"))),
    );
    let actual = (
        echoed.get("artifact_type"),
        echoed.get("artifact_directory"),
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_shares_its_state_between_models() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_shares_its_state_between_models",
        "mnist",
        &[
            "fixture_backend_state",
            "fixture_backend_state_other",
            "mnist",
        ],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let mut counted = Vec::new();
    for model in [
        config.model("fixture_backend_state"),
        config.model("fixture_backend_state_other"),
    ] {
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
            other => return Err(anyhow::anyhow!("no count from {model}: {other:?}")),
        });
    }

    let expected = true;
    let actual = counted[1] > counted[0];
    assert_eq!(actual, expected);
    Ok(())
}
