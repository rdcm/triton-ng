#[path = "common/mod.rs"]
mod common;

use crate::common::support::string;
use crate::common::sut::Sut;
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn backend_state_preserves_its_configuration() -> Result<()> {
    let sut = Sut::new("fixture_backend_config", &[]).await?;

    let echoed = sut
        .infer_with_parameters(HashMap::from([(
            "echo_backend_config".to_string(),
            Parameter::Bool(true),
        )]))
        .await?
        .parameters;
    let actual = echoed.get("greeting");
    let expected = Some(&string("hello"));
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn backend_artifacts_describe_the_backend_directory() -> Result<()> {
    let sut = Sut::new("fixture_backend_config", &[]).await?;

    let echoed = sut
        .infer_with_parameters(HashMap::from([(
            "echo_backend_config".to_string(),
            Parameter::Bool(true),
        )]))
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
    let sut = Sut::new("fixture_backend_state", &["fixture_backend_state_other"]).await?;

    let mut counted = Vec::new();
    for model in [
        sut.model("fixture_backend_state"),
        sut.model("fixture_backend_state_other"),
    ] {
        let echoed = sut
            .infer_model_with_parameters(
                model,
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
