#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::support::string;
use crate::common::sut::Sut;
use anyhow::Result;
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn custom_backend_reads_the_metadata_of_a_request() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_reads_the_metadata_of_a_request",
        "fixture_metadata",
        &["fixture_metadata"],
    )
    .await?;
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

    let expected = (
        Some(&string("metadata-1")),
        Some(&string("42")),
        Some(&Parameter::Bool(true)),
        Some(&Parameter::Bool(false)),
        Some(&Parameter::Int(5_000_000)),
    );
    let actual = (
        echoed.get("request_id"),
        echoed.get("correlation_id"),
        echoed.get("sequence_start"),
        echoed.get("sequence_end"),
        echoed.get("timeout_us"),
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn request_metadata_preserves_a_string_correlation_id() -> Result<()> {
    let config = Config::isolated(
        "request_metadata_preserves_a_string_correlation_id",
        "fixture_metadata",
        &["fixture_metadata"],
    )
    .await?;
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
    let expected = (Some(&string("call-7")), Some(&Parameter::Bool(false)), None);
    let actual = (
        echoed.get("correlation_id"),
        echoed.get("sequence_start"),
        echoed.get("timeout_us"),
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn request_metadata_is_empty_when_not_provided() -> Result<()> {
    let config = Config::isolated(
        "request_metadata_is_empty_when_not_provided",
        "fixture_metadata",
        &["fixture_metadata"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let bare = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("echo_metadata".to_string(), Parameter::Bool(true))]),
        )
        .await?
        .parameters;
    let expected = (None, None);
    let actual = (bare.get("request_id"), bare.get("correlation_id"));
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_passes_metadata_and_parameters_to_a_bls_request() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_passes_metadata_and_parameters_to_a_bls_request",
        "fixture_metadata",
        &["fixture_metadata"],
    )
    .await?;
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

    let expected = (
        Some(&string("metadata-3/bls")),
        Some(&string("42")),
        Some(&Parameter::Bool(false)),
        Some(&Parameter::Bool(true)),
        Some(&Parameter::Int(6_000_000)),
        Some(&string("by the backend")),
    );
    let actual = (
        echoed.get("request_id"),
        echoed.get("correlation_id"),
        echoed.get("sequence_start"),
        echoed.get("sequence_end"),
        echoed.get("timeout_us"),
        echoed.get("forwarded"),
    );
    assert_eq!(actual, expected);
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
    if result.id != id {
        return Err(anyhow::anyhow!("response id is {}", result.id));
    }
    Ok(result.parameters)
}
