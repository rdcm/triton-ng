#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::sut::Sut;
use anyhow::{Context, Result};
use std::collections::HashMap;
use tonic::Code;
use triton_ng_client::Parameter;

#[tokio::test]
async fn custom_backend_answers_with_the_requested_error_code() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_answers_with_the_requested_error_code",
        "fixture_errors",
        &["fixture_errors"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let mut actual = Vec::new();
    let mut expected = Vec::new();
    for (code, expected_code) in [
        ("INTERNAL", Code::Internal),
        ("NOT_FOUND", Code::NotFound),
        ("INVALID_ARG", Code::InvalidArgument),
        ("UNAVAILABLE", Code::Unavailable),
        ("UNSUPPORTED", Code::Unimplemented),
        ("ALREADY_EXISTS", Code::AlreadyExists),
        ("CANCELLED", Code::Cancelled),
    ] {
        let actual_code = grpc_code_of_failed_inference(&sut, &config, "error_code", code).await?;
        actual.push(actual_code);
        expected.push(expected_code);
    }
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_passes_on_the_code_of_a_failed_bls_call() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_passes_on_the_code_of_a_failed_bls_call",
        "fixture_errors",
        &["fixture_errors"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let code = grpc_code_of_failed_inference(&sut, &config, "infer_model", "missing_model").await?;

    let actual = code;
    let expected = Code::NotFound;
    assert_eq!(actual, expected);
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
        _ => return Err(anyhow::anyhow!("not a gRPC error: {error:#}")),
    }
}
