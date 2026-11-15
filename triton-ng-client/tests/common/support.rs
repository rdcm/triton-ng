use crate::common;
use crate::common::config::Config;
use crate::common::sut::Sut;
use anyhow::{Context, Result};
use triton_ng_client::Parameter;

pub(crate) async fn infer_digit(sut: &Sut, config: &Config) -> Result<i32> {
    let digit = sut
        .infer(
            &config.backend_model,
            None,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            None,
        )
        .await?
        .outputs
        .into_iter()
        .find(|o| o.name == config.backend_output_name)
        .context("digit not found in response")?
        .data_i32;
    Ok(digit[0])
}

pub(crate) fn string(value: &str) -> Parameter {
    Parameter::String(value.to_string())
}

pub(crate) async fn infer_backend(
    sut: &Sut,
    config: &Config,
    parameters: &[(&str, Parameter)],
) -> Result<common::dto::InferResult> {
    sut.infer_with_parameters(
        &config.backend_model,
        vec![config.mnist_blank_input()],
        &[&config.backend_output_name],
        parameters
            .iter()
            .map(|(name, value)| (name.to_string(), value.clone()))
            .collect(),
    )
    .await
}

pub(crate) fn response_parameter(
    result: &common::dto::InferResult,
    name: &str,
) -> Result<Parameter> {
    result
        .parameters
        .get(name)
        .cloned()
        .with_context(|| format!("response parameter {name} is missing"))
}
