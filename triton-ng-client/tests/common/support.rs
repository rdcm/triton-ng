use crate::common;
use anyhow::{Context, Result};
use triton_ng_client::Parameter;

pub(crate) fn string(value: &str) -> Parameter {
    Parameter::String(value.to_string())
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
