use super::support::{infer_backend, response_parameter};
use crate::common::config::Config;
use crate::common::sut::Sut;
use anyhow::{Context, Result};
use triton_ng_client::Parameter;

async fn kept_inputs(sut: &Sut, config: &Config) -> Result<i64> {
    let result = infer_backend(sut, config, &[]).await?;
    match response_parameter(&result, "kept_inputs")? {
        Parameter::Int(kept) => Ok(kept),
        other => return Err(anyhow::anyhow!("kept_inputs is {other:?}")),
    }
}

pub(crate) async fn wait_for_kept_inputs(sut: &Sut, config: &Config) -> Result<()> {
    let mut kept = 0;
    for _ in 0..10_000 {
        kept = kept_inputs(sut, config).await?;
        if kept == 1 {
            return Ok(());
        }
    }
    return Err(anyhow::anyhow!(
        "the backend kept {kept} inputs of its requests"
    ));
}

pub(crate) async fn infer_digit_keeping_shared_input_with_statistics(
    config: Config,
) -> Result<i32> {
    let sut = Sut::new(&config.triton_url).await?;
    let result = infer_backend(
        &sut,
        &config,
        &[
            ("shared_input", Parameter::Bool(true)),
            ("enable_statistics", Parameter::Bool(true)),
        ],
    )
    .await?;
    let digit = result
        .outputs
        .into_iter()
        .find(|o| o.name == config.backend_output_name)
        .context("digit not found in response")?
        .data_i32;
    Ok(digit[0])
}
