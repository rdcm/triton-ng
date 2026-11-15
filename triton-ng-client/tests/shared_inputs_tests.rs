#[path = "common/mod.rs"]
mod common;

use crate::common::shared_inputs::wait_for_kept_inputs;

use crate::common::config::Config;
use crate::common::support::{infer_backend, infer_digit};
use crate::common::sut::Sut;
use anyhow::{Context, Result};
use triton_ng_client::Parameter;

#[tokio::test]
async fn custom_backend_passes_an_input_of_its_request_on_without_a_copy() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_passes_an_input_of_its_request_on_without_a_copy",
        "fixture_shared_inputs",
        &["fixture_shared_inputs"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let expected = infer_digit(&sut, &config).await?;

    let digit = infer_digit_with_shared_input(&sut, &config, false).await?;

    let actual = digit;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn a_shared_input_holds_back_its_request_response() -> Result<()> {
    let config = Config::isolated(
        "a_shared_input_holds_back_its_request_response",
        "fixture_shared_inputs",
        &["fixture_shared_inputs"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let held = tokio::spawn(infer_digit_keeping_shared_input(config.clone()));
    wait_for_kept_inputs(&sut, &config).await?;
    let actual = !held.is_finished();
    infer_backend(
        &sut,
        &config,
        &[("release_shared_inputs", Parameter::Bool(true))],
    )
    .await?;
    held.await??;
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn dropping_a_shared_input_allows_its_request_response() -> Result<()> {
    let config = Config::isolated(
        "dropping_a_shared_input_allows_its_request_response",
        "fixture_shared_inputs",
        &["fixture_shared_inputs"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let expected = infer_digit(&sut, &config).await?;

    let held = tokio::spawn(infer_digit_keeping_shared_input(config.clone()));
    wait_for_kept_inputs(&sut, &config).await?;
    infer_backend(
        &sut,
        &config,
        &[("release_shared_inputs", Parameter::Bool(true))],
    )
    .await?;

    let actual = held.await??;
    assert_eq!(actual, expected);
    Ok(())
}

async fn infer_digit_with_shared_input(sut: &Sut, config: &Config, keep: bool) -> Result<i32> {
    let result = infer_backend(sut, config, &[("shared_input", Parameter::Bool(keep))]).await?;
    let digit = result
        .outputs
        .into_iter()
        .find(|o| o.name == config.backend_output_name)
        .context("digit not found in response")?
        .data_i32;
    Ok(digit[0])
}

async fn infer_digit_keeping_shared_input(config: Config) -> Result<i32> {
    let sut = Sut::new(&config.triton_url).await?;
    infer_digit_with_shared_input(&sut, &config, true).await
}
