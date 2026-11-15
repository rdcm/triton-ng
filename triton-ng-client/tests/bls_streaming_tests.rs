#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::streaming::stream_counters;
use crate::common::support::{infer_backend, infer_digit, response_parameter, string};
use crate::common::sut::Sut;
use anyhow::{Context, Result};
use triton_ng_client::Parameter;

#[tokio::test]
async fn a_bls_stream_reports_its_output_shape() -> Result<()> {
    let config = Config::isolated(
        "a_bls_stream_reports_its_output_shape",
        "fixture_bls_streaming",
        &["fixture_bls_streaming", "mnist_stream"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let result = infer_backend(&sut, &config, &[("stream_images", Parameter::Int(3))]).await?;

    let actual = response_parameter(&result, "streamed_digits")?;
    let expected = Parameter::Int(3);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn a_bls_stream_returns_every_decoupled_response() -> Result<()> {
    let config = Config::isolated(
        "a_bls_stream_returns_every_decoupled_response",
        "fixture_bls_streaming",
        &["fixture_bls_streaming", "mnist_stream"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let expected = infer_digit(&sut, &config).await?;

    let result = infer_backend(&sut, &config, &[("stream_images", Parameter::Int(3))]).await?;

    let digit = result
        .outputs
        .into_iter()
        .find(|o| o.name == config.backend_output_name)
        .context("digit not found in response")?;
    let actual = digit.data_i32;
    let expected = vec![expected];
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn bls_stream_of_a_decoupled_model_without_responses_is_empty() -> Result<()> {
    let config = Config::isolated(
        "bls_stream_of_a_decoupled_model_without_responses_is_empty",
        "fixture_bls_streaming",
        &["fixture_bls_streaming", "mnist_stream"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let result = infer_backend(&sut, &config, &[("stream_images", Parameter::Int(0))]).await?;

    let actual = response_parameter(&result, "streamed_digits")?;
    let expected = Parameter::Int(0);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn infer_async_fails_when_a_decoupled_model_sends_several_responses() -> Result<()> {
    let config = Config::isolated(
        "infer_async_fails_when_a_decoupled_model_sends_several_responses",
        "fixture_bls_streaming",
        &["fixture_bls_streaming", "mnist_stream"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let error = infer_async_error(&sut, &config, 2).await?;

    let actual = error.contains("more than one response");
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn infer_async_fails_when_a_decoupled_model_sends_no_response() -> Result<()> {
    let config = Config::isolated(
        "infer_async_fails_when_a_decoupled_model_sends_no_response",
        "fixture_bls_streaming",
        &["fixture_bls_streaming", "mnist_stream"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let error = infer_async_error(&sut, &config, 0).await?;

    let actual = error.contains("sent no response");
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn cancelled_bls_stream_ends_with_the_cancellation_of_the_model() -> Result<()> {
    let config = Config::isolated(
        "cancelled_bls_stream_ends_with_the_cancellation_of_the_model",
        "fixture_bls_streaming",
        &["fixture_bls_streaming", "mnist_stream"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let (_, before) = stream_counters(&sut, &config).await?;

    let result = infer_backend(&sut, &config, &[("cancel_stream", string("cancel"))]).await?;

    let actual = response_parameter(&result, "stream_ended_with")?;
    let expected = string("CANCELLED");
    wait_for_stream_cancellation(&sut, &config, before).await?;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn dropped_bls_stream_cancels_its_request() -> Result<()> {
    let config = Config::isolated(
        "dropped_bls_stream_cancels_its_request",
        "fixture_bls_streaming",
        &["fixture_bls_streaming", "mnist_stream"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let (_, before) = stream_counters(&sut, &config).await?;

    let result = infer_backend(&sut, &config, &[("cancel_stream", string("drop"))]).await?;

    let actual = response_parameter(&result, "stream_ended_with")?;
    let expected = string("dropped");
    wait_for_stream_cancellation(&sut, &config, before).await?;
    assert_eq!(actual, expected);
    Ok(())
}

async fn infer_async_error(sut: &Sut, config: &Config, images: i64) -> Result<String> {
    let error = infer_backend(
        sut,
        config,
        &[
            ("stream_images", Parameter::Int(images)),
            ("via_infer_async", Parameter::Bool(true)),
        ],
    )
    .await
    .err()
    .context("the inference succeeded")?;
    Ok(format!("{error:#}"))
}

async fn wait_for_stream_cancellation(sut: &Sut, config: &Config, before: i64) -> Result<()> {
    for _ in 0..10_000 {
        let (held, cancelled) = stream_counters(sut, config).await?;
        if held == 0 && cancelled == before + 1 {
            return Ok(());
        }
    }
    return Err(anyhow::anyhow!(
        "the decoupled model did not see the cancellation"
    ));
}
