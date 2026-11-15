#[path = "common/mod.rs"]
mod common;

use crate::common::support::{response_parameter, string};
use crate::common::sut::Sut;
use anyhow::{Context, Result};
use triton_ng_client::Parameter;

#[tokio::test]
async fn a_bls_stream_reports_its_response_count() -> Result<()> {
    let sut = Sut::new("fixture_bls_streaming", &["mnist_stream"]).await?;

    let result = sut
        .infer_async(&[("stream_images", Parameter::Int(3))])
        .await?;

    let actual = response_parameter(&result, "streamed_digits")?;
    let expected = Parameter::Int(3);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn a_bls_stream_returns_every_decoupled_response() -> Result<()> {
    let sut = Sut::new("fixture_bls_streaming", &["mnist_stream"]).await?;
    let expected = sut.infer_digit().await?;

    let result = sut
        .infer_async(&[("stream_images", Parameter::Int(3))])
        .await?;

    let digit = result
        .outputs
        .into_iter()
        .find(|o| o.name == "DIGIT")
        .context("digit not found in response")?;
    let actual = digit.data_i32;
    let expected = vec![expected];
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn bls_stream_of_a_decoupled_model_without_responses_is_empty() -> Result<()> {
    let sut = Sut::new("fixture_bls_streaming", &["mnist_stream"]).await?;

    let result = sut
        .infer_async(&[("stream_images", Parameter::Int(0))])
        .await?;

    let actual = response_parameter(&result, "streamed_digits")?;
    let expected = Parameter::Int(0);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn infer_async_fails_when_a_decoupled_model_sends_several_responses() -> Result<()> {
    let sut = Sut::new("fixture_bls_streaming", &["mnist_stream"]).await?;

    let error = sut.infer_async_error(2).await?;

    let actual = error.contains("more than one response");
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn infer_async_fails_when_a_decoupled_model_sends_no_response() -> Result<()> {
    let sut = Sut::new("fixture_bls_streaming", &["mnist_stream"]).await?;

    let error = sut.infer_async_error(0).await?;

    let actual = error.contains("sent no response");
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn cancelled_bls_stream_ends_with_the_cancellation_of_the_model() -> Result<()> {
    let sut = Sut::new("fixture_bls_streaming", &["mnist_stream"]).await?;
    let (_, before) = sut.stream_counters().await?;

    let result = sut
        .infer_async(&[("cancel_stream", string("cancel"))])
        .await?;

    let actual = response_parameter(&result, "stream_ended_with")?;
    let expected = string("CANCELLED");
    sut.wait_for_stream_cancellation(before).await?;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn dropped_bls_stream_cancels_its_request() -> Result<()> {
    let sut = Sut::new("fixture_bls_streaming", &["mnist_stream"]).await?;
    let (_, before) = sut.stream_counters().await?;

    let result = sut
        .infer_async(&[("cancel_stream", string("drop"))])
        .await?;

    let actual = response_parameter(&result, "stream_ended_with")?;
    let expected = string("dropped");
    sut.wait_for_stream_cancellation(before).await?;
    assert_eq!(actual, expected);
    Ok(())
}
