#[path = "common/mod.rs"]
mod common;

use crate::common::sut::Sut;
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn decoupled_model_streams_a_digit_per_image() -> Result<()> {
    let sut = Sut::new("mnist", &["mnist_stream"]).await?;
    let expected = sut.infer_digit().await?;

    let (digits, error) = sut.streamed_digits(3, None).await?;

    let expected = (true, vec![expected; 3]);
    let actual = (error.is_none(), digits);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn decoupled_model_finishes_a_stream_without_responses() -> Result<()> {
    let sut = Sut::new("mnist_stream", &[]).await?;

    let (digits, error) = sut.streamed_digits(0, None).await?;

    let expected = (true, true);
    let actual = (error.is_none(), digits.is_empty());
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn dropping_a_stream_sender_preserves_already_sent_responses() -> Result<()> {
    let sut = Sut::new("mnist_stream", &[]).await?;

    let (digits, _error) = sut.streamed_digits(2, Some("unfinished")).await?;

    let actual = digits.len();
    let expected = 2;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn dropping_a_stream_sender_reports_a_missing_final_response() -> Result<()> {
    let sut = Sut::new("mnist_stream", &[]).await?;

    let (_digits, error) = sut.streamed_digits(2, Some("unfinished")).await?;

    let error = error.context("the stream ended without an error")?;
    let actual = error
        .to_string()
        .contains("did not produce a final response");
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn request_of_a_model_that_is_not_decoupled_gets_no_sender() -> Result<()> {
    let sut = Sut::new("fixture_sender_refusal", &[]).await?;

    let error = sut
        .infer_with_parameters(HashMap::from([(
            "into_sender".to_string(),
            Parameter::Bool(true),
        )]))
        .await
        .err()
        .context("the inference succeeded")?;

    let actual = format!("{error:#}").contains("only requests of a decoupled model");
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn stream_cancelled_by_the_client_is_answered_as_cancelled() -> Result<()> {
    let sut = Sut::new("mnist_stream", &[]).await?;
    let (_, cancelled_before) = sut.stream_counters().await?;

    let mut stream = sut
        .infer_stream(HashMap::from([(
            "hold_until_cancelled".to_string(),
            Parameter::Bool(true),
        )]))
        .await?;
    stream.next().await.context("the stream is empty")??;
    for _ in 0..10_000 {
        if sut.stream_counters().await?.0 == 1 {
            break;
        }
    }
    drop(stream);

    let mut actual = (1, cancelled_before);
    for _ in 0..10_000 {
        actual = sut.stream_counters().await?;
        if actual == (0, cancelled_before + 1) {
            break;
        }
        tokio::task::yield_now().await;
    }
    let expected = (0, cancelled_before + 1);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn unary_inference_of_a_decoupled_model_is_rejected() -> Result<()> {
    let sut = Sut::new("mnist_stream", &[]).await?;

    let error = sut
        .infer_async(&[])
        .await
        .err()
        .context("the inference succeeded")?;

    let actual = format!("{error:#}").contains("decoupled");
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}
