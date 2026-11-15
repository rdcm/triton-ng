#[path = "common/mod.rs"]
mod common;

use crate::common::streaming::{stream_counters, streamed_digits};

use crate::common::config::Config;
use crate::common::support::infer_digit;
use crate::common::sut::Sut;
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn decoupled_model_streams_a_digit_per_image() -> Result<()> {
    let config = Config::isolated(
        "decoupled_model_streams_a_digit_per_image",
        "mnist",
        &["mnist", "mnist_stream"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let expected = infer_digit(&sut, &config).await?;

    let (digits, error) = streamed_digits(&sut, &config, 3, None).await?;

    let expected = (true, vec![expected; 3]);
    let actual = (error.is_none(), digits);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn decoupled_model_finishes_a_stream_without_responses() -> Result<()> {
    let config = Config::isolated(
        "decoupled_model_finishes_a_stream_without_responses",
        "mnist",
        &["mnist", "mnist_stream"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let (digits, error) = streamed_digits(&sut, &config, 0, None).await?;

    let expected = (true, true);
    let actual = (error.is_none(), digits.is_empty());
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn dropping_a_stream_sender_preserves_already_sent_responses() -> Result<()> {
    let config = Config::isolated(
        "dropping_a_stream_sender_preserves_already_sent_responses",
        "mnist",
        &["mnist", "mnist_stream"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let (digits, _error) = streamed_digits(&sut, &config, 2, Some("unfinished")).await?;

    let actual = digits.len();
    let expected = 2;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn dropping_a_stream_sender_reports_a_missing_final_response() -> Result<()> {
    let config = Config::isolated(
        "dropping_a_stream_sender_reports_a_missing_final_response",
        "mnist",
        &["mnist", "mnist_stream"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let (_digits, error) = streamed_digits(&sut, &config, 2, Some("unfinished")).await?;

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
    let config = Config::isolated(
        "request_of_a_model_that_is_not_decoupled_gets_no_sender",
        "fixture_sender_refusal",
        &["fixture_sender_refusal", "mnist_stream"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let error = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("into_sender".to_string(), Parameter::Bool(true))]),
        )
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
    let config = Config::isolated(
        "stream_cancelled_by_the_client_is_answered_as_cancelled",
        "mnist",
        &["mnist", "mnist_stream"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let (_, cancelled_before) = stream_counters(&sut, &config).await?;

    let mut stream = sut
        .infer_stream(
            &config.stream_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("hold_until_cancelled".to_string(), Parameter::Bool(true))]),
        )
        .await?;
    stream.next().await.context("the stream is empty")??;
    for _ in 0..10_000 {
        if stream_counters(&sut, &config).await?.0 == 1 {
            break;
        }
    }
    drop(stream);

    let mut actual = (1, cancelled_before);
    for _ in 0..10_000 {
        actual = stream_counters(&sut, &config).await?;
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
    let config = Config::isolated(
        "unary_inference_of_a_decoupled_model_is_rejected",
        "mnist",
        &["mnist", "mnist_stream"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let error = sut
        .infer(
            &config.stream_model,
            None,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            None,
        )
        .await
        .err()
        .context("the inference succeeded")?;

    let actual = format!("{error:#}").contains("decoupled");
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}
