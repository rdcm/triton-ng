#[path = "common/mod.rs"]
mod common;

use crate::common::support::string;
use crate::common::sut::Sut;
use anyhow::{Context, Result};
use triton_ng_client::Parameter;

#[tokio::test]
async fn response_statistics_ignore_a_unary_response_without_opt_in() -> Result<()> {
    let sut = Sut::new("fixture_response_statistics", &[]).await?;
    let before = sut.response_statistics().await?;
    sut.infer_async(&[("enable_statistics", Parameter::Bool(false))])
        .await?;
    let after = sut.response_statistics().await?;
    let expected = 0;
    let actual = after - before;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn response_statistics_ignore_stream_responses_without_opt_in() -> Result<()> {
    let sut = Sut::new("mnist_stream", &[]).await?;
    let before = sut.response_statistics().await?;
    let (digits, error) = sut.streamed_digits(2, None).await?;
    let after = sut.response_statistics().await?;
    let expected = (2, true, 0);
    let actual = (digits.len(), error.is_none(), after - before);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn response_statistics_record_an_enabled_unary_response() -> Result<()> {
    let sut = Sut::new("fixture_response_statistics", &[]).await?;
    let before = sut.response_statistics().await?;
    sut.infer_async(&[("enable_statistics", Parameter::Bool(true))])
        .await?;
    let after = sut.response_statistics().await?;
    let expected = 1;
    let actual = after - before;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn response_statistics_record_each_enabled_stream_response() -> Result<()> {
    let sut = Sut::new("mnist_stream", &[]).await?;
    let before = sut.response_statistics().await?;
    let (digits, error) = sut.streamed_digits(2, Some("enable_statistics")).await?;
    let after = sut.response_statistics().await?;
    let expected = (2, true, 2);
    let actual = (digits.len(), error.is_none(), after - before);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn response_statistics_count_a_response_enabled_twice_once() -> Result<()> {
    let sut = Sut::new("fixture_response_statistics", &[]).await?;
    let before = sut.response_statistics().await?;
    let work_micros = 50_000_i64;
    sut.infer_async(&[
        ("enable_statistics", Parameter::Bool(true)),
        ("hold_micros", Parameter::Int(work_micros)),
    ])
    .await?;
    let after = sut.response_statistics().await?;

    let expected = 1;
    let actual = after - before;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn response_statistics_include_work_done_before_opt_in() -> Result<()> {
    let sut = Sut::new("fixture_response_statistics", &[]).await?;
    let ns_before = sut.successful_response_ns().await?;
    let work_micros = 50_000_i64;
    sut.infer_async(&[
        ("enable_statistics", Parameter::Bool(true)),
        ("hold_micros", Parameter::Int(work_micros)),
    ])
    .await?;
    let ns_after = sut.successful_response_ns().await?;
    let expected = true;
    let actual = ns_after - ns_before >= u64::try_from(work_micros)? * 1_000;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn response_statistics_count_a_failed_response() -> Result<()> {
    let sut = Sut::new("fixture_response_statistics", &[]).await?;
    let (before, _) = sut.failed_and_cancelled_responses().await?;
    let error = sut
        .infer_async(&[
            ("error_code", string("INVALID_ARG")),
            ("enable_statistics", Parameter::Bool(true)),
        ])
        .await
        .err()
        .context("the inference succeeded")?;
    let (after, _) = sut.failed_and_cancelled_responses().await?;
    let expected = (true, 1);
    let actual = (
        format!("{error:#}").contains("error requested by the client"),
        after - before,
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn response_statistics_ignore_a_held_response_until_its_input_is_released() -> Result<()> {
    let sut = Sut::new("fixture_shared_inputs", &[]).await?;
    let before = sut.response_statistics().await?;
    let held = tokio::spawn(
        sut.clone()
            .infer_digit_keeping_shared_input_with_statistics(),
    );
    sut.wait_for_kept_inputs().await?;
    let held_count = sut.response_statistics().await?;
    sut.infer_async(&[("release_shared_inputs", Parameter::Bool(true))])
        .await?;
    held.await??;

    let expected = 0;
    let actual = held_count - before;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn response_statistics_record_a_held_response_after_its_input_is_released() -> Result<()> {
    let sut = Sut::new("fixture_shared_inputs", &[]).await?;
    let before = sut.response_statistics().await?;
    let held = tokio::spawn(
        sut.clone()
            .infer_digit_keeping_shared_input_with_statistics(),
    );
    sut.wait_for_kept_inputs().await?;
    sut.infer_async(&[("release_shared_inputs", Parameter::Bool(true))])
        .await?;
    held.await??;
    let sent_count = sut.response_statistics().await?;
    let expected = 1;
    let actual = sent_count - before;
    assert_eq!(actual, expected);
    Ok(())
}
