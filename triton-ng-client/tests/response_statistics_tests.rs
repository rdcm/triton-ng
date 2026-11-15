#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::shared_inputs::{
    infer_digit_keeping_shared_input_with_statistics, wait_for_kept_inputs,
};
use crate::common::streaming::streamed_digits;
use crate::common::support::{infer_backend, string};
use crate::common::sut::Sut;
use anyhow::{Context, Result};
use triton_ng_client::Parameter;

#[tokio::test]
async fn response_statistics_ignore_a_unary_response_without_opt_in() -> Result<()> {
    let config = Config::isolated(
        "response_statistics_ignore_a_unary_response_without_opt_in",
        "fixture_response_statistics",
        &["fixture_response_statistics"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let before = sut.response_statistics(&config.backend_model).await?;
    infer_backend(
        &sut,
        &config,
        &[("enable_statistics", Parameter::Bool(false))],
    )
    .await?;
    let after = sut.response_statistics(&config.backend_model).await?;
    let expected = 0;
    let actual = after - before;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn response_statistics_ignore_stream_responses_without_opt_in() -> Result<()> {
    let config = Config::isolated(
        "response_statistics_ignore_stream_responses_without_opt_in",
        "fixture_response_statistics",
        &["fixture_response_statistics", "mnist_stream"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let before = sut.response_statistics(&config.stream_model).await?;
    let (digits, error) = streamed_digits(&sut, &config, 2, None).await?;
    let after = sut.response_statistics(&config.stream_model).await?;
    let expected = (2, true, 0);
    let actual = (digits.len(), error.is_none(), after - before);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn response_statistics_record_an_enabled_unary_response() -> Result<()> {
    let config = Config::isolated(
        "response_statistics_record_an_enabled_unary_response",
        "fixture_response_statistics",
        &["fixture_response_statistics"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let before = sut.response_statistics(&config.backend_model).await?;
    infer_backend(
        &sut,
        &config,
        &[("enable_statistics", Parameter::Bool(true))],
    )
    .await?;
    let after = sut.response_statistics(&config.backend_model).await?;
    let expected = 1;
    let actual = after - before;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn response_statistics_record_each_enabled_stream_response() -> Result<()> {
    let config = Config::isolated(
        "response_statistics_record_each_enabled_stream_response",
        "fixture_response_statistics",
        &["fixture_response_statistics", "mnist_stream"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let before = sut.response_statistics(&config.stream_model).await?;
    let (digits, error) = streamed_digits(&sut, &config, 2, Some("enable_statistics")).await?;
    let after = sut.response_statistics(&config.stream_model).await?;
    let expected = (2, true, 2);
    let actual = (digits.len(), error.is_none(), after - before);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn response_statistics_count_a_response_enabled_twice_once() -> Result<()> {
    let config = Config::isolated(
        "response_statistics_count_a_response_enabled_twice_once",
        "fixture_response_statistics",
        &["fixture_response_statistics"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let before = sut.response_statistics(&config.backend_model).await?;
    let work_micros = 50_000_i64;
    infer_backend(
        &sut,
        &config,
        &[
            ("enable_statistics", Parameter::Bool(true)),
            ("hold_micros", Parameter::Int(work_micros)),
        ],
    )
    .await?;
    let after = sut.response_statistics(&config.backend_model).await?;

    let expected = 1;
    let actual = after - before;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn response_statistics_include_work_done_before_opt_in() -> Result<()> {
    let config = Config::isolated(
        "response_statistics_include_work_done_before_opt_in",
        "fixture_response_statistics",
        &["fixture_response_statistics"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let ns_before = sut.successful_response_ns(&config.backend_model).await?;
    let work_micros = 50_000_i64;
    infer_backend(
        &sut,
        &config,
        &[
            ("enable_statistics", Parameter::Bool(true)),
            ("hold_micros", Parameter::Int(work_micros)),
        ],
    )
    .await?;
    let ns_after = sut.successful_response_ns(&config.backend_model).await?;
    let expected = true;
    let actual = ns_after - ns_before >= u64::try_from(work_micros)? * 1_000;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn response_statistics_count_a_failed_response() -> Result<()> {
    let config = Config::isolated(
        "response_statistics_count_a_failed_response",
        "fixture_response_statistics",
        &["fixture_response_statistics"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let (before, _) = sut
        .failed_and_cancelled_responses(&config.backend_model)
        .await?;
    let error = infer_backend(
        &sut,
        &config,
        &[
            ("error_code", string("INVALID_ARG")),
            ("enable_statistics", Parameter::Bool(true)),
        ],
    )
    .await
    .err()
    .context("the inference succeeded")?;
    let (after, _) = sut
        .failed_and_cancelled_responses(&config.backend_model)
        .await?;
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
    let config = Config::isolated(
        "response_statistics_ignore_a_held_response_until_its_input_is_released",
        "fixture_shared_inputs",
        &["fixture_shared_inputs"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let before = sut.response_statistics(&config.backend_model).await?;
    let held = tokio::spawn(infer_digit_keeping_shared_input_with_statistics(
        config.clone(),
    ));
    wait_for_kept_inputs(&sut, &config).await?;
    let held_count = sut.response_statistics(&config.backend_model).await?;
    infer_backend(
        &sut,
        &config,
        &[("release_shared_inputs", Parameter::Bool(true))],
    )
    .await?;
    held.await??;

    let expected = 0;
    let actual = held_count - before;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn response_statistics_record_a_held_response_after_its_input_is_released() -> Result<()> {
    let config = Config::isolated(
        "response_statistics_record_a_held_response_after_its_input_is_released",
        "fixture_shared_inputs",
        &["fixture_shared_inputs"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let before = sut.response_statistics(&config.backend_model).await?;
    let held = tokio::spawn(infer_digit_keeping_shared_input_with_statistics(
        config.clone(),
    ));
    wait_for_kept_inputs(&sut, &config).await?;
    infer_backend(
        &sut,
        &config,
        &[("release_shared_inputs", Parameter::Bool(true))],
    )
    .await?;
    held.await??;
    let sent_count = sut.response_statistics(&config.backend_model).await?;
    let expected = 1;
    let actual = sent_count - before;
    assert_eq!(actual, expected);
    Ok(())
}
