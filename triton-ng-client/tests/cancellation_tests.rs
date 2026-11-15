#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::sut::Sut;
use anyhow::Result;
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn custom_backend_answers_cancelled_request_as_cancelled() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_answers_cancelled_request_as_cancelled",
        "fixture_cancellation",
        &["fixture_cancellation"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let (_, cancelled_before) = backend_counters(&sut, &config).await?;
    let (_, statistics_before) = sut
        .failed_and_cancelled_responses(&config.backend_model)
        .await?;

    let held = tokio::spawn(hold_request_until_cancelled(config.clone()));

    wait_for_backend_counters(&sut, &config, |held, _| held == 1).await?;
    held.abort();
    wait_for_backend_counters(&sut, &config, |held, cancelled| {
        held == 0 && cancelled == cancelled_before + 1
    })
    .await?;

    // The cancelled request is answered with an error the statistics record as a
    // cancellation, see `FinalResponse::send_error`.
    let (_, statistics_after) = sut
        .failed_and_cancelled_responses(&config.backend_model)
        .await?;

    let expected = 1;
    let actual = statistics_after - statistics_before;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_model_unloads_only_after_its_held_request_is_answered() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_model_unloads_only_after_its_held_request_is_answered",
        "fixture_cancellation",
        &["fixture_cancellation"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let held = tokio::spawn(hold_request_until_cancelled(config.clone()));
    wait_for_backend_counters(&sut, &config, |held, _| held == 1).await?;

    sut.repository_model_unload(&config.backend_model).await?;
    let unloading = wait_for_model_state(&sut, &config.backend_model, "UNLOADING").await;
    held.abort();
    let unloaded = wait_for_model_state(&sut, &config.backend_model, "UNAVAILABLE").await;
    sut.repository_model_load(&config.backend_model).await?;

    unloading?;
    unloaded?;
    let actual = sut.model_ready(&config.backend_model, None).await?;
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

/// `held_requests` and `cancelled_requests` reported by the custom backend.
async fn backend_counters(sut: &Sut, config: &Config) -> Result<(i64, i64)> {
    let output_name = config.backend_output_name.clone();
    let result = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&output_name],
            HashMap::new(),
        )
        .await?;
    let counter = |name: &str| match result.parameters.get(name) {
        Some(Parameter::Int(value)) => Ok(*value),
        other => return Err(anyhow::anyhow!("response parameter {name} is {other:?}")),
    };
    Ok((counter("held_requests")?, counter("cancelled_requests")?))
}

async fn wait_for_backend_counters(
    sut: &Sut,
    config: &Config,
    expected: impl Fn(i64, i64) -> bool,
) -> Result<()> {
    for _ in 0..10_000 {
        let (held, cancelled) = backend_counters(sut, config).await?;
        if expected(held, cancelled) {
            return Ok(());
        }
    }
    return Err(anyhow::anyhow!(
        "backend counters did not reach the expected values"
    ));
}

async fn hold_request_until_cancelled(config: Config) -> Result<()> {
    let sut = Sut::new(&config.triton_url).await?;
    sut.infer_with_parameters(
        &config.backend_model,
        vec![config.mnist_blank_input()],
        &[&config.backend_output_name],
        HashMap::from([
            ("hold_until_cancelled".to_string(), Parameter::Bool(true)),
            ("enable_statistics".to_string(), Parameter::Bool(true)),
        ]),
    )
    .await?;
    Ok(())
}

async fn wait_for_model_state(sut: &Sut, model: &str, state: &str) -> Result<()> {
    for _ in 0..10_000 {
        let models = sut.repository_index(false).await?;
        if models.iter().any(|m| m.name == model && m.state == state) {
            return Ok(());
        }
    }
    return Err(anyhow::anyhow!(
        "model {model} did not reach the {state} state"
    ));
}
