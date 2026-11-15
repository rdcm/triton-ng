#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::support::string;
use crate::common::sut::Sut;
use anyhow::Result;
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn custom_backend_answers_concurrent_requests() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_answers_concurrent_requests",
        "mnist",
        &["mnist"],
    )
    .await?;
    let mut tasks = tokio::task::JoinSet::new();
    for _ in 0..200 {
        let config = config.clone();
        tasks.spawn(async move {
            Sut::new(&config.triton_url)
                .await?
                .infer(
                    &config.backend_model,
                    None,
                    vec![config.mnist_blank_input()],
                    &[&config.backend_output_name],
                    None,
                )
                .await
                .map(|result| result.outputs.len())
        });
    }
    let mut actual = Vec::new();
    while let Some(result) = tasks.join_next().await {
        actual.push(result??);
    }
    let expected = vec![1; 200];
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn device_blocking_serializes_gpu_instances() -> Result<()> {
    let config = Config::isolated(
        "device_blocking_serializes_gpu_instances",
        "mnist",
        &["device_blocking"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let count = concurrent_executions(&sut, &config, config.model("device_blocking")).await?;
    let expected = true;
    let actual = count == 1;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn blocking_allows_cpu_instances_to_execute_concurrently() -> Result<()> {
    let config = Config::isolated(
        "blocking_allows_cpu_instances_to_execute_concurrently",
        "mnist",
        &["cpu_parallel"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let count = concurrent_executions(&sut, &config, config.model("cpu_parallel")).await?;
    let expected = true;
    let actual = count >= 2;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn backend_reports_blocking_execution_policy() -> Result<()> {
    let config = Config::isolated(
        "backend_reports_blocking_execution_policy",
        "mnist",
        &["fixture_backend_config"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let echoed = sut
        .infer_with_parameters(
            config.model("fixture_backend_config"),
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("echo_backend_config".to_string(), Parameter::Bool(true))]),
        )
        .await?
        .parameters;
    let expected = Some(string("Blocking"));
    let actual = echoed.get("execution_policy").cloned();
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn backend_reports_deviceblocking_execution_policy() -> Result<()> {
    let config = Config::isolated(
        "backend_reports_deviceblocking_execution_policy",
        "mnist",
        &["device_blocking"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let echoed = sut
        .infer_with_parameters(
            config.model("device_blocking"),
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("echo_backend_config".to_string(), Parameter::Bool(true))]),
        )
        .await?
        .parameters;
    let expected = Some(string("DeviceBlocking"));
    let actual = echoed.get("execution_policy").cloned();
    assert_eq!(actual, expected);
    Ok(())
}

/// Sends eight overlapping requests to `model` and answers the most execute
/// calls of that model that ran at once.
async fn concurrent_executions(sut: &Sut, config: &Config, model: &str) -> Result<i64> {
    let mut tasks = tokio::task::JoinSet::new();
    for _ in 0..8 {
        let model = model.to_string();
        let input = config.mnist_blank_input();
        let output = config.backend_output_name.clone();
        let url = config.triton_url.clone();
        tasks.spawn(async move {
            let sut = Sut::new(&url).await?;
            sut.infer_with_parameters(
                &model,
                vec![input],
                &[&output],
                HashMap::from([("hold_micros".to_string(), Parameter::Int(50_000))]),
            )
            .await
        });
    }
    while let Some(result) = tasks.join_next().await {
        result??;
    }

    let echoed = sut
        .infer_with_parameters(
            model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("echo_concurrency".to_string(), Parameter::Bool(true))]),
        )
        .await?
        .parameters;
    match echoed.get("max_in_flight") {
        Some(Parameter::Int(max)) => Ok(*max),
        other => return Err(anyhow::anyhow!("no maximum from {model}: {other:?}")),
    }
}
