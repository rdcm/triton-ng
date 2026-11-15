#[path = "common/mod.rs"]
mod common;

use crate::common::support::string;
use crate::common::sut::Sut;
use anyhow::Result;
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn custom_backend_answers_concurrent_requests() -> Result<()> {
    let sut = Sut::new("mnist", &[]).await?;
    let mut tasks = tokio::task::JoinSet::new();
    for _ in 0..200 {
        let sut = sut.clone();
        tasks.spawn(async move {
            sut.infer_async(&[])
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
    let sut = Sut::new("device_blocking", &[]).await?;
    let count = sut.concurrent_executions().await?;
    let expected = true;
    let actual = count == 1;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn blocking_allows_cpu_instances_to_execute_concurrently() -> Result<()> {
    let sut = Sut::new("cpu_parallel", &[]).await?;
    let count = sut.concurrent_executions().await?;
    let expected = true;
    let actual = count >= 2;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn backend_reports_blocking_execution_policy() -> Result<()> {
    let sut = Sut::new("fixture_backend_config", &[]).await?;
    let echoed = sut
        .infer_with_parameters(HashMap::from([(
            "echo_backend_config".to_string(),
            Parameter::Bool(true),
        )]))
        .await?
        .parameters;
    let expected = Some(string("Blocking"));
    let actual = echoed.get("execution_policy").cloned();
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn backend_reports_deviceblocking_execution_policy() -> Result<()> {
    let sut = Sut::new("device_blocking", &[]).await?;
    let echoed = sut
        .infer_with_parameters(HashMap::from([(
            "echo_backend_config".to_string(),
            Parameter::Bool(true),
        )]))
        .await?
        .parameters;
    let expected = Some(string("DeviceBlocking"));
    let actual = echoed.get("execution_policy").cloned();
    assert_eq!(actual, expected);
    Ok(())
}
