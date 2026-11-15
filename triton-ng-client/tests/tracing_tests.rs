#[path = "common/mod.rs"]
mod common;

use crate::common::sut::Sut;
use anyhow::Result;
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn trace_settings_are_readable() -> Result<()> {
    let sut = Sut::unloaded("mnist", &[]).await?;
    let settings = sut.trace_settings(HashMap::new()).await?;
    let expected = false;
    let actual = settings.is_empty();
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn reporting_an_activity_without_tracing_succeeds() -> Result<()> {
    let sut = Sut::new("fixture_tracing", &[]).await?;
    let result = sut
        .infer_with_id_and_parameters(
            "activity",
            HashMap::from([(
                "report_activity".to_string(),
                Parameter::String("decode".to_string()),
            )]),
        )
        .await;
    let expected = true;
    let actual = result.is_ok();
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn tracing_rejects_activity_names_with_quotes() -> Result<()> {
    let sut = Sut::new("fixture_tracing", &[]).await?;
    let result = sut
        .infer_with_id_and_parameters(
            "activity",
            HashMap::from([(
                "report_activity".to_string(),
                Parameter::String("decode \"first\"".to_string()),
            )]),
        )
        .await;
    let expected = false;
    let actual = result.is_ok();
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_records_its_activity_inside_its_request() -> Result<()> {
    let sut = Sut::new("fixture_tracing", &[]).await?;
    let request_id = sut.model("fixture_tracing").to_string();
    let activity = format!("activity_{request_id}");
    sut.set_tracing("TIMESTAMPS").await?;
    sut.infer_with_id_and_parameters(
        &request_id,
        HashMap::from([(
            "report_activity".to_string(),
            Parameter::String(activity.clone()),
        )]),
    )
    .await?;
    sut.set_tracing("OFF").await?;
    let mut found = false;
    for _ in 0..100 {
        found = sut.activity_is_inside_its_request(&request_id, &activity)?;
        if found {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let expected = true;
    let actual = found;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn a_bls_call_is_traced_as_a_child_of_its_client_request() -> Result<()> {
    let sut = Sut::new("fixture_tracing", &[]).await?;
    let request_id = sut.model("fixture_tracing").to_string();
    sut.set_tracing("TIMESTAMPS").await?;
    let mut tasks = tokio::task::JoinSet::new();
    for index in 0..1 {
        let sut = sut.clone();
        let id = format!("{request_id}-{index}");
        tasks.spawn(async move { sut.infer_with_id_and_parameters(&id, HashMap::new()).await });
    }
    let mut responses = 0;
    while let Some(result) = tasks.join_next().await {
        result??;
        responses += 1;
    }
    sut.set_tracing("OFF").await?;
    let first = format!("{request_id}-0");
    let mut found = false;
    for _ in 0..100 {
        found = has_traced_child(&sut.trace_entries()?, &first, sut.model("fixture_tracing"));
        if found {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let expected = (1, true);
    let actual = (responses, found);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn an_early_response_preserves_its_bls_child_trace() -> Result<()> {
    let sut = Sut::new("fixture_tracing", &[]).await?;
    let request_id = sut.model("fixture_tracing").to_string();
    sut.set_tracing("TIMESTAMPS").await?;
    let mut tasks = tokio::task::JoinSet::new();
    for index in 0..50 {
        let sut = sut.clone();
        let id = format!("{request_id}-{index}");
        tasks.spawn(async move {
            sut.infer_with_id_and_parameters(
                &id,
                HashMap::from([("answer_before_bls".to_string(), Parameter::Bool(true))]),
            )
            .await
        });
    }
    let mut responses = 0;
    while let Some(result) = tasks.join_next().await {
        result??;
        responses += 1;
    }
    sut.set_tracing("OFF").await?;
    let first = format!("{request_id}-0");
    let mut found = false;
    for _ in 0..100 {
        found = has_traced_child(&sut.trace_entries()?, &first, sut.model("fixture_tracing"));
        if found {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let expected = (50, true);
    let actual = (responses, found);
    assert_eq!(actual, expected);
    Ok(())
}

fn has_traced_child(entries: &[serde_json::Value], request_id: &str, model: &str) -> bool {
    let parents: Vec<_> = entries
        .iter()
        .filter(|entry| entry["request_id"] == request_id && entry["model_name"] == model)
        .filter_map(|entry| entry["id"].as_u64())
        .collect();
    entries.iter().any(|entry| {
        entry["model_name"] == "mnist_onnx"
            && entry["parent_id"]
                .as_u64()
                .is_some_and(|parent| parents.contains(&parent))
    })
}
