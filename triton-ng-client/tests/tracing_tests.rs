#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::sut::Sut;
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn trace_settings_are_readable() -> Result<()> {
    let config = Config::isolated("trace_settings_are_readable", "mnist", &[]).await?;
    let sut = Sut::new(&config.triton_url).await?;
    let settings = sut.trace_settings(HashMap::new(), None).await?;
    let expected = false;
    let actual = settings.is_empty();
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn reporting_an_activity_without_tracing_succeeds() -> Result<()> {
    let config = Config::isolated(
        "reporting_an_activity_without_tracing_succeeds",
        "fixture_tracing",
        &["fixture_tracing"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let result = sut
        .infer_with_id_and_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
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
    let config = Config::isolated(
        "tracing_rejects_activity_names_with_quotes",
        "fixture_tracing",
        &["fixture_tracing"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let result = sut
        .infer_with_id_and_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
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
    let config = Config::isolated(
        "custom_backend_records_its_activity_inside_its_request",
        "fixture_tracing",
        &["fixture_tracing"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let request_id = config.backend_model.clone();
    let activity = format!("activity_{request_id}");
    set_mnist_tracing(&sut, &config, "TIMESTAMPS").await?;
    sut.infer_with_id_and_parameters(
        &config.backend_model,
        vec![config.mnist_blank_input()],
        &[&config.backend_output_name],
        &request_id,
        HashMap::from([(
            "report_activity".to_string(),
            Parameter::String(activity.clone()),
        )]),
    )
    .await?;
    set_mnist_tracing(&sut, &config, "OFF").await?;
    let mut found = false;
    for _ in 0..100 {
        found = activity_is_inside_its_request(&config, &request_id, &activity)?;
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
    let config = Config::isolated(
        "a_bls_call_is_traced_as_a_child_of_its_client_request",
        "fixture_tracing",
        &["fixture_tracing"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let request_id = config.backend_model.clone();
    set_mnist_tracing(&sut, &config, "TIMESTAMPS").await?;
    let mut tasks = tokio::task::JoinSet::new();
    for index in 0..1 {
        let config = config.clone();
        let id = format!("{request_id}-{index}");
        tasks.spawn(async move {
            Sut::new(&config.triton_url)
                .await?
                .infer_with_id_and_parameters(
                    &config.backend_model,
                    vec![config.mnist_blank_input()],
                    &[&config.backend_output_name],
                    &id,
                    HashMap::new(),
                )
                .await
        });
    }
    let mut responses = 0;
    while let Some(result) = tasks.join_next().await {
        result??;
        responses += 1;
    }
    set_mnist_tracing(&sut, &config, "OFF").await?;
    let first = format!("{request_id}-0");
    let mut found = false;
    for _ in 0..100 {
        found = has_traced_child(&trace_entries(&config)?, &first, &config.backend_model);
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
    let config = Config::isolated(
        "an_early_response_preserves_its_bls_child_trace",
        "fixture_tracing",
        &["fixture_tracing"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let request_id = config.backend_model.clone();
    set_mnist_tracing(&sut, &config, "TIMESTAMPS").await?;
    let mut tasks = tokio::task::JoinSet::new();
    for index in 0..50 {
        let config = config.clone();
        let id = format!("{request_id}-{index}");
        tasks.spawn(async move {
            Sut::new(&config.triton_url)
                .await?
                .infer_with_id_and_parameters(
                    &config.backend_model,
                    vec![config.mnist_blank_input()],
                    &[&config.backend_output_name],
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
    set_mnist_tracing(&sut, &config, "OFF").await?;
    let first = format!("{request_id}-0");
    let mut found = false;
    for _ in 0..100 {
        found = has_traced_child(&trace_entries(&config)?, &first, &config.backend_model);
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

/// Whether the trace of `request_id` places `activity` between its
/// `REQUEST_START` and its `REQUEST_END`.
///
/// Fails when a trace file that holds the activity is not JSON: the collector
/// writes the name into the file without escaping it, so a name it cannot
/// write breaks the file rather than the entry.
fn activity_is_inside_its_request(
    config: &Config,
    request_id: &str,
    activity: &str,
) -> Result<bool> {
    let Ok(dir) = std::fs::read_dir(config.trace_dir()) else {
        return Ok(false);
    };
    for file in dir {
        let path = file?.path();
        if !path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("trace.json"))
        {
            continue;
        }
        let content = std::fs::read_to_string(&path)?;
        if !content.contains(activity) {
            continue;
        }
        let entries: Vec<serde_json::Value> = serde_json::from_str(&content)
            .with_context(|| format!("{} does not hold JSON", path.display()))?;
        let Some(trace_id) = entries
            .iter()
            .find(|entry| entry["request_id"] == request_id)
            .and_then(|entry| entry["id"].as_u64())
        else {
            continue;
        };

        let mut start = None;
        let mut end = None;
        let mut reported = None;
        for entry in entries
            .iter()
            .filter(|entry| entry["id"].as_u64() == Some(trace_id))
        {
            let timestamps = entry["timestamps"].as_array().into_iter().flatten();
            for timestamp in timestamps {
                let Some(name) = timestamp["name"].as_str() else {
                    continue;
                };
                let Some(ns) = timestamp["ns"].as_u64() else {
                    continue;
                };
                if name == activity {
                    reported = Some(ns);
                } else if name == "REQUEST_START" {
                    start = Some(ns);
                } else if name == "REQUEST_END" {
                    end = Some(ns);
                }
            }
        }
        if let (Some(start), Some(reported), Some(end)) = (start, reported, end) {
            if start > reported || reported > end {
                return Err(anyhow::anyhow!(
                    "the activity {activity} is at {reported}, outside {start}..{end} of request {request_id}"
                ));
            }
            return Ok(true);
        }
    }
    Ok(false)
}

async fn set_mnist_tracing(sut: &Sut, config: &Config, level: &str) -> Result<()> {
    let settings = [
        ("trace_level", level),
        ("trace_rate", "1"),
        ("log_frequency", "1"),
    ];
    sut.trace_settings(
        settings
            .into_iter()
            .map(|(name, value)| (name.to_string(), vec![value.to_string()]))
            .collect(),
        Some(&config.backend_model),
    )
    .await?;
    Ok(())
}

/// Every trace entry written into the trace files so far.
fn trace_entries(config: &Config) -> Result<Vec<serde_json::Value>> {
    let mut entries = Vec::new();
    let Ok(dir) = std::fs::read_dir(config.trace_dir()) else {
        return Ok(entries);
    };
    for file in dir {
        let path = file?.path();
        if !path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("trace.json"))
        {
            continue;
        }
        let content = std::fs::read_to_string(&path)?;
        if let Ok(serde_json::Value::Array(values)) = serde_json::from_str(&content) {
            entries.extend(values);
        }
    }
    Ok(entries)
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
