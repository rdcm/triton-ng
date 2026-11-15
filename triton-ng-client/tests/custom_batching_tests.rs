#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::sut::{Sut, scrape_metrics};
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

/// The batching strategy of the `batch_strategy` fixture groups the requests of
/// a batch by their `batch_key`, and Triton lets it look no further than the
/// request that ended the batch.
#[tokio::test]
async fn batch_strategy_groups_adjacent_requests_by_key() -> Result<()> {
    let config = Config::isolated(
        "batch_strategy_groups_adjacent_requests_by_key",
        "mnist",
        &["batch_strategy"],
    )
    .await?;
    let actual = check_batch_composition(&config, config.model("batch_strategy")).await?;
    let expected = HashMap::from([
        ("1".to_string(), "A,A".to_string()),
        ("2".to_string(), "A,A".to_string()),
        ("3".to_string(), "B".to_string()),
        ("4".to_string(), "A".to_string()),
    ]);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn batch_strategy_example_groups_adjacent_requests_by_key() -> Result<()> {
    let config = Config::isolated(
        "batch_strategy_example_groups_adjacent_requests_by_key",
        "mnist",
        &["batch_strategy_example"],
    )
    .await?;
    let actual = check_batch_composition(&config, config.model("batch_strategy_example")).await?;
    let expected = HashMap::from([
        ("1".to_string(), "A,A".to_string()),
        ("2".to_string(), "A,A".to_string()),
        ("3".to_string(), "B".to_string()),
        ("4".to_string(), "A".to_string()),
    ]);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_batching_releases_unfinished_batches_on_repeated_unload() -> Result<()> {
    let config = Config::isolated(
        "custom_batching_releases_unfinished_batches_on_repeated_unload",
        "mnist",
        &[],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let model = config.model("batch_strategy_lifecycle");
    let mut actual = Vec::new();
    for _ in 0..3 {
        sut.repository_model_load(model).await?;
        sut.repository_model_unload(model).await?;
        actual.push(wait_for_strategy_cleanup(&config, model).await?);
    }
    let expected = vec![true; 3];
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_batching_contains_init_error() -> Result<()> {
    let config = Config::isolated("custom_batching_contains_init_error", "mnist", &[]).await?;
    let sut = Sut::new(&config.triton_url).await?;
    let model = config.model("batch_strategy_init_error");
    let failed = sut.repository_model_load(model).await.is_err();
    // Recover the retained failed version before unloading it.
    sut.repository_model_load(model).await?;
    sut.repository_model_unload(model).await?;
    let cleaned = wait_for_strategy_cleanup(&config, model).await?;
    let expected = (true, true, true);
    let actual = (failed, cleaned, sut.server_live().await?);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_batching_contains_init_panic() -> Result<()> {
    let config = Config::isolated("custom_batching_contains_init_panic", "mnist", &[]).await?;
    let sut = Sut::new(&config.triton_url).await?;
    let model = config.model("batch_strategy_init_panic");
    let failed = sut.repository_model_load(model).await.is_err();
    // Recover the retained failed version before unloading it.
    sut.repository_model_load(model).await?;
    sut.repository_model_unload(model).await?;
    let cleaned = wait_for_strategy_cleanup(&config, model).await?;
    let expected = (true, true, true);
    let actual = (failed, cleaned, sut.server_live().await?);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_batching_contains_batch_panic() -> Result<()> {
    let config = Config::isolated("custom_batching_contains_batch_panic", "mnist", &[]).await?;
    let sut = Sut::new(&config.triton_url).await?;
    let model = config.model("batch_strategy_batch_panic");
    sut.repository_model_load(model).await?;
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        sut.infer_with_parameters(
            model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("echo_batch_keys".to_string(), Parameter::Bool(true))]),
        ),
    )
    .await
    .context("strategy stalled past its queue timeout")?;
    let error = result
        .err()
        .context("a refusing strategy answered successfully")?;
    let message = format!("{error:#}");
    let answered = message.contains("timeout") || message.contains("expired");
    sut.repository_model_unload(model).await?;
    let cleaned = wait_for_strategy_cleanup(&config, model).await?;
    let expected = (true, true, true);
    let actual = (answered, cleaned, sut.server_live().await?);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_batching_contains_include_error() -> Result<()> {
    let config = Config::isolated("custom_batching_contains_include_error", "mnist", &[]).await?;
    let sut = Sut::new(&config.triton_url).await?;
    let model = config.model("batch_strategy_include_error");
    sut.repository_model_load(model).await?;
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        sut.infer_with_parameters(
            model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("echo_batch_keys".to_string(), Parameter::Bool(true))]),
        ),
    )
    .await
    .context("strategy stalled past its queue timeout")?;
    let error = result
        .err()
        .context("a refusing strategy answered successfully")?;
    let message = format!("{error:#}");
    let answered = message.contains("timeout") || message.contains("expired");
    sut.repository_model_unload(model).await?;
    let cleaned = wait_for_strategy_cleanup(&config, model).await?;
    let expected = (true, true, true);
    let actual = (answered, cleaned, sut.server_live().await?);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_batching_contains_include_panic() -> Result<()> {
    let config = Config::isolated("custom_batching_contains_include_panic", "mnist", &[]).await?;
    let sut = Sut::new(&config.triton_url).await?;
    let model = config.model("batch_strategy_include_panic");
    sut.repository_model_load(model).await?;
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        sut.infer_with_parameters(
            model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("echo_batch_keys".to_string(), Parameter::Bool(true))]),
        ),
    )
    .await
    .context("strategy stalled past its queue timeout")?;
    let error = result
        .err()
        .context("a refusing strategy answered successfully")?;
    let message = format!("{error:#}");
    let answered = message.contains("timeout") || message.contains("expired");
    sut.repository_model_unload(model).await?;
    let cleaned = wait_for_strategy_cleanup(&config, model).await?;
    let expected = (true, true, true);
    let actual = (answered, cleaned, sut.server_live().await?);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_batching_contains_refuse() -> Result<()> {
    let config = Config::isolated("custom_batching_contains_refuse", "mnist", &[]).await?;
    let sut = Sut::new(&config.triton_url).await?;
    let model = config.model("batch_strategy_refuse");
    sut.repository_model_load(model).await?;
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        sut.infer_with_parameters(
            model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("echo_batch_keys".to_string(), Parameter::Bool(true))]),
        ),
    )
    .await
    .context("strategy stalled past its queue timeout")?;
    let error = result
        .err()
        .context("a refusing strategy answered successfully")?;
    let message = format!("{error:#}");
    let answered = message.contains("timeout") || message.contains("expired");
    sut.repository_model_unload(model).await?;
    let cleaned = wait_for_strategy_cleanup(&config, model).await?;
    let expected = (true, true, true);
    let actual = (answered, cleaned, sut.server_live().await?);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_batching_contains_drop_panic() -> Result<()> {
    let config = Config::isolated("custom_batching_contains_drop_panic", "mnist", &[]).await?;
    let sut = Sut::new(&config.triton_url).await?;
    let model = config.model("batch_strategy_drop_panic");
    sut.repository_model_load(model).await?;
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        sut.infer_with_parameters(
            model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("echo_batch_keys".to_string(), Parameter::Bool(true))]),
        ),
    )
    .await
    .context("strategy stalled past its queue timeout")?;
    let answered = result.is_ok();
    sut.repository_model_unload(model).await?;
    let cleaned = wait_for_strategy_cleanup(&config, model).await?;
    let expected = (true, true, true);
    let actual = (answered, cleaned, sut.server_live().await?);
    assert_eq!(actual, expected);
    Ok(())
}

/// The value Triton reports for the metric `name` of `model`, summed over its
/// versions, `None` when the endpoint has no such series.
fn model_metric(metrics: &str, name: &str, model: &str) -> Result<Option<f64>> {
    let label = format!("model=\"{model}\"");
    let mut total: Option<f64> = None;
    for line in metrics.lines() {
        let Some((series, value)) = line.rsplit_once(' ') else {
            continue;
        };
        let Some(rest) = series.strip_prefix(name) else {
            continue;
        };
        if !rest.starts_with('{') || !rest.contains(&label) {
            continue;
        }
        let value: f64 = value.trim().parse().context(format!("{name} is {value}"))?;
        *total.get_or_insert(0.0) += value;
    }
    Ok(total)
}

/// Waits until the metric `name` of `model` reports a value `done` accepts.
async fn wait_for_metric(
    config: &Config,
    name: &str,
    model: &str,
    done: impl Fn(f64) -> bool,
) -> Result<f64> {
    for _ in 0..500 {
        let metrics = scrape_metrics(&config.metrics_address)?;
        if let Some(value) = model_metric(&metrics, name, model)?
            && done(value)
        {
            return Ok(value);
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    return Err(anyhow::anyhow!(
        "{name} of {model} never reached the value the test waits for"
    ));
}

async fn check_batch_composition(config: &Config, model: &str) -> Result<HashMap<String, String>> {
    let model = model.to_string();
    let output = config.backend_output_name.clone();

    // Hold the sole instance so the four requests queue in the batcher.
    let mut holder = tokio::task::JoinSet::new();
    {
        let url = config.triton_url.clone();
        let model = model.clone();
        let output = output.clone();
        let input = config.mnist_blank_input();
        holder.spawn(async move {
            Sut::new(&url)
                .await?
                .infer_with_parameters(
                    &model,
                    vec![input],
                    &[&output],
                    HashMap::from([("hold_micros".to_string(), Parameter::Int(1_500_000))]),
                )
                .await
        });
    }
    wait_for_metric(&config, "triton_ng_example_held_requests", &model, |held| {
        held >= 1.0
    })
    .await?;

    let mut clients = tokio::task::JoinSet::new();
    for (index, key) in ["A", "A", "B", "A"].into_iter().enumerate() {
        let url = config.triton_url.clone();
        let request_model = model.clone();
        let output = output.clone();
        let input = config.mnist_blank_input();
        let request_id = (index + 1).to_string();
        clients.spawn({
            let request_id = request_id.clone();
            async move {
                let parameters = HashMap::from([
                    ("batch_key".to_string(), Parameter::String(key.to_string())),
                    ("echo_batch_keys".to_string(), Parameter::Bool(true)),
                ]);
                let response = Sut::new(&url)
                    .await?
                    .infer_with_id_and_parameters(
                        &request_model,
                        vec![input],
                        &[&output],
                        &request_id,
                        parameters,
                    )
                    .await?;
                Ok::<_, anyhow::Error>((request_id, response))
            }
        });
        // The next request is sent only once this one is in the queue, so
        // Triton queues them in the order of this loop.
        let queued = f64::from(u32::try_from(index + 1)?);
        wait_for_metric(
            &config,
            "nv_inference_pending_request_count",
            &model,
            |pending| pending >= queued,
        )
        .await?;
    }

    let mut batches = HashMap::new();
    while let Some(result) = clients.join_next().await {
        let (request_id, result) = result??;
        match result.parameters.get("batch_keys") {
            Some(Parameter::String(keys)) => {
                batches.insert(request_id, keys.clone());
            }
            other => {
                return Err(anyhow::anyhow!(
                    "request {request_id} answered without batch keys: {other:?}"
                ));
            }
        }
    }
    while let Some(result) = holder.join_next().await {
        result??;
    }

    Ok(batches)
}

/// Waits for asynchronous model destruction and checks both values' drop counts.
async fn wait_for_strategy_cleanup(config: &Config, model: &str) -> Result<bool> {
    for _ in 0..100 {
        let path = config.trace_dir().join(format!("{model}.lifecycle.json"));
        if let Ok(contents) = std::fs::read_to_string(path)
            && let Ok([created, dropped, batches, released]) =
                serde_json::from_str::<[u64; 4]>(&contents)
            && created > 0
            && created == dropped
            && batches == released
        {
            return Ok(true);
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    Ok(false)
}
