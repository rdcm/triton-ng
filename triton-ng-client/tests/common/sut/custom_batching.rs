use crate::common::sut::Sut;
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

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

impl Sut {
    /// Waits until the metric `name` of `model` reports a value `done` accepts.
    pub(crate) async fn wait_for_metric(
        &self,
        name: &str,
        model: &str,
        done: impl Fn(f64) -> bool,
    ) -> Result<f64> {
        for _ in 0..500 {
            let metrics = self.scrape_metrics()?;
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

    pub(crate) async fn check_batch_composition(&self) -> Result<HashMap<String, String>> {
        let config = &self.config;

        let model = config.backend_model.clone();
        let output = config.backend_output_name.clone();

        // Hold the sole instance so the four requests queue in the batcher.
        let mut holder = tokio::task::JoinSet::new();
        {
            let client = self.clone();
            let model = model.clone();
            let output = output.clone();
            let input = config.mnist_blank_input();
            holder.spawn(async move {
                client
                    .infer_raw_with_parameters(
                        &model,
                        vec![input],
                        &[&output],
                        HashMap::from([("hold_micros".to_string(), Parameter::Int(1_500_000))]),
                    )
                    .await
            });
        }
        self.wait_for_metric("triton_ng_example_held_requests", &model, |held| {
            held >= 1.0
        })
        .await?;

        let mut clients = tokio::task::JoinSet::new();
        for (index, key) in ["A", "A", "B", "A"].into_iter().enumerate() {
            let client = self.clone();
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
                    let response = client
                        .infer_raw_with_id_and_parameters(
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
            self.wait_for_metric("nv_inference_pending_request_count", &model, |pending| {
                pending >= queued
            })
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
    pub(crate) async fn wait_for_strategy_cleanup(&self, model: &str) -> Result<bool> {
        let config = &self.config;

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
}
