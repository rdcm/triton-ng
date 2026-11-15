use crate::common::sut::Sut;
use anyhow::Result;
use std::collections::HashMap;
use triton_ng_client::Parameter;

impl Sut {
    /// Sends eight overlapping requests to `model` and answers the most execute
    /// calls of that model that ran at once.
    pub(crate) async fn concurrent_executions(&self) -> Result<i64> {
        let config = &self.config;
        let model = &config.backend_model;

        let mut tasks = tokio::task::JoinSet::new();
        for _ in 0..8 {
            let model = model.to_string();
            let input = config.mnist_blank_input();
            let output = config.backend_output_name.clone();
            let client = self.clone();
            tasks.spawn(async move {
                client
                    .infer_raw_with_parameters(
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

        let echoed = self
            .infer_raw_with_parameters(
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
}
