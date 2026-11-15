use crate::common::sut::Sut;
use anyhow::Result;
use std::collections::HashMap;
use triton_ng_client::Parameter;

impl Sut {
    /// `held_requests` and `cancelled_requests` reported by the custom backend.
    pub(crate) async fn backend_counters(&self) -> Result<(i64, i64)> {
        let config = &self.config;

        let output_name = config.backend_output_name.clone();
        let result = self
            .infer_raw_with_parameters(
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

    pub(crate) async fn wait_for_backend_counters(
        &self,
        expected: impl Fn(i64, i64) -> bool,
    ) -> Result<()> {
        for _ in 0..10_000 {
            let (held, cancelled) = self.backend_counters().await?;
            if expected(held, cancelled) {
                return Ok(());
            }
        }
        return Err(anyhow::anyhow!(
            "backend counters did not reach the expected values"
        ));
    }

    pub(crate) async fn hold_request_until_cancelled(self) -> Result<()> {
        let config = &self.config;

        self.infer_raw_with_parameters(
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

    pub(crate) async fn wait_for_model_state(&self, model: &str, state: &str) -> Result<()> {
        for _ in 0..10_000 {
            let models = self.repository_index(false).await?;
            if models.iter().any(|m| m.name == model && m.state == state) {
                return Ok(());
            }
        }
        return Err(anyhow::anyhow!(
            "model {model} did not reach the {state} state"
        ));
    }
}
