use crate::common::support::response_parameter;
use crate::common::sut::Sut;
use anyhow::{Context, Result};
use triton_ng_client::Parameter;

impl Sut {
    pub(crate) async fn kept_inputs(&self) -> Result<i64> {
        let result = self.infer_async(&[]).await?;
        match response_parameter(&result, "kept_inputs")? {
            Parameter::Int(kept) => Ok(kept),
            other => return Err(anyhow::anyhow!("kept_inputs is {other:?}")),
        }
    }

    pub(crate) async fn wait_for_kept_inputs(&self) -> Result<()> {
        let mut kept = 0;
        for _ in 0..10_000 {
            kept = self.kept_inputs().await?;
            if kept == 1 {
                return Ok(());
            }
        }
        return Err(anyhow::anyhow!(
            "the backend kept {kept} inputs of its requests"
        ));
    }

    pub(crate) async fn infer_digit_keeping_shared_input_with_statistics(self) -> Result<i32> {
        let config = &self.config;

        let result = self
            .infer_async(&[
                ("shared_input", Parameter::Bool(true)),
                ("enable_statistics", Parameter::Bool(true)),
            ])
            .await?;
        let digit = result
            .outputs
            .into_iter()
            .find(|o| o.name == config.backend_output_name)
            .context("digit not found in response")?
            .data_i32;
        digit.first().copied().context("the digit output is empty")
    }
}

impl Sut {
    pub(crate) async fn infer_digit_with_shared_input(&self, keep: bool) -> Result<i32> {
        let config = &self.config;

        let result = self
            .infer_async(&[("shared_input", Parameter::Bool(keep))])
            .await?;
        let digit = result
            .outputs
            .into_iter()
            .find(|o| o.name == config.backend_output_name)
            .context("digit not found in response")?
            .data_i32;
        digit.first().copied().context("the digit output is empty")
    }

    pub(crate) async fn infer_digit_keeping_shared_input(self) -> Result<i32> {
        self.infer_digit_with_shared_input(true).await
    }
}
