use crate::common;
use crate::common::sut::Sut;
use anyhow::{Context, Result};
use triton_ng_client::Parameter;

impl Sut {
    pub(crate) async fn infer_digit(&self) -> Result<i32> {
        let config = &self.config;

        let digit = self
            .infer_raw(
                &config.backend_model,
                None,
                vec![config.mnist_blank_input()],
                &[&config.backend_output_name],
                None,
            )
            .await?
            .outputs
            .into_iter()
            .find(|o| o.name == config.backend_output_name)
            .context("digit not found in response")?
            .data_i32;
        digit.first().copied().context("the digit output is empty")
    }

    pub(crate) async fn infer_async(
        &self,
        parameters: &[(&str, Parameter)],
    ) -> Result<common::dto::InferResult> {
        self.infer_with_parameters(
            parameters
                .iter()
                .map(|(name, value)| (name.to_string(), value.clone()))
                .collect(),
        )
        .await
    }
}
