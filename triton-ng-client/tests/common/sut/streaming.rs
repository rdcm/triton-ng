use crate::common::sut::Sut;
use crate::common::sut::infer_result;
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

impl Sut {
    /// Digits streamed by the decoupled model for `images` blank images, and the
    /// error the stream ended with, if any.
    pub(crate) async fn streamed_digits(
        &self,
        images: usize,
        parameter: Option<&str>,
    ) -> Result<(Vec<i32>, Option<triton_ng_client::Error>)> {
        let config = &self.config;

        let parameters = parameter
            .map(|name| HashMap::from([(name.to_string(), Parameter::Bool(true))]))
            .unwrap_or_default();
        let mut stream = self
            .infer_raw_stream(
                &config.stream_model,
                vec![config.mnist_blank_images(images)],
                &[&config.backend_output_name],
                parameters,
            )
            .await?;
        let mut digits = Vec::new();
        while let Some(response) = stream.next().await {
            match response {
                Ok(response) => {
                    let output = infer_result(response)
                        .outputs
                        .into_iter()
                        .find(|o| o.name == config.backend_output_name)
                        .context("digit not found in response")?;
                    if output.shape != vec![1, 1] {
                        return Err(anyhow::anyhow!("shape {:?}", output.shape));
                    }
                    digits.extend(output.data_i32);
                }
                Err(error) => return Ok((digits, Some(error))),
            }
        }
        Ok((digits, None))
    }

    /// `held_requests` and `cancelled_requests` reported by the decoupled model.
    pub(crate) async fn stream_counters(&self) -> Result<(i64, i64)> {
        let config = &self.config;

        let mut stream = self
            .infer_raw_stream(
                &config.stream_model,
                vec![config.mnist_blank_input()],
                &[&config.backend_output_name],
                HashMap::new(),
            )
            .await?;
        let response = stream.next().await.context("the stream is empty")??;
        let counter = |name: &str| match response.parameters.get(name) {
            Some(Parameter::Int(value)) => Ok(*value),
            other => return Err(anyhow::anyhow!("response parameter {name} is {other:?}")),
        };
        Ok((counter("held_requests")?, counter("cancelled_requests")?))
    }
}
