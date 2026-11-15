use crate::common::sut::{Sut, infer_result};
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

impl Sut {
    pub(crate) async fn streamed_steps(
        &self,
        steps: i64,
    ) -> Result<(Vec<i32>, Option<triton_ng_client::Error>)> {
        let config = &self.config;

        let mut stream = self
            .infer_raw_stream(
                &config.backend_model,
                vec![config.mnist_blank_images(1)],
                &[&config.backend_output_name],
                HashMap::from([("steps".to_string(), Parameter::Int(steps))]),
            )
            .await?;
        let mut numbers = Vec::new();
        while let Some(response) = stream.next().await {
            match response {
                Ok(response) => {
                    let output = infer_result(response)
                        .outputs
                        .into_iter()
                        .find(|o| o.name == config.backend_output_name)
                        .context("step not found in response")?;
                    numbers.extend(output.data_i32);
                }
                Err(error) => return Ok((numbers, Some(error))),
            }
        }
        Ok((numbers, None))
    }
}
