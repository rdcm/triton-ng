use crate::common::sut::Sut;
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

impl Sut {
    pub(crate) async fn infer_digit_with_memory(&self, memory: &str) -> Result<(i32, i64)> {
        let config = &self.config;

        let result = self
            .infer_raw_with_parameters(
                &config.backend_model,
                vec![config.mnist_blank_input()],
                &[&config.backend_output_name],
                HashMap::from([("memory".to_string(), Parameter::String(memory.to_string()))]),
            )
            .await?;
        let digit = result
            .outputs
            .into_iter()
            .find(|o| o.name == config.backend_output_name)
            .context("digit not found in response")?
            .data_i32;
        let allocated = match result
            .parameters
            .get(&format!("{memory}_buffers_allocated"))
        {
            Some(Parameter::Int(value)) => *value,
            other => return Err(anyhow::anyhow!("{memory}_buffers_allocated is {other:?}")),
        };
        Ok((digit[0], allocated))
    }
}
