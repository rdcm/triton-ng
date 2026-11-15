use crate::common::sut::Sut;
use anyhow::{Context, Result};
use std::collections::HashMap;
use tonic::Code;
use triton_ng_client::Parameter;

impl Sut {
    pub(crate) async fn grpc_code_of_failed_inference(
        &self,
        parameter: &str,
        value: &str,
    ) -> Result<Code> {
        let config = &self.config;

        let error = self
            .infer_raw_with_parameters(
                &config.backend_model,
                vec![config.mnist_blank_input()],
                &[&config.backend_output_name],
                HashMap::from([(parameter.to_string(), Parameter::String(value.to_string()))]),
            )
            .await
            .err()
            .context("the inference succeeded")?;
        match error.downcast_ref::<triton_ng_client::Error>() {
            Some(triton_ng_client::Error::Grpc(status)) => Ok(status.code()),
            _ => return Err(anyhow::anyhow!("not a gRPC error: {error:#}")),
        }
    }
}
