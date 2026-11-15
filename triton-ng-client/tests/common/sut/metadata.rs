use crate::common::sut::Sut;
use anyhow::Result;
use std::collections::HashMap;
use triton_ng_client::Parameter;

impl Sut {
    pub(crate) async fn echoed_metadata(
        &self,
        id: &str,
        parameters: &[(&str, Parameter)],
    ) -> Result<HashMap<String, Parameter>> {
        let config = &self.config;

        let result = self
            .infer_raw_with_id_and_parameters(
                &config.backend_model,
                vec![config.mnist_blank_input()],
                &[&config.backend_output_name],
                id,
                parameters
                    .iter()
                    .map(|(name, value)| (name.to_string(), value.clone()))
                    .collect(),
            )
            .await?;
        if result.id != id {
            return Err(anyhow::anyhow!("response id is {}", result.id));
        }
        Ok(result.parameters)
    }
}
