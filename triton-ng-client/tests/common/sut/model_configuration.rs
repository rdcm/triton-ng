use crate::common::sut::Sut;
use anyhow::Result;
use std::collections::HashMap;
use triton_ng_client::Parameter;

impl Sut {
    pub(crate) async fn instance_details(&self) -> Result<Vec<String>> {
        let config = &self.config;

        let echoed = self
            .infer_raw_with_parameters(
                self.model("instance_details"),
                vec![config.mnist_blank_input()],
                &[&config.backend_output_name],
                HashMap::from([("echo_instance_details".to_string(), Parameter::Bool(true))]),
            )
            .await?
            .parameters;
        let mut details: Vec<_> = echoed
            .into_iter()
            .filter(|(name, _)| name.starts_with("instance_"))
            .filter_map(|(_, value)| match value {
                Parameter::String(detail) => Some(detail),
                _ => None,
            })
            .collect();
        details.sort();
        Ok(details)
    }
}
