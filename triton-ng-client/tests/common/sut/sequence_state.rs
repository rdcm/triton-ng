use crate::common::sut::Sut;
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

impl Sut {
    /// Sends one request of the sequence `id` to the model whose requests are
    /// counted in its sequence state, and answers the counter it returned.
    pub(crate) async fn count_request(&self, id: i64, start: bool) -> Result<f32> {
        let config = &self.config;

        let mut parameters = HashMap::from([("sequence_id".to_string(), Parameter::Int(id))]);
        if start {
            parameters.insert("sequence_start".to_string(), Parameter::Bool(true));
        }
        let counter = self
            .infer_raw_with_parameters(
                &config.sequence_state_model,
                vec![config.mnist_blank_input()],
                &["COUNTER"],
                parameters,
            )
            .await?
            .outputs
            .into_iter()
            .find(|output| output.name == "COUNTER")
            .context("COUNTER not found in response")?
            .data_f32;
        counter.first().copied().context("the counter is empty")
    }
}
