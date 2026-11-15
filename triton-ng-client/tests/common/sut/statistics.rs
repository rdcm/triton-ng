use crate::common::sut::Sut;
use anyhow::{Context, Result};

impl Sut {
    /// Successful requests and executions of the custom backend model.
    pub(crate) async fn backend_statistics(&self) -> Result<(u64, u64)> {
        let config = &self.config;

        let stats = self
            .model_statistics_at(Some(&config.backend_model), None)
            .await?;
        let stats = stats
            .iter()
            .find(|s| s.name == config.backend_model)
            .context("backend model not found in statistics")?;
        Ok((stats.inference_count, stats.execution_count))
    }
}
