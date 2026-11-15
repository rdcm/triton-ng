use crate::common::sut::Sut;
use anyhow::{Context, Result};
use triton_ng_client::Parameter;

impl Sut {
    pub(crate) async fn infer_async_error(&self, images: i64) -> Result<String> {
        let error = self
            .infer_async(&[
                ("stream_images", Parameter::Int(images)),
                ("via_infer_async", Parameter::Bool(true)),
            ])
            .await
            .err()
            .context("the inference succeeded")?;
        Ok(format!("{error:#}"))
    }

    pub(crate) async fn wait_for_stream_cancellation(&self, before: i64) -> Result<()> {
        for _ in 0..10_000 {
            let (held, cancelled) = self.stream_counters().await?;
            if held == 0 && cancelled == before + 1 {
                return Ok(());
            }
        }
        return Err(anyhow::anyhow!(
            "the decoupled model did not see the cancellation"
        ));
    }
}
