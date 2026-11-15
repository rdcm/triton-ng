#[path = "common/mod.rs"]
mod common;

use crate::common::sut::Sut;
use anyhow::{Context, Result};

#[tokio::test]
async fn custom_backend_answered_total_increases_after_inference() -> Result<()> {
    let sut = Sut::new("mnist", &[]).await?;
    sut.infer_digit().await?;
    let before = mnist_metric(
        &sut.scrape_metrics()?,
        "triton_ng_example_answered_total",
        sut.model("mnist"),
    )?
    .context("metric missing")?;
    sut.infer_digit().await?;
    let after = mnist_metric(
        &sut.scrape_metrics()?,
        "triton_ng_example_answered_total",
        sut.model("mnist"),
    )?
    .context("metric missing")?;
    let expected = true;
    let actual = after > before;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_bls_seconds_count_increases_after_inference() -> Result<()> {
    let sut = Sut::new("mnist", &[]).await?;
    sut.infer_digit().await?;
    let before = mnist_metric(
        &sut.scrape_metrics()?,
        "triton_ng_example_bls_seconds_count",
        sut.model("mnist"),
    )?
    .context("metric missing")?;
    sut.infer_digit().await?;
    let after = mnist_metric(
        &sut.scrape_metrics()?,
        "triton_ng_example_bls_seconds_count",
        sut.model("mnist"),
    )?
    .context("metric missing")?;
    let expected = true;
    let actual = after > before;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_exports_its_metric_families() -> Result<()> {
    let sut = Sut::new("mnist", &[]).await?;
    sut.infer_digit().await?;
    let metrics = sut.scrape_metrics()?;
    let names = [
        "triton_ng_example_answered_total",
        "triton_ng_example_bls_seconds_count",
        "triton_ng_example_held_requests",
    ];
    let mut present = Vec::new();
    for name in names {
        present.push(mnist_metric(&metrics, name, sut.model("mnist"))?.is_some());
    }
    let histogram = metrics.contains("# TYPE triton_ng_example_bls_seconds histogram");
    let bucket = metrics.contains(&format!(
        "triton_ng_example_bls_seconds_bucket{{model=\"{}\",le=\"0.01\"}}",
        sut.model("mnist")
    ));
    let expected = (vec![true; 3], true, true);
    let actual = (present, histogram, bucket);
    assert_eq!(actual, expected);
    Ok(())
}

/// The value of the series `name{model="mnist"}`, `None` when it is not
/// exported.
fn mnist_metric(metrics: &str, name: &str, model: &str) -> Result<Option<f64>> {
    let prefix = format!("{name}{{model=\"{model}\"}} ");
    metrics
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .map(|value| {
            value
                .trim()
                .parse::<f64>()
                .context(format!("{name} is {value}"))
        })
        .transpose()
}
