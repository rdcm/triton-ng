#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::support::infer_digit;
use crate::common::sut::{Sut, scrape_metrics};
use anyhow::{Context, Result};

#[tokio::test]
async fn custom_backend_answered_total_increases_after_inference() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_answered_total_increases_after_inference",
        "mnist",
        &["mnist"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    infer_digit(&sut, &config).await?;
    let before = mnist_metric(
        &scrape_metrics(&config.metrics_address)?,
        "triton_ng_example_answered_total",
        &config.backend_model,
    )?
    .context("metric missing")?;
    infer_digit(&sut, &config).await?;
    let after = mnist_metric(
        &scrape_metrics(&config.metrics_address)?,
        "triton_ng_example_answered_total",
        &config.backend_model,
    )?
    .context("metric missing")?;
    let expected = true;
    let actual = after > before;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_bls_seconds_count_increases_after_inference() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_bls_seconds_count_increases_after_inference",
        "mnist",
        &["mnist"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    infer_digit(&sut, &config).await?;
    let before = mnist_metric(
        &scrape_metrics(&config.metrics_address)?,
        "triton_ng_example_bls_seconds_count",
        &config.backend_model,
    )?
    .context("metric missing")?;
    infer_digit(&sut, &config).await?;
    let after = mnist_metric(
        &scrape_metrics(&config.metrics_address)?,
        "triton_ng_example_bls_seconds_count",
        &config.backend_model,
    )?
    .context("metric missing")?;
    let expected = true;
    let actual = after > before;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_exports_its_metric_families() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_exports_its_metric_families",
        "mnist",
        &["mnist"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    infer_digit(&sut, &config).await?;
    let metrics = scrape_metrics(&config.metrics_address)?;
    let names = [
        "triton_ng_example_answered_total",
        "triton_ng_example_bls_seconds_count",
        "triton_ng_example_held_requests",
    ];
    let mut present = Vec::new();
    for name in names {
        present.push(mnist_metric(&metrics, name, &config.backend_model)?.is_some());
    }
    let histogram = metrics.contains("# TYPE triton_ng_example_bls_seconds histogram");
    let bucket = metrics.contains(&format!(
        "triton_ng_example_bls_seconds_bucket{{model=\"{}\",le=\"0.01\"}}",
        config.backend_model
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
