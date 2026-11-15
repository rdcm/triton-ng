#[path = "common/mod.rs"]
mod common;

use crate::common::sut::Sut;
use anyhow::Result;

#[tokio::test]
async fn custom_backend_reports_the_memory_the_model_and_its_instances_hold() -> Result<()> {
    let sut = Sut::new("mnist", &[]).await?;
    let usage = sut.model_memory_usage().await?;

    let cpu: Vec<_> = usage
        .iter()
        .filter(|usage| usage.memory_type == "CPU" && usage.id == 0)
        .collect();
    let expected = (1, true);
    let actual = (cpu.len(), cpu[0].byte_size > 0);
    assert_eq!(actual, expected);
    Ok(())
}

/// The fixture of `models/memory_usage` reports two entries of one memory type
/// and id and a pinned buffer that it really allocates; its sizes are the
/// constants of the example backend.
#[tokio::test]
async fn model_memory_reports_describe_their_types_and_sizes() -> Result<()> {
    let sut = Sut::new("memory_usage", &[]).await?;
    let usage = sut.model_memory_usage().await;
    let unloaded = sut.repository_model_unload().await;
    let usage = usage?;
    unloaded?;

    let cpu: Vec<_> = usage
        .iter()
        .filter(|usage| usage.memory_type == "CPU" && usage.id == 0)
        .collect();
    let expected = (1, 111 + 222);
    let actual = (cpu.len(), cpu[0].byte_size);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn instance_memory_reports_preserve_the_model_memory_totals() -> Result<()> {
    let sut = Sut::new("memory_usage", &[]).await?;
    let usage = sut.model_memory_usage().await;
    let unloaded = sut.repository_model_unload().await;
    let usage = usage?;
    unloaded?;

    let pinned: u64 = usage
        .iter()
        .filter(|usage| usage.memory_type == "CPU_PINNED")
        .map(|usage| usage.byte_size)
        .sum();
    let actual = pinned;
    let expected = 4096;
    assert_eq!(actual, expected);
    Ok(())
}
