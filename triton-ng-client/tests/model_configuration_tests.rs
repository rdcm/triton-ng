#[path = "common/mod.rs"]
mod common;

use crate::common::sut::Sut;
use anyhow::{Context, Result};

#[tokio::test]
async fn model_with_a_config_completed_by_the_backend_answers() -> Result<()> {
    let sut = Sut::new("mnist_auto", &[]).await?;

    let digit = sut
        .infer_async(&[])
        .await?
        .outputs
        .into_iter()
        .find(|o| o.name == "DIGIT")
        .context("digit not found in response")?;
    let expected = (vec![1, 1], 1);
    let actual = (digit.shape, digit.data_i32.len());
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn backend_instance_details_list_every_instance() -> Result<()> {
    let sut = Sut::new("instance_details", &[]).await?;
    let details = sut.instance_details().await?;

    let expected = 2;
    let actual = details.len();
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn passive_instance_details_preserve_secondary_devices() -> Result<()> {
    let sut = Sut::new("instance_details", &[]).await?;
    let details = sut.instance_details().await?;
    let detail = details
        .iter()
        .find(|detail| detail.contains("passive=true"))
        .context("instance not found")?;
    let expected = true;
    let actual =
        detail.contains("secondary_devices=[SecondaryDevice { kind: \"KIND_NVDLA\", id: 7 }]");
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn active_instance_details_have_no_secondary_devices() -> Result<()> {
    let sut = Sut::new("instance_details", &[]).await?;
    let details = sut.instance_details().await?;
    let detail = details
        .iter()
        .find(|detail| detail.contains("passive=false"))
        .context("instance not found")?;
    let expected = true;
    let actual = detail.contains("secondary_devices=[]");
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn model_without_instance_group_gets_the_one_the_backend_prefers() -> Result<()> {
    let sut = Sut::new("mnist_stream", &[]).await?;

    let cfg = sut.model_config().await?.context("model config was None")?;
    let actual = cfg.instance_groups;
    let expected = vec![(2, 2)];
    assert_eq!(actual, expected);
    Ok(())
}
