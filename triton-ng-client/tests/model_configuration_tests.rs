#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::sut::Sut;
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn model_with_a_config_completed_by_the_backend_answers() -> Result<()> {
    let config = Config::isolated(
        "model_with_a_config_completed_by_the_backend_answers",
        "mnist",
        &["instance_details", "mnist", "mnist_auto", "mnist_stream"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let digit = sut
        .infer(
            config.model("mnist_auto"),
            None,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            None,
        )
        .await?
        .outputs
        .into_iter()
        .find(|o| o.name == config.backend_output_name)
        .context("digit not found in response")?;
    let expected = (vec![1, 1], 1);
    let actual = (digit.shape, digit.data_i32.len());
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn backend_instance_details_list_every_instance() -> Result<()> {
    let config = Config::isolated(
        "backend_instance_details_list_every_instance",
        "mnist",
        &["instance_details"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let details = instance_details(&sut, &config).await?;

    let expected = 2;
    let actual = details.len();
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn passive_instance_details_preserve_secondary_devices() -> Result<()> {
    let config = Config::isolated(
        "passive_instance_details_preserve_secondary_devices",
        "mnist",
        &["instance_details"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let details = instance_details(&sut, &config).await?;
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
    let config = Config::isolated(
        "active_instance_details_have_no_secondary_devices",
        "mnist",
        &["instance_details"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let details = instance_details(&sut, &config).await?;
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
    let config = Config::isolated(
        "model_without_instance_group_gets_the_one_the_backend_prefers",
        "mnist",
        &["instance_details", "mnist", "mnist_auto", "mnist_stream"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let cfg = sut
        .model_config(&config.stream_model, None)
        .await?
        .context("model config was None")?;
    let actual = cfg.instance_groups;
    let expected = vec![(2, 2)];
    assert_eq!(actual, expected);
    Ok(())
}

async fn instance_details(sut: &Sut, config: &Config) -> Result<Vec<String>> {
    let echoed = sut
        .infer_with_parameters(
            config.model("instance_details"),
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
