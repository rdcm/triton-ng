#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::sut::Sut;
use anyhow::Result;
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn bls_request_can_remove_and_rebuild_inputs_and_outputs() -> Result<()> {
    let config = Config::isolated(
        "bls_request_can_remove_and_rebuild_inputs_and_outputs",
        "fixture_bls_mutation",
        &["fixture_bls_mutation"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let result = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("check_bls_mutation".to_string(), Parameter::Bool(true))]),
        )
        .await;
    let expected = true;
    let actual = result.is_ok();
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn bls_request_can_append_distinct_host_policy_buffers() -> Result<()> {
    let config = Config::isolated(
        "bls_request_can_append_distinct_host_policy_buffers",
        "fixture_policy_append",
        &["fixture_policy_append"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let result = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("check_bls_policy_append".to_string(), Parameter::Bool(true))]),
        )
        .await;
    let expected = true;
    let actual = result.is_ok();
    assert_eq!(actual, expected);
    Ok(())
}
