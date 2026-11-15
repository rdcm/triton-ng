#[path = "common/mod.rs"]
mod common;

use crate::common::sut::Sut;
use anyhow::Result;
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn bls_request_can_remove_and_rebuild_inputs_and_outputs() -> Result<()> {
    let sut = Sut::new("fixture_bls_mutation", &[]).await?;
    let result = sut
        .infer_with_parameters(HashMap::from([(
            "check_bls_mutation".to_string(),
            Parameter::Bool(true),
        )]))
        .await;
    let expected = true;
    let actual = result.is_ok();
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn bls_request_can_append_distinct_host_policy_buffers() -> Result<()> {
    let sut = Sut::new("fixture_policy_append", &[]).await?;
    let result = sut
        .infer_with_parameters(HashMap::from([(
            "check_bls_policy_append".to_string(),
            Parameter::Bool(true),
        )]))
        .await;
    let expected = true;
    let actual = result.is_ok();
    assert_eq!(actual, expected);
    Ok(())
}
