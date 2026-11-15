#[path = "common/mod.rs"]
mod common;

use crate::common::support::string;
use crate::common::sut::Sut;
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn host_policy_pinned_reports_its_name_and_settings() -> Result<()> {
    let sut = Sut::new("host_policy_pinned", &[]).await?;
    let echoed = sut
        .infer_with_parameters(HashMap::from([(
            "echo_host_policy".to_string(),
            Parameter::Bool(true),
        )]))
        .await?
        .parameters;
    let expected = (Some(string("pinned")), Some(string("cpu-cores=0-7")));
    let actual = (
        echoed.get("host_policy_name").cloned(),
        echoed.get("host_policy_settings").cloned(),
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn host_policy_unknown_reports_its_name_and_settings() -> Result<()> {
    let sut = Sut::new("host_policy_unknown", &[]).await?;
    let echoed = sut
        .infer_with_parameters(HashMap::from([(
            "echo_host_policy".to_string(),
            Parameter::Bool(true),
        )]))
        .await?
        .parameters;
    let expected = (Some(string("unset_policy")), Some(string("")));
    let actual = (
        echoed.get("host_policy_name").cloned(),
        echoed.get("host_policy_settings").cloned(),
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn host_policy_default_reports_its_name_and_settings() -> Result<()> {
    let sut = Sut::new("host_policy_default", &[]).await?;
    let echoed = sut
        .infer_with_parameters(HashMap::from([(
            "echo_host_policy".to_string(),
            Parameter::Bool(true),
        )]))
        .await?
        .parameters;
    let expected = (Some(string("cpu")), Some(string("cpu-cores=0-7")));
    let actual = (
        echoed.get("host_policy_name").cloned(),
        echoed.get("host_policy_settings").cloned(),
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn configured_host_policy_preserves_input_metadata_and_bytes() -> Result<()> {
    let sut = Sut::new("host_policy_pinned", &[]).await?;
    let mut input = vec![0.0_f32; 784];
    for (index, value) in input.iter_mut().enumerate() {
        *value = index as f32 * 0.25 - 3.0;
    }
    let parameters = HashMap::from([
        ("echo_policy_input".to_string(), Parameter::Bool(true)),
        ("inspect_policy_input".to_string(), Parameter::Bool(true)),
    ]);
    let echoed = sut.infer_with_input(input, parameters).await?.parameters;
    let keys = [
        "policy_input_matches",
        "input_name",
        "input_datatype",
        "input_shape",
        "input_byte_size",
    ];
    let actual_values = keys.map(|key| echoed.get(key).cloned());
    let expected_values = [
        Some(Parameter::Bool(true)),
        Some(string("Input3")),
        Some(string("Fp32")),
        Some(string("[1, 1, 28, 28]")),
        Some(Parameter::Int(784 * 4)),
    ];

    let expected = expected_values;
    let actual = actual_values;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn unset_host_policy_preserves_input_metadata_and_bytes() -> Result<()> {
    let sut = Sut::new("host_policy_unknown", &[]).await?;
    let mut input = vec![0.0_f32; 784];
    for (index, value) in input.iter_mut().enumerate() {
        *value = index as f32 * 0.25 - 3.0;
    }
    let parameters = HashMap::from([
        ("echo_policy_input".to_string(), Parameter::Bool(true)),
        ("inspect_policy_input".to_string(), Parameter::Bool(true)),
    ]);
    let echoed = sut.infer_with_input(input, parameters).await?.parameters;
    let keys = [
        "policy_input_matches",
        "input_name",
        "input_datatype",
        "input_shape",
        "input_byte_size",
    ];
    let actual_values = keys.map(|key| echoed.get(key).cloned());
    let expected_values = [
        Some(Parameter::Bool(true)),
        Some(string("Input3")),
        Some(string("Fp32")),
        Some(string("[1, 1, 28, 28]")),
        Some(Parameter::Int(784 * 4)),
    ];

    let expected = expected_values;
    let actual = actual_values;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn batching_host_policy_preserves_input_metadata_and_bytes() -> Result<()> {
    let sut = Sut::new("host_policy_batching", &[]).await?;
    let mut input = vec![0.0_f32; 784];
    for (index, value) in input.iter_mut().enumerate() {
        *value = index as f32 * 0.25 - 3.0;
    }
    let parameters = HashMap::from([
        ("echo_policy_input".to_string(), Parameter::Bool(true)),
        ("inspect_policy_input".to_string(), Parameter::Bool(true)),
    ]);
    let echoed = sut.infer_with_input(input, parameters).await?.parameters;
    let keys = [
        "policy_input_matches",
        "input_name",
        "input_datatype",
        "input_shape",
        "input_byte_size",
    ];
    let actual_values = keys.map(|key| echoed.get(key).cloned());
    let expected_values = [
        Some(Parameter::Bool(true)),
        Some(string("Input3")),
        Some(string("Fp32")),
        Some(string("[1, 1, 28, 28]")),
        Some(Parameter::Int(784 * 4)),
    ];

    let expected = expected_values;
    let actual = actual_values;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn missing_host_policy_preserves_input_metadata_and_bytes() -> Result<()> {
    let sut = Sut::new("host_policy_pinned", &[]).await?;
    let mut input = vec![0.0_f32; 784];
    for (index, value) in input.iter_mut().enumerate() {
        *value = index as f32 * 0.25 - 3.0;
    }
    let mut parameters = HashMap::from([
        ("echo_policy_input".to_string(), Parameter::Bool(true)),
        ("inspect_policy_input".to_string(), Parameter::Bool(true)),
    ]);
    parameters.insert("input_policy".to_string(), string("missing-policy"));
    let echoed = sut.infer_with_input(input, parameters).await?.parameters;
    let keys = [
        "policy_input_matches",
        "input_name",
        "input_datatype",
        "input_shape",
        "input_byte_size",
    ];
    let actual_values = keys.map(|key| echoed.get(key).cloned());
    let expected_values = [
        Some(Parameter::Bool(true)),
        Some(string("Input3")),
        Some(string("Fp32")),
        Some(string("[1, 1, 28, 28]")),
        Some(Parameter::Int(784 * 4)),
    ];

    let expected = (expected_values, Some(string("missing-policy")));
    let actual = (actual_values, echoed.get("input_policy").cloned());
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn empty_host_policy_preserves_input_metadata_and_bytes() -> Result<()> {
    let sut = Sut::new("host_policy_pinned", &[]).await?;
    let mut input = vec![0.0_f32; 784];
    for (index, value) in input.iter_mut().enumerate() {
        *value = index as f32 * 0.25 - 3.0;
    }
    let mut parameters = HashMap::from([
        ("echo_policy_input".to_string(), Parameter::Bool(true)),
        ("inspect_policy_input".to_string(), Parameter::Bool(true)),
    ]);
    parameters.insert("input_policy".to_string(), string(""));
    let echoed = sut.infer_with_input(input, parameters).await?.parameters;
    let keys = [
        "policy_input_matches",
        "input_name",
        "input_datatype",
        "input_shape",
        "input_byte_size",
    ];
    let actual_values = keys.map(|key| echoed.get(key).cloned());
    let expected_values = [
        Some(Parameter::Bool(true)),
        Some(string("Input3")),
        Some(string("Fp32")),
        Some(string("[1, 1, 28, 28]")),
        Some(Parameter::Int(784 * 4)),
    ];

    let expected = (expected_values, Some(string("")));
    let actual = (actual_values, echoed.get("input_policy").cloned());
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn host_policy_input_rejects_nul_names() -> Result<()> {
    let sut = Sut::new("host_policy_pinned", &[]).await?;
    let error = sut
        .infer_with_parameters(HashMap::from([
            ("echo_policy_input".to_string(), Parameter::Bool(true)),
            ("invalid_policy_name".to_string(), Parameter::Bool(true)),
        ]))
        .await
        .err()
        .context("invalid host-policy arguments were accepted")?;
    let expected = true;
    let actual = format!("{error:#}").to_lowercase().contains("nul byte");
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn host_policy_input_rejects_out_of_range_buffers() -> Result<()> {
    let sut = Sut::new("host_policy_pinned", &[]).await?;
    let error = sut
        .infer_with_parameters(HashMap::from([
            ("echo_policy_input".to_string(), Parameter::Bool(true)),
            ("invalid_policy_index".to_string(), Parameter::Bool(true)),
        ]))
        .await
        .err()
        .context("invalid host-policy arguments were accepted")?;
    let expected = true;
    let actual = format!("{error:#}").to_lowercase().contains("out of range");
    assert_eq!(actual, expected);
    Ok(())
}
