#[path = "common/mod.rs"]
mod common;

use std::collections::HashMap;

use crate::common::sut::Sut;
use anyhow::{Context, Result};

#[tokio::test]
async fn mnist_onnx_inference_returns_ten_class_scores() -> Result<()> {
    let sut = Sut::new("mnist_onnx", &[]).await?;
    let result = sut.infer_async(&[]).await?;
    let output = result
        .outputs
        .iter()
        .find(|o| o.name == "Plus214_Output_0")
        .context("output tensor not found in response")?;
    let expected = (vec![1, 10], 10);
    let actual = (output.shape.clone(), output.data_f32.len());
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_returns_the_digit_with_the_highest_target_model_score() -> Result<()> {
    let sut = Sut::new("mnist", &["mnist_onnx"]).await?;

    let scores = sut
        .infer_model_with_parameters("mnist_onnx", HashMap::new())
        .await?
        .outputs
        .into_iter()
        .find(|o| o.name == "Plus214_Output_0")
        .context("scores not found in response")?
        .data_f32;
    let expected = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(digit, _)| digit as i32)
        .context("no scores")?;

    let digit = sut
        .infer_async(&[])
        .await?
        .outputs
        .into_iter()
        .find(|o| o.name == "DIGIT")
        .context("digit not found in response")?;
    let expected = (vec![1, 1], vec![expected]);
    let actual = (digit.shape, digit.data_i32);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn inference_request_id_is_echoed_in_response() -> Result<()> {
    let sut = Sut::new("mnist_onnx", &[]).await?;
    let request_id = "test-echo-id-42".to_string();
    let result = sut
        .infer_with_id_and_parameters(&request_id, HashMap::new())
        .await?;
    let actual = result.id;
    let expected = request_id;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn minimal_backend_forwards_its_input_over_bls() -> Result<()> {
    let sut = Sut::new("minimal_backend", &[]).await?;
    let data: Vec<f32> = (0..10).map(|value| value as f32 - 3.0).collect();
    let result = sut.infer_with_input(data.clone(), HashMap::new()).await?;
    let output = result
        .outputs
        .into_iter()
        .find(|output| output.name == "OUTPUT")
        .context("OUTPUT not found")?;
    let expected = (vec![1, 10], data);
    let actual = (output.shape, output.data_f32);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn minimal_batching_forwards_its_input_over_bls() -> Result<()> {
    let sut = Sut::new("minimal_batching", &[]).await?;
    let data: Vec<f32> = (0..10).map(|value| value as f32 - 3.0).collect();
    let result = sut.infer_with_input(data.clone(), HashMap::new()).await?;
    let output = result
        .outputs
        .into_iter()
        .find(|output| output.name == "OUTPUT")
        .context("OUTPUT not found")?;
    let expected = (vec![1, 10], data);
    let actual = (output.shape, output.data_f32);
    assert_eq!(actual, expected);
    Ok(())
}
