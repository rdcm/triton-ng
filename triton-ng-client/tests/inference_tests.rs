#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::sut::Sut;
use anyhow::{Context, Result};

#[tokio::test]
async fn mnist_onnx_inference_returns_ten_class_scores() -> Result<()> {
    let config = Config::isolated(
        "mnist_onnx_inference_returns_ten_class_scores",
        "mnist",
        &["minimal_backend", "minimal_batching", "mnist", "mnist_onnx"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let output_name = config.output_name.clone();
    let result = sut
        .infer(
            &config.mnist_model,
            None,
            vec![config.mnist_blank_input()],
            &[&output_name],
            None,
        )
        .await?;
    let output = result
        .outputs
        .iter()
        .find(|o| o.name == output_name)
        .context("output tensor not found in response")?;
    let expected = (vec![1, 10], 10);
    let actual = (output.shape.clone(), output.data_f32.len());
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_returns_the_digit_with_the_highest_target_model_score() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_returns_the_digit_with_the_highest_target_model_score",
        "mnist",
        &["minimal_backend", "minimal_batching", "mnist", "mnist_onnx"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let scores = sut
        .infer(
            &config.mnist_model,
            None,
            vec![config.mnist_blank_input()],
            &[&config.output_name],
            None,
        )
        .await?
        .outputs
        .into_iter()
        .find(|o| o.name == config.output_name)
        .context("scores not found in response")?
        .data_f32;
    let expected = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(digit, _)| digit as i32)
        .context("no scores")?;

    let digit = sut
        .infer(
            &config.backend_model,
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
    let expected = (vec![1, 1], vec![expected]);
    let actual = (digit.shape, digit.data_i32);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn inference_request_id_is_echoed_in_response() -> Result<()> {
    let config = Config::isolated(
        "inference_request_id_is_echoed_in_response",
        "mnist",
        &["minimal_backend", "minimal_batching", "mnist", "mnist_onnx"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let output_name = config.output_name.clone();
    let request_id = "test-echo-id-42".to_string();
    let result = sut
        .infer(
            &config.mnist_model,
            None,
            vec![config.mnist_blank_input()],
            &[&output_name],
            Some(request_id.clone()),
        )
        .await?;
    let actual = result.id;
    let expected = request_id;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn minimal_backend_forwards_its_input_over_bls() -> Result<()> {
    let config = Config::isolated(
        "minimal_backend_forwards_its_input_over_bls",
        "mnist",
        &["minimal_backend"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let data: Vec<f32> = (0..10).map(|value| value as f32 - 3.0).collect();
    let result = sut
        .infer(
            config.model("minimal_backend"),
            None,
            vec![common::dto::InferInput {
                name: "INPUT".to_string(),
                shape: vec![1, 10],
                data: data.clone(),
            }],
            &["OUTPUT"],
            None,
        )
        .await?;
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
    let config = Config::isolated(
        "minimal_batching_forwards_its_input_over_bls",
        "mnist",
        &["minimal_batching"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let data: Vec<f32> = (0..10).map(|value| value as f32 - 3.0).collect();
    let result = sut
        .infer(
            config.model("minimal_batching"),
            None,
            vec![common::dto::InferInput {
                name: "INPUT".to_string(),
                shape: vec![1, 10],
                data: data.clone(),
            }],
            &["OUTPUT"],
            None,
        )
        .await?;
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
