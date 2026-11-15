#[path = "common/mod.rs"]
mod common;

use crate::common::sut::Sut;
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn an_iterative_sequence_streams_its_first_step() -> Result<()> {
    let sut = Sut::new("mnist_steps", &[]).await?;

    let (numbers, error) = sut.streamed_steps(1).await?;
    let actual = (numbers, error.map(|e| e.to_string()));
    let expected = (vec![1], None);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn an_iterative_sequence_streams_each_rescheduled_step() -> Result<()> {
    let sut = Sut::new("mnist_steps", &[]).await?;

    let (numbers, error) = sut.streamed_steps(2).await?;
    let actual = (numbers, error.map(|e| e.to_string()));
    let expected = (vec![1, 2], None);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn an_iterative_sequence_stops_after_its_second_step() -> Result<()> {
    let sut = Sut::new("mnist_steps", &[]).await?;

    let (numbers, _error) = sut.streamed_steps(3).await?;
    let actual = numbers;
    let expected = vec![1, 2];
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn an_iterative_sequence_rejects_a_second_reschedule() -> Result<()> {
    let sut = Sut::new("mnist_steps", &[]).await?;

    let (_numbers, error) = sut.streamed_steps(3).await?;
    let error = error.context("the third step was rescheduled")?;
    let actual = error.to_string().contains("only once");
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn request_of_a_model_without_iterative_sequence_is_not_rescheduled() -> Result<()> {
    let sut = Sut::new("fixture_reschedule_refusal", &[]).await?;

    let error = sut
        .infer_with_parameters(HashMap::from([(
            "reschedule".to_string(),
            Parameter::Bool(true),
        )]))
        .await
        .err()
        .context("the request was answered")?;
    let actual = format!("{error:#}").contains("iterative_sequence");
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}
