#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::sut::{Sut, infer_result};
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn an_iterative_sequence_streams_its_first_step() -> Result<()> {
    let config = Config::isolated(
        "an_iterative_sequence_streams_its_first_step",
        "mnist",
        &["mnist", "mnist_steps"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let (numbers, error) = streamed_steps(&sut, &config, 1).await?;
    let actual = (numbers, error.map(|e| e.to_string()));
    let expected = (vec![1], None);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn an_iterative_sequence_streams_each_rescheduled_step() -> Result<()> {
    let config = Config::isolated(
        "an_iterative_sequence_streams_each_rescheduled_step",
        "mnist",
        &["mnist", "mnist_steps"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let (numbers, error) = streamed_steps(&sut, &config, 2).await?;
    let actual = (numbers, error.map(|e| e.to_string()));
    let expected = (vec![1, 2], None);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn an_iterative_sequence_stops_after_its_second_step() -> Result<()> {
    let config = Config::isolated(
        "an_iterative_sequence_stops_after_its_second_step",
        "mnist",
        &["mnist", "mnist_steps"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let (numbers, _error) = streamed_steps(&sut, &config, 3).await?;
    let actual = numbers;
    let expected = vec![1, 2];
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn an_iterative_sequence_rejects_a_second_reschedule() -> Result<()> {
    let config = Config::isolated(
        "an_iterative_sequence_rejects_a_second_reschedule",
        "mnist",
        &["mnist", "mnist_steps"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let (_numbers, error) = streamed_steps(&sut, &config, 3).await?;
    let error = error.context("the third step was rescheduled")?;
    let actual = error.to_string().contains("only once");
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn request_of_a_model_without_iterative_sequence_is_not_rescheduled() -> Result<()> {
    let config = Config::isolated(
        "request_of_a_model_without_iterative_sequence_is_not_rescheduled",
        "fixture_reschedule_refusal",
        &["fixture_reschedule_refusal", "mnist_steps"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let error = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("reschedule".to_string(), Parameter::Bool(true))]),
        )
        .await
        .err()
        .context("the request was answered")?;
    let actual = format!("{error:#}").contains("iterative_sequence");
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

async fn streamed_steps(
    sut: &Sut,
    config: &Config,
    steps: i64,
) -> Result<(Vec<i32>, Option<triton_ng_client::Error>)> {
    let mut stream = sut
        .infer_stream(
            "mnist_steps",
            vec![config.mnist_blank_images(1)],
            &[&config.backend_output_name],
            HashMap::from([("steps".to_string(), Parameter::Int(steps))]),
        )
        .await?;
    let mut numbers = Vec::new();
    while let Some(response) = stream.next().await {
        match response {
            Ok(response) => {
                let output = infer_result(response)
                    .outputs
                    .into_iter()
                    .find(|o| o.name == config.backend_output_name)
                    .context("step not found in response")?;
                numbers.extend(output.data_i32);
            }
            Err(error) => return Ok((numbers, Some(error))),
        }
    }
    Ok((numbers, None))
}
