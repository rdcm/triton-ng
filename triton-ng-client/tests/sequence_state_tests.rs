#[path = "common/mod.rs"]
mod common;

use crate::common::sut::Sut;
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn sequence_state_counts_only_the_requests_of_its_own_sequence() -> Result<()> {
    let sut = Sut::new("sequence_state", &[]).await?;
    let mut first = Vec::new();
    let mut second = Vec::new();
    for step in 0..4 {
        if step < 3 {
            first.push(sut.count_request(41, step == 0).await?);
        }
        second.push(sut.count_request(42, step == 0).await?);
    }
    let expected = (vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0, 4.0]);
    let actual = (first, second);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn state_of_a_shape_the_configuration_does_not_declare_is_refused() -> Result<()> {
    let sut = Sut::new("sequence_state", &[]).await?;

    let error = sut
        .infer_with_parameters(HashMap::from([
            ("sequence_id".to_string(), Parameter::Int(42)),
            ("sequence_start".to_string(), Parameter::Bool(true)),
            ("bad_state_shape".to_string(), Parameter::Bool(true)),
        ]))
        .await
        .err()
        .context("a state of the wrong shape was created")?;

    let message = format!("{error:#}");
    let actual = message.contains("has the dimension 2 where the configuration declares 1");
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn state_of_another_model_is_refused() -> Result<()> {
    let sut = Sut::new("sequence_state_other", &["sequence_state"]).await?;

    // Use the FP32[1] declaration from sequence_state on an FP32[2] model.
    let error = sut
        .infer_with_parameters(HashMap::from([
            ("sequence_id".to_string(), Parameter::Int(41)),
            ("sequence_start".to_string(), Parameter::Bool(true)),
            (
                "foreign_state_model".to_string(),
                Parameter::String(sut.model("sequence_state").to_string()),
            ),
        ]))
        .await
        .err()
        .context("a state of another model was created")?;

    let message = format!("{error:#}");
    let expected = (true, true);
    let actual = (
        message.contains(&format!(
            "is declared by the model '{}'",
            sut.model("sequence_state")
        )),
        message.contains("the request runs on another model"),
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn state_of_a_shared_input_is_refused() -> Result<()> {
    let sut = Sut::new("sequence_state", &[]).await?;

    // The input state of the sequence is the memory of `counter_in`, and the
    // shared input hands that memory to another request: the state cannot be
    // written while it is out there.
    let error = sut
        .infer_with_parameters(HashMap::from([
            ("sequence_id".to_string(), Parameter::Int(43)),
            ("sequence_start".to_string(), Parameter::Bool(true)),
            ("shared_state_input".to_string(), Parameter::Bool(true)),
        ]))
        .await
        .err()
        .context("a state of a shared input was created")?;

    let message = format!("{error:#}");
    let actual = message.contains("input 'counter_in' of the state 'counter_out' is shared");
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn state_that_the_model_configuration_does_not_declare_is_refused() -> Result<()> {
    let sut = Sut::new("sequence_state", &[]).await?;

    let error = sut
        .infer_with_parameters(HashMap::from([
            ("sequence_id".to_string(), Parameter::Int(41)),
            ("sequence_start".to_string(), Parameter::Bool(true)),
            ("missing_state".to_string(), Parameter::Bool(true)),
        ]))
        .await
        .err()
        .context("the state was created")?;

    let message = format!("{error:#}");
    let expected = (true, true);
    let actual = (
        message.contains("no_such_state"),
        message.contains("the model configuration declares no state"),
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn reallocated_state_buffer_requires_a_new_write() -> Result<()> {
    let sut = Sut::new("sequence_state", &[]).await?;
    let error = sut
        .infer_with_parameters(HashMap::from([
            ("sequence_id".to_string(), Parameter::Int(331)),
            ("sequence_start".to_string(), Parameter::Bool(true)),
            ("sequence_end".to_string(), Parameter::Bool(true)),
            ("reallocate_state_buffer".to_string(), Parameter::Bool(true)),
        ]))
        .await
        .err()
        .context("the replacement buffer was accepted")?;
    let message = format!("{error:#}");
    let actual = message.contains("the state was never written");
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn batched_sequence_state_preserves_its_batch_dimension() -> Result<()> {
    let sut = Sut::new("sequence_state_batched", &[]).await?;
    let mut actual = Vec::new();
    for step in 0..3 {
        let result = sut
            .infer_with_parameters(HashMap::from([
                ("sequence_id".to_string(), Parameter::Int(332)),
                ("sequence_start".to_string(), Parameter::Bool(step == 0)),
                ("sequence_end".to_string(), Parameter::Bool(step == 2)),
                ("check_batched_state".to_string(), Parameter::Bool(true)),
            ]))
            .await?;
        let counter = result
            .outputs
            .into_iter()
            .find(|output| output.name == "COUNTER")
            .context("COUNTER not found")?;
        actual.push(counter.data_f32);
    }
    let expected = vec![vec![1.0], vec![2.0], vec![3.0]];
    assert_eq!(actual, expected);
    Ok(())
}
