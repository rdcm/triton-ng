#[path = "common/mod.rs"]
mod common;

use crate::common::sut::Sut;
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn custom_backend_answers_from_cached_ram_buffers() -> Result<()> {
    let sut = Sut::new("fixture_memory_buffers", &[]).await?;
    let expected = sut.infer_digit().await?;

    let (digit, allocated) = sut.infer_digit_with_memory("ram").await?;
    let (_, allocated_after_reuse) = sut.infer_digit_with_memory("ram").await?;
    let (_, allocated_at_last) = sut.infer_digit_with_memory("ram").await?;

    let expected = (expected, true, allocated, allocated);
    let actual = (
        digit,
        allocated >= 1,
        allocated_after_reuse,
        allocated_at_last,
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_answers_from_cached_pinned_buffers() -> Result<()> {
    let sut = Sut::new("fixture_memory_buffers", &[]).await?;
    let expected = sut.infer_digit().await?;

    let (digit, allocated) = sut.infer_digit_with_memory("pinned").await?;
    let (_, allocated_after_reuse) = sut.infer_digit_with_memory("pinned").await?;
    let (_, allocated_at_last) = sut.infer_digit_with_memory("pinned").await?;

    let expected = (expected, true, allocated, allocated);
    let actual = (
        digit,
        allocated >= 1,
        allocated_after_reuse,
        allocated_at_last,
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_reuses_cached_gpu_output_buffers() -> Result<()> {
    let sut = Sut::new("fixture_memory_buffers", &[]).await?;
    let expected = sut.infer_digit().await?;

    let (digit, allocated) = sut.infer_digit_with_memory("gpu").await?;
    let (_, allocated_after_reuse) = sut.infer_digit_with_memory("gpu").await?;
    let (_, allocated_at_last) = sut.infer_digit_with_memory("gpu").await?;

    let expected = (expected, true, allocated, allocated);
    let actual = (
        digit,
        allocated >= 1,
        allocated_after_reuse,
        allocated_at_last,
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_passes_a_gpu_output_as_the_input_of_two_requests() -> Result<()> {
    let sut = Sut::new("fixture_memory_buffers", &[]).await?;
    let expected = sut.infer_digit().await?;

    let digit = sut
        .infer_with_parameters(HashMap::from([(
            "memory".to_string(),
            Parameter::String("gpu_input".to_string()),
        )]))
        .await?
        .outputs
        .into_iter()
        .find(|o| o.name == "DIGIT")
        .context("digit not found in response")?
        .data_i32;

    let actual = digit;
    let expected = vec![expected];
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_passes_a_gpu_input_of_its_request_on_without_a_copy() -> Result<()> {
    let sut = Sut::new("fixture_memory_buffers", &[]).await?;
    let expected = sut.infer_digit().await?;

    let digit = sut
        .infer_async(&[("memory", Parameter::String("gpu_forward".to_string()))])
        .await?
        .outputs
        .into_iter()
        .find(|o| o.name == "DIGIT")
        .context("digit not found in response")?
        .data_i32;

    let actual = digit;
    let expected = vec![expected];
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_fails_output_into_buffer_of_wrong_size() -> Result<()> {
    let sut = Sut::new("fixture_memory_buffers", &[]).await?;

    let error = sut
        .infer_digit_with_memory("wrong_size")
        .await
        .err()
        .context("output into a buffer of the wrong size succeeded")?;

    let actual = format!("{error:#}").contains("has 40 bytes, its buffer has 4");
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}
