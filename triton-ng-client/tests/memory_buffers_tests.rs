#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::support::{infer_backend, infer_digit};
use crate::common::sut::Sut;
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

#[tokio::test]
async fn custom_backend_answers_from_cached_ram_buffers() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_answers_from_cached_ram_buffers",
        "fixture_memory_buffers",
        &["fixture_memory_buffers"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let expected = infer_digit(&sut, &config).await?;

    let (digit, allocated) = infer_digit_with_memory(&sut, &config, "ram").await?;
    let (_, allocated_after_reuse) = infer_digit_with_memory(&sut, &config, "ram").await?;
    let (_, allocated_at_last) = infer_digit_with_memory(&sut, &config, "ram").await?;

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
    let config = Config::isolated(
        "custom_backend_answers_from_cached_pinned_buffers",
        "fixture_memory_buffers",
        &["fixture_memory_buffers"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let expected = infer_digit(&sut, &config).await?;

    let (digit, allocated) = infer_digit_with_memory(&sut, &config, "pinned").await?;
    let (_, allocated_after_reuse) = infer_digit_with_memory(&sut, &config, "pinned").await?;
    let (_, allocated_at_last) = infer_digit_with_memory(&sut, &config, "pinned").await?;

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
    let config = Config::isolated(
        "custom_backend_reuses_cached_gpu_output_buffers",
        "fixture_memory_buffers",
        &["fixture_memory_buffers"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let expected = infer_digit(&sut, &config).await?;

    let (digit, allocated) = infer_digit_with_memory(&sut, &config, "gpu").await?;
    let (_, allocated_after_reuse) = infer_digit_with_memory(&sut, &config, "gpu").await?;
    let (_, allocated_at_last) = infer_digit_with_memory(&sut, &config, "gpu").await?;

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
    let config = Config::isolated(
        "custom_backend_passes_a_gpu_output_as_the_input_of_two_requests",
        "fixture_memory_buffers",
        &["fixture_memory_buffers"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let expected = infer_digit(&sut, &config).await?;

    let digit = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([(
                "memory".to_string(),
                Parameter::String("gpu_input".to_string()),
            )]),
        )
        .await?
        .outputs
        .into_iter()
        .find(|o| o.name == config.backend_output_name)
        .context("digit not found in response")?
        .data_i32;

    let actual = digit;
    let expected = vec![expected];
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_passes_a_gpu_input_of_its_request_on_without_a_copy() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_passes_a_gpu_input_of_its_request_on_without_a_copy",
        "fixture_memory_buffers",
        &["fixture_memory_buffers"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let expected = infer_digit(&sut, &config).await?;

    let digit = infer_backend(
        &sut,
        &config,
        &[("memory", Parameter::String("gpu_forward".to_string()))],
    )
    .await?
    .outputs
    .into_iter()
    .find(|o| o.name == config.backend_output_name)
    .context("digit not found in response")?
    .data_i32;

    let actual = digit;
    let expected = vec![expected];
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_fails_output_into_buffer_of_wrong_size() -> Result<()> {
    let config = Config::isolated(
        "custom_backend_fails_output_into_buffer_of_wrong_size",
        "fixture_memory_buffers",
        &["fixture_memory_buffers"],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;

    let error = infer_digit_with_memory(&sut, &config, "wrong_size")
        .await
        .err()
        .context("output into a buffer of the wrong size succeeded")?;

    let actual = format!("{error:#}").contains("has 40 bytes, its buffer has 4");
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

async fn infer_digit_with_memory(sut: &Sut, config: &Config, memory: &str) -> Result<(i32, i64)> {
    let result = sut
        .infer_with_parameters(
            &config.backend_model,
            vec![config.mnist_blank_input()],
            &[&config.backend_output_name],
            HashMap::from([("memory".to_string(), Parameter::String(memory.to_string()))]),
        )
        .await?;
    let digit = result
        .outputs
        .into_iter()
        .find(|o| o.name == config.backend_output_name)
        .context("digit not found in response")?
        .data_i32;
    let allocated = match result
        .parameters
        .get(&format!("{memory}_buffers_allocated"))
    {
        Some(Parameter::Int(value)) => *value,
        other => return Err(anyhow::anyhow!("{memory}_buffers_allocated is {other:?}")),
    };
    Ok((digit[0], allocated))
}
