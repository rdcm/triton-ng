#[path = "common/mod.rs"]
mod common;

use crate::common::sut::Sut;
use anyhow::{Context, Result};
use std::collections::HashMap;
use triton_ng_client::Parameter;

/// The batching strategy of the `batch_strategy` fixture groups the requests of
/// a batch by their `batch_key`, and Triton lets it look no further than the
/// request that ended the batch.
#[tokio::test]
async fn batch_strategy_groups_adjacent_requests_by_key() -> Result<()> {
    let sut = Sut::new("batch_strategy", &[]).await?;
    let actual = sut.check_batch_composition().await?;
    let expected = HashMap::from([
        ("1".to_string(), "A,A".to_string()),
        ("2".to_string(), "A,A".to_string()),
        ("3".to_string(), "B".to_string()),
        ("4".to_string(), "A".to_string()),
    ]);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn batch_strategy_example_groups_adjacent_requests_by_key() -> Result<()> {
    let sut = Sut::new("batch_strategy_example", &[]).await?;
    let actual = sut.check_batch_composition().await?;
    let expected = HashMap::from([
        ("1".to_string(), "A,A".to_string()),
        ("2".to_string(), "A,A".to_string()),
        ("3".to_string(), "B".to_string()),
        ("4".to_string(), "A".to_string()),
    ]);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_batching_releases_unfinished_batches_on_repeated_unload() -> Result<()> {
    let sut = Sut::unloaded("batch_strategy_lifecycle", &[]).await?;
    let model = sut.model("batch_strategy_lifecycle");
    let mut actual = Vec::new();
    for _ in 0..3 {
        sut.repository_model_load().await?;
        sut.repository_model_unload().await?;
        actual.push(sut.wait_for_strategy_cleanup(model).await?);
    }
    let expected = vec![true; 3];
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_batching_contains_init_error() -> Result<()> {
    let sut = Sut::unloaded("batch_strategy_init_error", &[]).await?;
    let model = sut.model("batch_strategy_init_error");
    let failed = sut.repository_model_load().await.is_err();
    // Recover the retained failed version before unloading it.
    sut.repository_model_load().await?;
    sut.repository_model_unload().await?;
    let cleaned = sut.wait_for_strategy_cleanup(model).await?;
    let expected = (true, true, true);
    let actual = (failed, cleaned, sut.server_live().await?);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_batching_contains_init_panic() -> Result<()> {
    let sut = Sut::unloaded("batch_strategy_init_panic", &[]).await?;
    let model = sut.model("batch_strategy_init_panic");
    let failed = sut.repository_model_load().await.is_err();
    // Recover the retained failed version before unloading it.
    sut.repository_model_load().await?;
    sut.repository_model_unload().await?;
    let cleaned = sut.wait_for_strategy_cleanup(model).await?;
    let expected = (true, true, true);
    let actual = (failed, cleaned, sut.server_live().await?);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_batching_contains_batch_panic() -> Result<()> {
    let sut = Sut::unloaded("batch_strategy_batch_panic", &[]).await?;
    let model = sut.model("batch_strategy_batch_panic");
    sut.repository_model_load().await?;
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        sut.infer_with_parameters(HashMap::from([(
            "echo_batch_keys".to_string(),
            Parameter::Bool(true),
        )])),
    )
    .await
    .context("strategy stalled past its queue timeout")?;
    let error = result
        .err()
        .context("a refusing strategy answered successfully")?;
    let message = format!("{error:#}");
    let answered = message.contains("timeout") || message.contains("expired");
    sut.repository_model_unload().await?;
    let cleaned = sut.wait_for_strategy_cleanup(model).await?;
    let expected = (true, true, true);
    let actual = (answered, cleaned, sut.server_live().await?);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_batching_contains_include_error() -> Result<()> {
    let sut = Sut::unloaded("batch_strategy_include_error", &[]).await?;
    let model = sut.model("batch_strategy_include_error");
    sut.repository_model_load().await?;
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        sut.infer_with_parameters(HashMap::from([(
            "echo_batch_keys".to_string(),
            Parameter::Bool(true),
        )])),
    )
    .await
    .context("strategy stalled past its queue timeout")?;
    let error = result
        .err()
        .context("a refusing strategy answered successfully")?;
    let message = format!("{error:#}");
    let answered = message.contains("timeout") || message.contains("expired");
    sut.repository_model_unload().await?;
    let cleaned = sut.wait_for_strategy_cleanup(model).await?;
    let expected = (true, true, true);
    let actual = (answered, cleaned, sut.server_live().await?);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_batching_contains_include_panic() -> Result<()> {
    let sut = Sut::unloaded("batch_strategy_include_panic", &[]).await?;
    let model = sut.model("batch_strategy_include_panic");
    sut.repository_model_load().await?;
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        sut.infer_with_parameters(HashMap::from([(
            "echo_batch_keys".to_string(),
            Parameter::Bool(true),
        )])),
    )
    .await
    .context("strategy stalled past its queue timeout")?;
    let error = result
        .err()
        .context("a refusing strategy answered successfully")?;
    let message = format!("{error:#}");
    let answered = message.contains("timeout") || message.contains("expired");
    sut.repository_model_unload().await?;
    let cleaned = sut.wait_for_strategy_cleanup(model).await?;
    let expected = (true, true, true);
    let actual = (answered, cleaned, sut.server_live().await?);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_batching_contains_refuse() -> Result<()> {
    let sut = Sut::unloaded("batch_strategy_refuse", &[]).await?;
    let model = sut.model("batch_strategy_refuse");
    sut.repository_model_load().await?;
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        sut.infer_with_parameters(HashMap::from([(
            "echo_batch_keys".to_string(),
            Parameter::Bool(true),
        )])),
    )
    .await
    .context("strategy stalled past its queue timeout")?;
    let error = result
        .err()
        .context("a refusing strategy answered successfully")?;
    let message = format!("{error:#}");
    let answered = message.contains("timeout") || message.contains("expired");
    sut.repository_model_unload().await?;
    let cleaned = sut.wait_for_strategy_cleanup(model).await?;
    let expected = (true, true, true);
    let actual = (answered, cleaned, sut.server_live().await?);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_batching_contains_drop_panic() -> Result<()> {
    let sut = Sut::unloaded("batch_strategy_drop_panic", &[]).await?;
    let model = sut.model("batch_strategy_drop_panic");
    sut.repository_model_load().await?;
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        sut.infer_with_parameters(HashMap::from([(
            "echo_batch_keys".to_string(),
            Parameter::Bool(true),
        )])),
    )
    .await
    .context("strategy stalled past its queue timeout")?;
    let answered = result.is_ok();
    sut.repository_model_unload().await?;
    let cleaned = sut.wait_for_strategy_cleanup(model).await?;
    let expected = (true, true, true);
    let actual = (answered, cleaned, sut.server_live().await?);
    assert_eq!(actual, expected);
    Ok(())
}
