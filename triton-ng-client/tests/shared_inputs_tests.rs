#[path = "common/mod.rs"]
mod common;

use crate::common::sut::Sut;
use anyhow::Result;
use triton_ng_client::Parameter;

#[tokio::test]
async fn custom_backend_passes_an_input_of_its_request_on_without_a_copy() -> Result<()> {
    let sut = Sut::new("fixture_shared_inputs", &[]).await?;
    let expected = sut.infer_digit().await?;

    let digit = sut.infer_digit_with_shared_input(false).await?;

    let actual = digit;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn a_shared_input_holds_back_its_request_response() -> Result<()> {
    let sut = Sut::new("fixture_shared_inputs", &[]).await?;

    let held = tokio::spawn(sut.clone().infer_digit_keeping_shared_input());
    sut.wait_for_kept_inputs().await?;
    let actual = !held.is_finished();
    sut.infer_async(&[("release_shared_inputs", Parameter::Bool(true))])
        .await?;
    held.await??;
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn dropping_a_shared_input_allows_its_request_response() -> Result<()> {
    let sut = Sut::new("fixture_shared_inputs", &[]).await?;
    let expected = sut.infer_digit().await?;

    let held = tokio::spawn(sut.clone().infer_digit_keeping_shared_input());
    sut.wait_for_kept_inputs().await?;
    sut.infer_async(&[("release_shared_inputs", Parameter::Bool(true))])
        .await?;

    let actual = held.await??;
    assert_eq!(actual, expected);
    Ok(())
}
