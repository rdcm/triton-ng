#[path = "common/mod.rs"]
mod common;

use crate::common::sut::Sut;
use anyhow::Result;

#[tokio::test]
async fn custom_backend_answers_cancelled_request_as_cancelled() -> Result<()> {
    let sut = Sut::new("fixture_cancellation", &[]).await?;
    let (_, cancelled_before) = sut.backend_counters().await?;
    let (_, statistics_before) = sut.failed_and_cancelled_responses().await?;

    let held = tokio::spawn(sut.clone().hold_request_until_cancelled());

    sut.wait_for_backend_counters(|held, _| held == 1).await?;
    held.abort();
    sut.wait_for_backend_counters(|held, cancelled| held == 0 && cancelled == cancelled_before + 1)
        .await?;

    // The cancelled request is answered with an error the statistics record as a
    // cancellation, see `FinalResponse::send_error`.
    let (_, statistics_after) = sut.failed_and_cancelled_responses().await?;

    let expected = 1;
    let actual = statistics_after - statistics_before;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_model_unloads_only_after_its_held_request_is_answered() -> Result<()> {
    let sut = Sut::new("fixture_cancellation", &[]).await?;
    let held = tokio::spawn(sut.clone().hold_request_until_cancelled());
    sut.wait_for_backend_counters(|held, _| held == 1).await?;

    sut.repository_model_unload().await?;
    let unloading = sut
        .wait_for_model_state(sut.model("fixture_cancellation"), "UNLOADING")
        .await;
    held.abort();
    let unloaded = sut
        .wait_for_model_state(sut.model("fixture_cancellation"), "UNAVAILABLE")
        .await;
    sut.repository_model_load().await?;

    unloading?;
    unloaded?;
    let actual = sut.model_ready().await?;
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}
