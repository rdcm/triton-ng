//! Cancellation integration-test fixture.

use super::*;

pub(super) async fn hold_until_cancelled(
    state: &State,
    answer: impl Future<Output = Result<(), Error>>,
) -> Result<(), Error> {
    state.held.fetch_add(1, Ordering::SeqCst);
    state.metrics.held.increment(1.0)?;
    let result = answer.await;
    if result.is_ok() {
        state.cancelled.fetch_add(1, Ordering::SeqCst);
    }
    state.held.fetch_sub(1, Ordering::SeqCst);
    state.metrics.held.decrement(1.0)?;
    result
}

pub(super) async fn work_step(ctx: &Context, state: &State, images: &[u8]) -> Result<(), Error> {
    let image = images
        .get(..IMAGE_BYTES)
        .ok_or("the request has no image")?;
    let inference = host_inference(ctx, state, image.to_vec())?;
    ctx.server().infer_async(inference).await?;
    Ok(())
}

pub(super) async fn respond_when_cancelled(
    ctx: &Context,
    state: &State,
    mut request: Request,
) -> Result<(), Error> {
    let images = request.get_input(IMAGE)?.as_bytes()?;
    while !request.is_cancelled()? {
        work_step(ctx, state, &images).await?;
    }
    if request.parameter("enable_statistics")? == Some(Parameter::Bool(true)) {
        request.enable_statistics();
    }
    request.respond_cancelled()?;
    Ok(())
}

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let state = ctx.state::<State>()?;
    let mut held = Vec::new();
    let mut remaining = Vec::new();
    for request in requests {
        if request.parameter("hold_until_cancelled")? == Some(Parameter::Bool(true)) {
            held.push(request);
            continue;
        }
        remaining.push(request);
    }
    inference::execute(ctx, remaining).await?;
    for request in held {
        hold_until_cancelled(state, respond_when_cancelled(ctx, state, request)).await?;
    }
    Ok(())
}
