//! Bls mutation integration-test fixture.

use super::*;

pub(super) fn invalid_argument(result: Result<(), TritonError>) -> Result<(), Error> {
    match result {
        Err(error) if error.code() == ErrorCode::InvalidArg => Ok(()),
        Err(error) => Err(error.into()),
        Ok(()) => Err("an invalid request mutation unexpectedly succeeded".into()),
    }
}

pub(super) async fn check_bls_mutation(
    ctx: &Context,
    state: &State,
    image: Vec<u8>,
) -> Result<(), Error> {
    let ordinary = Arc::new(image.clone());
    let policy = Arc::new(image.clone());
    let failed = Arc::new(image.clone());
    let mut inference = InferenceRequest::new(ctx.server(), &state.target_model, -1)?;
    inference.add_input_with_buffer(IMAGE, DataType::Fp32, &IMAGE_SHAPE, ordinary.clone())?;
    inference.append_input_buffer_with_host_policy(IMAGE, policy.clone(), "pinned")?;
    invalid_argument(inference.remove_input("missing"))?;
    invalid_argument(inference.remove_input("Input3\0bad"))?;
    invalid_argument(inference.remove_requested_output("bad\0name"))?;
    invalid_argument(inference.append_input_buffer_with_host_policy(
        IMAGE,
        policy.clone(),
        "bad\0policy",
    ))?;
    invalid_argument(inference.append_input_data_with_host_policy("bad\0input", &image, "pinned"))?;
    invalid_argument(inference.append_input_buffer_with_host_policy(
        "missing",
        failed.clone(),
        "pinned",
    ))?;
    inference.remove_input(IMAGE)?;
    invalid_argument(inference.remove_input(IMAGE))?;
    if Arc::strong_count(&ordinary) != 2
        || Arc::strong_count(&policy) != 2
        || Arc::strong_count(&failed) != 2
    {
        return Err("removed or failed input buffers were not retained".into());
    }
    // Reuse the name, then remove all inputs twice, including the empty case.
    inference.add_input_with_data(IMAGE, DataType::Fp32, &IMAGE_SHAPE, image.clone())?;
    inference.remove_all_inputs()?;
    inference.remove_all_inputs()?;
    drop(inference);
    if Arc::strong_count(&ordinary) != 1
        || Arc::strong_count(&policy) != 1
        || Arc::strong_count(&failed) != 1
    {
        return Err("dropping an unsubmitted request did not release retained inputs".into());
    }

    let mut inference = InferenceRequest::new(ctx.server(), &state.target_model, -1)?;
    inference.add_input_with_buffer(IMAGE, DataType::Fp32, &IMAGE_SHAPE, ordinary.clone())?;
    inference.remove_all_inputs()?;
    inference.add_input_with_data(IMAGE, DataType::Fp32, &IMAGE_SHAPE, image.clone())?;
    // These undersized buffers must stop affecting allocation after removal.
    inference.add_requested_output_into(LOGITS, ctx.memory().ram(1)?)?;
    inference.remove_requested_output(LOGITS)?;
    inference.remove_requested_output("missing")?; // Triton deliberately succeeds.
    inference.add_requested_output_into(LOGITS, ctx.memory().ram(1)?)?;
    inference.add_requested_output("unused")?;
    inference.remove_all_requested_outputs()?;
    inference.remove_all_requested_outputs()?;
    // Empty requested-output list means all outputs, not none.
    let result = ctx.server().infer_async(inference).await?;
    let expected = infer_image(ctx, &state.target_model, image).await?;
    if result.output(LOGITS)?.bytes()? != expected.output(LOGITS)?.bytes()? {
        return Err("the rebuilt request returned different logits".into());
    }
    drop(result);
    wait_for_input_release(&ordinary)?;
    Ok(())
}

pub(super) fn wait_for_input_release(buffer: &Arc<Vec<u8>>) -> Result<(), Error> {
    let started = Instant::now();
    while Arc::strong_count(buffer) != 1 {
        if started.elapsed() > Duration::from_secs(2) {
            return Err("BLS input release did not complete".into());
        }
        std::thread::yield_now();
    }
    Ok(())
}

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let state = ctx.state::<State>()?;
    let mut remaining = Vec::new();
    for request in requests {
        if request.parameter("check_bls_mutation")? == Some(Parameter::Bool(true)) {
            let image = request.get_input(IMAGE)?.as_bytes()?;
            let checked = check_bls_mutation(ctx, state, image).await;
            match checked {
                Ok(()) => answer_done(request)?,
                Err(error) => request.respond_error(&error.to_string())?,
            }
            continue;
        }
        remaining.push(request);
    }
    inference::execute(ctx, remaining).await?;
    Ok(())
}
