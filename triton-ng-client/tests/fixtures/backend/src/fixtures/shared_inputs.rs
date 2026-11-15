use super::*;

pub(super) async fn digit_of_shared_input(
    ctx: &Context,
    state: &State,
    image: Arc<SharedInput>,
    copy: Vec<u8>,
    keep: bool,
) -> Result<i32, Error> {
    let mut inference = InferenceRequest::new(ctx.server(), &state.target_model, -1)?;
    let shape = image.shape().to_vec();
    inference.add_input_with_buffer(IMAGE, image.datatype(), &shape, Arc::clone(&image))?;
    inference.add_requested_output(LOGITS)?;
    let shared = ctx.server().infer_async(inference).await?;

    let copied = host_inference(ctx, state, copy)?;
    let copied = ctx.server().infer_async(copied).await?;
    if shared.output(LOGITS)?.bytes()? != copied.output(LOGITS)?.bytes()? {
        return Err("the shared input differs from its copy".into());
    }

    if keep {
        kept_inputs(state)?.push(image);
    }
    argmax(&shared.output(LOGITS)?.as_fp32_vec()?)
}

pub(super) fn kept_inputs(state: &State) -> Result<MutexGuard<'_, Vec<Arc<SharedInput>>>, Error> {
    state
        .kept_inputs
        .lock()
        .map_err(|_| "the kept inputs are poisoned".into())
}

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let state = ctx.state::<State>()?;
    let mut remaining = Vec::new();
    for request in requests {
        if let Some(Parameter::Bool(keep)) = request.parameter("shared_input")? {
            let (image, copy) = {
                let input = request.get_input(IMAGE)?;
                (Arc::new(input.share()?), input.as_bytes()?)
            };
            match digit_of_shared_input(ctx, state, image, copy, keep).await {
                Ok(digit) => respond(ctx, state, request, digit)?,
                Err(e) => request.respond_error(&e.to_string())?,
            }
            continue;
        }
        if request.parameter("release_shared_inputs")? == Some(Parameter::Bool(true)) {
            kept_inputs(state)?.clear();
        }
        remaining.push(request);
    }
    inference::execute(ctx, remaining).await?;
    Ok(())
}
