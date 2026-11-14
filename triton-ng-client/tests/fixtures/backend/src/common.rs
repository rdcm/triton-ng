//! Common integration-test fixture.

use super::*;

pub(super) fn answer_done(request: Request) -> Result<(), Error> {
    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[0])?;
    response.send()?;
    Ok(())
}

pub(super) async fn infer_image(
    ctx: &Context,
    model: &str,
    image: Vec<u8>,
) -> Result<InferenceResult, TritonError> {
    let mut inference = InferenceRequest::new(ctx.server(), model, -1)?;
    inference.add_input_with_data(IMAGE, DataType::Fp32, &IMAGE_SHAPE, image)?;
    ctx.server().infer_async(inference).await
}

pub(super) fn host_inference(
    ctx: &Context,
    state: &State,
    image: Vec<u8>,
) -> Result<InferenceRequest, Error> {
    let mut inference = InferenceRequest::new(ctx.server(), &state.target_model, -1)?;
    inference.add_input_with_data(IMAGE, DataType::Fp32, &IMAGE_SHAPE, image)?;
    inference.add_requested_output(LOGITS)?;
    Ok(inference)
}

pub(super) fn argmax(logits: &[f32]) -> Result<i32, Error> {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(digit, _)| digit as i32)
        .ok_or_else(|| "mnist returned no logits".into())
}

pub(super) fn respond(
    ctx: &Context,
    state: &State,
    request: Request,
    digit: i32,
) -> Result<(), Error> {
    let statistics = request.parameter("enable_statistics")? == Some(Parameter::Bool(true));
    let buffers = ctx.instance_state::<Buffers>()?;
    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[digit])?;
    response.set_parameter("held_requests", state.held.load(Ordering::SeqCst))?;
    response.set_parameter("cancelled_requests", state.cancelled.load(Ordering::SeqCst))?;
    response.set_parameter("kept_inputs", i64::try_from(kept_inputs(state)?.len())?)?;
    response.set_parameter(
        "ram_buffers_allocated",
        buffers.ram_allocated.load(Ordering::SeqCst),
    )?;
    response.set_parameter(
        "pinned_buffers_allocated",
        buffers.pinned_allocated.load(Ordering::SeqCst),
    )?;
    response.set_parameter(
        "gpu_buffers_allocated",
        buffers.gpu_allocated.load(Ordering::SeqCst),
    )?;
    state.metrics.answered.increment(1.0)?;
    if statistics {
        response.enable_statistics();
    }
    response.send()?;
    Ok(())
}
