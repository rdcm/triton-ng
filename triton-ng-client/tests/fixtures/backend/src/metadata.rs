//! Metadata integration-test fixture.

use super::*;

pub(super) fn echo_metadata(request: Request) -> Result<(), Error> {
    let id = request.id()?;
    let correlation_id = request.correlation_id()?;
    let flags = request.flags()?;
    let timeout = request.timeout()?;
    let forwarded = request.parameter("forwarded")?;

    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[0])?;
    if let Some(id) = id {
        response.set_parameter("request_id", id)?;
    }
    match correlation_id {
        Some(CorrelationId::Int(id)) => response.set_parameter("correlation_id", id.to_string())?,
        Some(CorrelationId::String(id)) => response.set_parameter("correlation_id", id)?,
        None => {}
    }
    response.set_parameter("sequence_start", flags.sequence_start)?;
    response.set_parameter("sequence_end", flags.sequence_end)?;
    if let Some(timeout) = timeout {
        response.set_parameter("timeout_us", i64::try_from(timeout.as_micros())?)?;
    }
    if let Some(forwarded) = forwarded {
        response.set_parameter("forwarded", forwarded)?;
    }
    response.send()?;
    Ok(())
}

pub(super) fn metadata_inference(
    ctx: &Context,
    state: &State,
    request: &Request,
    image: Vec<u8>,
) -> Result<InferenceRequest, TritonError> {
    let mut inference = InferenceRequest::new(ctx.server(), &state.model, -1)?;
    inference.add_input_with_data(IMAGE, DataType::Fp32, &IMAGE_SHAPE, image)?;
    inference.add_requested_output("DIGIT")?;
    let id = request.id()?.unwrap_or_default();
    inference.set_id(&format!("{id}/bls"))?;
    inference.set_correlation_id(match request.correlation_id()? {
        Some(CorrelationId::Int(id)) => CorrelationId::Int(id + 1),
        Some(CorrelationId::String(id)) => CorrelationId::String(format!("{id}/bls")),
        None => CorrelationId::Int(1),
    })?;
    let flags = request.flags()?;
    inference.set_flags(RequestFlags {
        sequence_start: !flags.sequence_start,
        sequence_end: !flags.sequence_end,
    })?;
    inference.set_timeout(request.timeout()?.unwrap_or(Duration::from_secs(10)) * 2)?;
    inference.set_priority(1)?;
    inference.set_parameter("echo_metadata", true)?;
    inference.set_parameter("forwarded", "by the backend")?;
    inference.set_parent_trace(request)?;
    Ok(inference)
}

pub(super) fn answer_with_parameters_of(
    request: Request,
    result: &InferenceResult,
) -> Result<(), Error> {
    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&result.output("DIGIT")?.as_i32_vec()?)?;
    for name in [
        "request_id",
        "correlation_id",
        "sequence_start",
        "sequence_end",
        "timeout_us",
        "forwarded",
    ] {
        if let Some(value) = result.parameter(name)? {
            response.set_parameter(name, value)?;
        }
    }
    response.send()?;
    Ok(())
}

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let state = ctx.state::<State>()?;
    let mut remaining = Vec::new();
    for request in requests {
        if request.parameter("echo_metadata")? == Some(Parameter::Bool(true)) {
            echo_metadata(request)?;
            continue;
        }
        if request.parameter("forward_metadata")? == Some(Parameter::Bool(true)) {
            let image = request.get_input(IMAGE)?.as_bytes()?;
            let result = match metadata_inference(ctx, state, &request, image) {
                Ok(inference) => ctx.server().infer_async(inference).await,
                Err(e) => Err(e),
            };
            match result {
                Ok(result) => answer_with_parameters_of(request, &result)?,
                Err(e) => request.respond_error_with_code(e.code(), &e.to_string())?,
            }
            continue;
        }
        remaining.push(request);
    }
    inference::execute(ctx, remaining).await?;
    Ok(())
}
