//! Bls streaming integration-test fixture.

use super::*;

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let state = ctx.state::<State>()?;
    let mut remaining = Vec::new();
    for request in requests {
        if let Some(Parameter::Int(images)) = request.parameter("stream_images")? {
            let via_infer_async =
                request.parameter("via_infer_async")? == Some(Parameter::Bool(true));
            let image = request.get_input(IMAGE)?.as_bytes()?;
            let inference = stream_inference(ctx, state, &request, image, images, false)?;
            if via_infer_async {
                match ctx.server().infer_async(inference).await {
                    Ok(_) => request.respond_error("the inference succeeded")?,
                    Err(e) => request.respond_error_with_code(e.code(), &e.to_string())?,
                }
            } else {
                answer_with_streamed_digits(ctx, request, inference).await?;
            }
            continue;
        }
        if let Some(Parameter::String(how)) = request.parameter("cancel_stream")? {
            let image = request.get_input(IMAGE)?.as_bytes()?;
            let inference = stream_inference(ctx, state, &request, image, 1, true)?;
            cancel_stream(ctx, request, inference, &how).await?;
            continue;
        }
        remaining.push(request);
    }
    inference::execute(ctx, remaining).await?;
    Ok(())
}
