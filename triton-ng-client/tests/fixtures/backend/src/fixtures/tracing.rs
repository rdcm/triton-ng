use super::*;

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let state = ctx.state::<State>()?;
    let mut remaining = Vec::new();
    for request in requests {
        if request.parameter("answer_before_bls")? == Some(Parameter::Bool(true)) {
            let image = request.get_input(IMAGE)?.as_bytes()?;
            let mut inference = host_inference(ctx, state, image)?;
            inference.set_parent_trace(&request)?;
            let unawaited = ctx.server().infer_async(inference);
            respond(ctx, state, request, 0)?;
            drop(unawaited);
            continue;
        }
        if let Some(Parameter::String(activity)) = request.parameter("report_activity")? {
            request.trace()?.report_activity(&activity)?;
            answer_done(request)?;
            continue;
        }
        remaining.push(request);
    }
    inference::execute(ctx, remaining).await?;
    Ok(())
}
