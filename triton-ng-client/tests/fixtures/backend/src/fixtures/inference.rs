use super::*;

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let state = ctx.state::<State>()?;
    let mut pending = Vec::new();
    for request in requests {
        let image = request.get_input(IMAGE)?.as_bytes()?;
        let mut inference = host_inference(ctx, state, image)?;
        inference.set_parent_trace(&request)?;
        pending.push((request, Instant::now(), ctx.server().infer_async(inference)));
    }
    for (request, started, inference) in pending {
        let result = inference.await?;
        state
            .metrics
            .bls_seconds
            .observe(started.elapsed().as_secs_f64())?;
        let digit = argmax(&result.output(LOGITS)?.as_fp32_vec()?)?;
        respond(ctx, state, request, digit)?;
    }
    Ok(())
}
