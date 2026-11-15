use super::*;

pub(super) async fn digit_with_buffers(
    ctx: &Context,
    state: &State,
    image: Vec<u8>,
    memory: OutputMemory,
) -> Result<i32, Error> {
    let buffers = ctx.instance_state::<Buffers>()?;
    let mut inference = InferenceRequest::new(ctx.server(), &state.target_model, -1)?;
    if let OutputMemory::Ram = memory {
        let mut ram_image = ctx.memory().ram(image.len())?;
        ram_image.copy_from_slice(&image);
        inference.add_input_with_buffer(IMAGE, DataType::Fp32, &IMAGE_SHAPE, ram_image)?;
    } else {
        let mut pinned_image = ctx.memory().pinned(image.len())?;
        pinned_image.copy_from_slice(&image);
        inference.add_input_with_buffer(IMAGE, DataType::Fp32, &IMAGE_SHAPE, pinned_image)?;
    }

    match memory {
        OutputMemory::Ram => {
            inference
                .add_requested_output_into(LOGITS, buffers.ram(ctx.memory(), LOGITS_BYTES)?)?;
            let mut result = ctx.server().infer_async(inference).await?;
            let digit = argmax(&result.output(LOGITS)?.as_fp32_vec()?)?;
            buffers.put_ram(result.take_ram(LOGITS)?)?;
            Ok(digit)
        }
        OutputMemory::Pinned => {
            inference
                .add_requested_output_into(LOGITS, buffers.pinned(ctx.memory(), LOGITS_BYTES)?)?;
            let mut result = ctx.server().infer_async(inference).await?;
            let digit = argmax(&result.output(LOGITS)?.as_fp32_vec()?)?;
            buffers.put_pinned(result.take_pinned(LOGITS)?)?;
            Ok(digit)
        }
        OutputMemory::Gpu => {
            let logits = buffers.gpu(ctx.memory(), ctx.device_id(), LOGITS_BYTES)?;
            inference.add_requested_output_into(LOGITS, logits)?;
            let mut result = ctx.server().infer_async(inference).await?;
            buffers.put_gpu(result.take_gpu(LOGITS)?)?;

            let inference = host_inference(ctx, state, image)?;
            let host = ctx.server().infer_async(inference).await?;
            argmax(&host.output(LOGITS)?.as_fp32_vec()?)
        }
        OutputMemory::GpuInput => {
            inference.add_requested_output_into(
                LOGITS,
                ctx.memory().gpu(ctx.device_id(), LOGITS_BYTES)?,
            )?;
            let mut result = ctx.server().infer_async(inference).await?;
            let logits = Arc::new(result.take_gpu(LOGITS)?);
            let copies = [
                identity_of(ctx, state, &logits)?,
                identity_of(ctx, state, &logits)?,
            ];
            drop(logits);

            let inference = host_inference(ctx, state, image)?;
            let host = ctx.server().infer_async(inference).await?;
            let expected = host.output(LOGITS)?.bytes()?;
            for copy in copies {
                if copy.await?.output(IDENTITY_OUTPUT)?.bytes()? != expected {
                    return Err("the GPU buffer read as an input differs from the logits".into());
                }
            }
            argmax(&host.output(LOGITS)?.as_fp32_vec()?)
        }
        OutputMemory::GpuForward => {
            inference.add_requested_output_into(
                LOGITS,
                ctx.memory().gpu(ctx.device_id(), LOGITS_BYTES)?,
            )?;
            let mut result = ctx.server().infer_async(inference).await?;
            let logits = result.take_gpu(LOGITS)?;
            let model = state
                .forward_model
                .as_deref()
                .ok_or("missing 'forward_model' config parameter")?;
            let mut forward = InferenceRequest::new(ctx.server(), model, -1)?;
            forward.add_input_with_buffer(IDENTITY_INPUT, DataType::Fp32, &[1, 10], logits)?;
            forward.add_requested_output(IDENTITY_OUTPUT)?;
            let forwarded = ctx.server().infer_async(forward).await?;

            let inference = host_inference(ctx, state, image)?;
            let host = ctx.server().infer_async(inference).await?;
            if forwarded.parameter("input_on_gpu")? != Some(Parameter::Bool(true)) {
                return Err("the forwarding model did not get its input on the GPU".into());
            }
            if forwarded.output(IDENTITY_OUTPUT)?.bytes()? != host.output(LOGITS)?.bytes()? {
                return Err("the forwarded GPU input differs from the logits".into());
            }
            argmax(&host.output(LOGITS)?.as_fp32_vec()?)
        }
        OutputMemory::WrongSize => {
            inference.add_requested_output_into(LOGITS, ctx.memory().pinned(4)?)?;
            ctx.server().infer_async(inference).await?;
            Err("output written into a buffer of the wrong size".into())
        }
    }
}

pub(super) fn identity_of(
    ctx: &Context,
    state: &State,
    logits: &Arc<GpuBuffer>,
) -> Result<impl Future<Output = Result<InferenceResult, TritonError>> + use<>, Error> {
    let mut inference = InferenceRequest::new(ctx.server(), &state.identity_model, -1)?;
    inference.add_input_with_buffer(
        IDENTITY_INPUT,
        DataType::Fp32,
        &[1, 10],
        Arc::clone(logits),
    )?;
    inference.add_requested_output(IDENTITY_OUTPUT)?;
    Ok(ctx.server().infer_async(inference))
}

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let state = ctx.state::<State>()?;
    let mut buffered = Vec::new();
    let mut remaining = Vec::new();
    for request in requests {
        if let Some(Parameter::String(memory)) = request.parameter("memory")? {
            match memory.parse::<OutputMemory>() {
                Ok(memory) => buffered.push((request, memory)),
                Err(e) => request.respond_error(&e.to_string())?,
            }
            continue;
        }
        remaining.push(request);
    }
    inference::execute(ctx, remaining).await?;
    for (request, memory) in buffered {
        let image = request.get_input(IMAGE)?.as_bytes()?;
        match digit_with_buffers(ctx, state, image, memory).await {
            Ok(digit) => respond(ctx, state, request, digit)?,
            Err(error) => request.respond_error(&error.to_string())?,
        }
    }
    Ok(())
}
