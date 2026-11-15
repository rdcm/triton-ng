use super::*;

pub(super) async fn stream_digits(
    ctx: &Context,
    state: &State,
    requests: Vec<Request>,
) -> Result<(), Error> {
    for request in requests {
        let images = request.get_input(IMAGE)?.as_bytes()?;
        let inferences = images
            .chunks(IMAGE_BYTES)
            .map(|image| {
                let inference = host_inference(ctx, state, image.to_vec())?;
                Ok(ctx.server().infer_async(inference))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let hold = request.parameter("hold_until_cancelled")? == Some(Parameter::Bool(true));
        let unfinished = request.parameter("unfinished")? == Some(Parameter::Bool(true));
        let statistics = request.parameter("enable_statistics")? == Some(Parameter::Bool(true));

        let mut sender = request.into_sender()?;
        for inference in inferences {
            let digit = argmax(&inference.await?.output(LOGITS)?.as_fp32_vec()?)?;
            let mut response = sender.response()?;
            response
                .create_output("DIGIT", DataType::Int32, &[1, 1])?
                .write_i32_vec(&[digit])?;
            response.set_parameter("held_requests", state.held.load(Ordering::SeqCst))?;
            response.set_parameter("cancelled_requests", state.cancelled.load(Ordering::SeqCst))?;
            if statistics {
                response.enable_statistics();
            }
            response.send()?;
        }

        if hold {
            hold_until_cancelled(state, finish_when_cancelled(ctx, state, sender, &images)).await?;
        } else if unfinished {
            drop(sender);
        } else {
            sender.finish()?;
        }
    }
    Ok(())
}

pub(super) fn stream_inference(
    ctx: &Context,
    state: &State,
    request: &Request,
    image: Vec<u8>,
    images: i64,
    hold_until_cancelled: bool,
) -> Result<InferenceRequest, Error> {
    let model = state
        .stream_model
        .as_deref()
        .ok_or("missing 'stream_model' config parameter")?;
    let mut inference = InferenceRequest::new(ctx.server(), model, -1)?;
    let shape = [images, 1, 28, 28];
    inference.add_input_with_data(
        IMAGE,
        DataType::Fp32,
        &shape,
        image.repeat(images.try_into()?),
    )?;
    inference.add_requested_output("DIGIT")?;
    if hold_until_cancelled {
        inference.set_parameter("hold_until_cancelled", true)?;
    }
    inference.set_parent_trace(request)?;
    Ok(inference)
}

pub(super) async fn answer_with_streamed_digits(
    ctx: &Context,
    request: Request,
    inference: InferenceRequest,
) -> Result<(), Error> {
    let mut stream = ctx.server().infer_stream(inference);
    let mut digits = Vec::new();
    while let Some(result) = stream.next().await {
        match result {
            Ok(result) => digits.extend(result.output("DIGIT")?.as_i32_vec()?),
            Err(e) => return Ok(request.respond_error_with_code(e.code(), &e.to_string())?),
        }
    }

    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[digits.first().copied().unwrap_or_default()])?;
    response.set_parameter("streamed_digits", i64::try_from(digits.len())?)?;
    response.send()?;
    Ok(())
}

pub(super) async fn cancel_stream(
    ctx: &Context,
    request: Request,
    inference: InferenceRequest,
    how: &str,
) -> Result<(), Error> {
    let mut stream = ctx.server().infer_stream(inference);
    stream.next().await.ok_or("the stream is empty")??;
    let ended_with = match how {
        "cancel" => {
            stream.cancel()?;
            let mut ended_with = "final flag".to_string();
            while let Some(result) = stream.next().await {
                if let Err(e) = result {
                    ended_with = e.code().to_string();
                }
            }
            ended_with
        }
        "drop" => {
            drop(stream);
            "dropped".to_string()
        }
        other => return Ok(request.respond_error(&format!("unknown cancel_stream '{other}'"))?),
    };

    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[0])?;
    response.set_parameter("stream_ended_with", ended_with)?;
    response.send()?;
    Ok(())
}

pub(super) async fn finish_when_cancelled(
    ctx: &Context,
    state: &State,
    sender: ResponseSender,
    images: &[u8],
) -> Result<(), Error> {
    while !sender.is_cancelled()? {
        work_step(ctx, state, images).await?;
    }
    sender.respond_cancelled()?;
    Ok(())
}
