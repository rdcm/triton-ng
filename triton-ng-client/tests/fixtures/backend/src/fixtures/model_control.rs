use super::*;

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let state = ctx.state::<State>()?;
    let mut remaining = Vec::new();
    for request in requests {
        if let Some(Parameter::String(model)) = request.parameter("load_model")? {
            let file = match request.parameter("load_file")? {
                Some(Parameter::String(file)) => file,
                _ => "file:1/model.onnx".to_string(),
            };
            let onnx = state
                .repository
                .join(&state.target_model)
                .join("1/model.onnx");
            let from_repository = request.parameter("load_from")?
                == Some(Parameter::String("repository".to_string()));
            let server = ctx.server();
            let loaded = if from_repository {
                server.load_model(&model, None, &[])
            } else {
                std::fs::read(onnx)
                    .map_err(|e| TritonError::from_message(e.to_string()))
                    .and_then(|onnx| {
                        server.load_model(
                            &model,
                            Some(r#"{"backend": "onnxruntime"}"#),
                            &[(&file, &onnx)],
                        )
                    })
            };
            match loaded {
                Ok(()) => answer_done(request)?,
                Err(e) => request.respond_error_with_code(e.code(), &e.to_string())?,
            }
            continue;
        }
        if let Some(Parameter::String(model)) = request.parameter("unload_model")? {
            match ctx.server().unload_model(&model, false) {
                Ok(()) => answer_done(request)?,
                Err(e) => request.respond_error_with_code(e.code(), &e.to_string())?,
            }
            continue;
        }
        remaining.push(request);
    }
    inference::execute(ctx, remaining).await?;
    Ok(())
}
