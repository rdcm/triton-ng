use super::*;

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let mut remaining = Vec::new();
    for mut request in requests {
        if let Some(Parameter::String(code)) = request.parameter("error_code")? {
            if request.parameter("enable_statistics")? == Some(Parameter::Bool(true)) {
                request.enable_statistics();
            }
            match code.parse::<ErrorCode>() {
                Ok(code) => {
                    request.respond_error_with_code(code, "error requested by the client")?
                }
                Err(e) => request.respond_error(&e)?,
            }
            continue;
        }
        if let Some(Parameter::String(model)) = request.parameter("infer_model")? {
            let image = request.get_input(IMAGE)?.as_bytes()?;
            match infer_image(ctx, &model, image).await {
                Ok(_) => request.respond_error("the inference succeeded")?,
                Err(e) => request.respond_error_with_code(e.code(), &e.to_string())?,
            }
            continue;
        }
        remaining.push(request);
    }
    inference::execute(ctx, remaining).await?;
    Ok(())
}
