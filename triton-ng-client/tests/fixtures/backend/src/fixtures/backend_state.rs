use super::*;

pub(super) fn echo_backend_state_count(ctx: &Context, request: Request) -> Result<(), Error> {
    let counted = ctx
        .backend_state::<BackendState>()?
        .requests
        .fetch_add(1, Ordering::SeqCst)
        + 1;
    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[0])?;
    response.set_parameter("backend_state_count", Parameter::Int(counted as i64))?;
    response.send()?;
    Ok(())
}

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let mut remaining = Vec::new();
    for request in requests {
        if request.parameter("count_backend_state")? == Some(Parameter::Bool(true)) {
            echo_backend_state_count(ctx, request)?;
            continue;
        }
        remaining.push(request);
    }
    inference::execute(ctx, remaining).await?;
    Ok(())
}
