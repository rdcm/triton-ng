use super::*;

pub(super) fn echo_backend_config(ctx: &Context, request: Request) -> Result<(), Error> {
    let state = ctx.backend_state::<BackendState>()?;
    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[0])?;
    for (key, value) in state.config.parameters() {
        response.set_parameter(key, value)?;
    }
    response.set_parameter("artifact_type", format!("{:?}", state.artifact_type))?;
    response.set_parameter("execution_policy", format!("{:?}", state.execution_policy))?;
    response.set_parameter(
        "artifact_directory",
        state.artifact_directory.to_string_lossy().as_ref(),
    )?;
    response.send()?;
    Ok(())
}

pub(crate) fn execution_policy(backend: &Backend) -> Result<ExecutionPolicy, Error> {
    match backend.config()?.parameter("execution-policy") {
        None | Some("blocking") => Ok(ExecutionPolicy::Blocking),
        Some("device-blocking") => Ok(ExecutionPolicy::DeviceBlocking),
        Some(other) => Err(format!("unknown execution-policy '{other}'").into()),
    }
}

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let mut remaining = Vec::new();
    for request in requests {
        if request.parameter("echo_backend_config")? == Some(Parameter::Bool(true)) {
            echo_backend_config(ctx, request)?;
            continue;
        }
        remaining.push(request);
    }
    inference::execute(ctx, remaining).await?;
    Ok(())
}
