use super::*;

pub(super) fn echo_instance_details(ctx: &Context, request: Request) -> Result<(), Error> {
    let mut details = ctx
        .state::<State>()?
        .instances
        .lock()
        .map_err(|_| "the instance list is poisoned")?
        .clone();
    details.sort();
    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[0])?;
    for (index, detail) in details.iter().enumerate() {
        response.set_parameter(&format!("instance_{index}"), detail.as_str())?;
    }
    response.send()?;
    Ok(())
}

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let mut remaining = Vec::new();
    for request in requests {
        if request.parameter("echo_instance_details")? == Some(Parameter::Bool(true)) {
            echo_instance_details(ctx, request)?;
            continue;
        }
        remaining.push(request);
    }
    inference::execute(ctx, remaining).await?;
    Ok(())
}
