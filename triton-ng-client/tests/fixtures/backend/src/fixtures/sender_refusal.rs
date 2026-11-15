use super::*;

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let mut remaining = Vec::new();
    for request in requests {
        if request.parameter("into_sender")? == Some(Parameter::Bool(true)) {
            if request.into_sender().is_ok() {
                return Err("a request of a model that is not decoupled got a sender".into());
            }
            continue;
        }
        remaining.push(request);
    }
    inference::execute(ctx, remaining).await?;
    Ok(())
}
