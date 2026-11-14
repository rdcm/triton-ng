//! Reschedule refusal integration-test fixture.

use super::*;

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let mut remaining = Vec::new();
    for request in requests {
        if request.parameter("reschedule")? == Some(Parameter::Bool(true)) {
            if let Err(e) = request.reschedule() {
                println!("refused to reschedule: {e}");
            }
            continue;
        }
        remaining.push(request);
    }
    inference::execute(ctx, remaining).await?;
    Ok(())
}
