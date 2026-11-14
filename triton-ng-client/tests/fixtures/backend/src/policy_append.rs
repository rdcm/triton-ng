//! Policy append integration-test fixture.

use super::*;

pub(super) async fn check_bls_policy_append(ctx: &Context, image: Vec<u8>) -> Result<(), Error> {
    let mut inference = InferenceRequest::new(ctx.server(), "host_policy_pinned", -1)?;
    inference.add_input_with_data(IMAGE, DataType::Fp32, &IMAGE_SHAPE, image)?;
    let mut first = 1.25_f32.to_ne_bytes().repeat(392);
    inference.append_input_data_with_host_policy(IMAGE, &first, "pinned")?;
    first.fill(0); // The copied append must not retain the caller's slice.
    let second = Arc::new(1.25_f32.to_ne_bytes().repeat(392));
    inference.append_input_buffer_with_host_policy(IMAGE, second.clone(), "pinned")?;
    inference.set_parameter("check_specific_policy", Parameter::Bool(true))?;
    let result = ctx.server().infer_async(inference).await?;
    drop(result);
    wait_for_input_release(&second)?;
    Ok(())
}

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let mut remaining = Vec::new();
    for request in requests {
        if request.parameter("check_bls_policy_append")? == Some(Parameter::Bool(true)) {
            let image = request.get_input(IMAGE)?.as_bytes()?;
            let checked = check_bls_policy_append(ctx, image).await;
            match checked {
                Ok(()) => answer_done(request)?,
                Err(error) => request.respond_error(&error.to_string())?,
            }
            continue;
        }
        remaining.push(request);
    }
    inference::execute(ctx, remaining).await?;
    Ok(())
}
