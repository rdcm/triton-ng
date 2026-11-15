use super::*;

pub(super) async fn forward_to_identity(
    ctx: &Context,
    state: &State,
    requests: Vec<Request>,
) -> Result<(), Error> {
    for request in requests {
        let (input_on_gpu, shape, result) = {
            let input = request.get_input(IDENTITY_INPUT)?;
            let input_on_gpu = input.bytes().is_err();
            let shared = input.share()?;
            let shape = shared.shape().to_vec();
            let mut inference = InferenceRequest::new(ctx.server(), &state.identity_model, -1)?;
            inference.add_input_with_buffer(IDENTITY_INPUT, shared.datatype(), &shape, shared)?;
            inference.add_requested_output(IDENTITY_OUTPUT)?;
            (input_on_gpu, shape, ctx.server().infer_async(inference))
        };
        match result.await {
            Ok(result) => {
                let mut response = Response::new(request)?;
                response
                    .create_output(IDENTITY_OUTPUT, DataType::Fp32, &shape)?
                    .write_bytes(result.output(IDENTITY_OUTPUT)?.bytes()?)?;
                response.set_parameter("input_on_gpu", input_on_gpu)?;
                response.send()?;
            }
            Err(e) => request.respond_error_with_code(e.code(), &e.to_string())?,
        }
    }
    Ok(())
}
