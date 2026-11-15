use triton_ng::{Context, CustomBackend, DataType, Error, InferenceRequest, Request, Response};

struct MyBackend;

#[triton_ng::backend]
impl CustomBackend for MyBackend {
    async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
        for request in requests {
            let data = request.get_input("INPUT")?.as_bytes()?;

            let mut inference = InferenceRequest::new(ctx.server(), "minimal_backend_target", -1)?;
            inference.add_input_with_data("INPUT0", DataType::Fp32, &[1, 10], data)?;
            inference.add_requested_output("OUTPUT0")?;
            let result = match ctx.server().infer_async(inference).await {
                Ok(result) => result,
                Err(e) => {
                    request.respond_error_with_code(e.code(), &e.to_string())?;
                    continue;
                }
            };

            let mut response = Response::new(request)?;
            response
                .create_output("OUTPUT", DataType::Fp32, &[1, 10])?
                .write_fp32_vec(&result.output("OUTPUT0")?.as_fp32_vec()?)?;
            response.send()?;
        }
        Ok(())
    }
}
