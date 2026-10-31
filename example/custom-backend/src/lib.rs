use std::sync::OnceLock;
use tokio::runtime::Runtime;
use triton_ng::backend::Backend;
use triton_ng::{BackendHandle, DataType, Error, InferenceRequest, Response};

struct MnistBackend;

fn runtime() -> Result<&'static Runtime, Error> {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    if let Some(runtime) = RUNTIME.get() {
        return Ok(runtime);
    }
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    Ok(RUNTIME.get_or_init(|| runtime))
}

impl Backend for MnistBackend {
    fn initialize(backend: &BackendHandle) -> Result<(), Error> {
        println!("Initializing backend: {}", backend.name()?);
        Ok(())
    }

    fn model_instance_execute(
        model: triton_ng::Model,
        requests: Vec<triton_ng::Request>,
    ) -> Result<(), Error> {
        let server = model.get_server()?;
        let version = model.version()? as i64;
        let target_model = model
            .config_parameter("target_model")?
            .ok_or("missing 'target_model' config parameter")?;

        let mut pending = Vec::with_capacity(requests.len());
        for request in requests {
            let input_names = request.input_names()?;
            let output_names = request.output_names()?;

            let mut inference_req = InferenceRequest::new(&server, &target_model, version)?;

            for name in &input_names {
                let input = request.get_input(name)?;
                let props = input.properties()?;

                let input_bytes: Vec<u8> = input
                    .as_fp32_vec()?
                    .iter()
                    .flat_map(|&f| f.to_le_bytes())
                    .collect();

                inference_req.add_input(name, DataType::Fp32, &props.shape)?;
                inference_req.append_input_data(name, &input_bytes)?;
            }

            for name in &output_names {
                inference_req.add_requested_output(name)?;
            }

            pending.push((request, output_names, server.infer_async(inference_req)));
        }

        runtime()?.block_on(async {
            for (request, output_names, inference) in pending {
                match inference.await {
                    Ok(result) => respond(request, &output_names, &result)?,
                    Err(e) => request.respond_error(&e.to_string())?,
                }
            }
            Ok(())
        })
    }
}

fn respond(
    request: triton_ng::Request,
    output_names: &[String],
    result: &triton_ng::InferenceResult,
) -> Result<(), Error> {
    let mut response = Response::new(request)?;
    for (output_tensor, name) in result.outputs.iter().zip(output_names) {
        let predictions: Vec<f32> = output_tensor
            .data
            .chunks_exact(4)
            .filter_map(|c| c.try_into().ok().map(f32::from_le_bytes))
            .collect();

        response
            .create_output(name, DataType::Fp32, &output_tensor.shape)?
            .write_fp32_vec(&predictions)?;
    }
    response.send()?;
    Ok(())
}

triton_ng::declare_backend!(MnistBackend);
