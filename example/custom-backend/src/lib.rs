use triton_ng::backend::Backend;
use triton_ng::{BackendHandle, Error, InferenceRequest, Response, sys};

struct MnistBackend;

const MODEL_NAME: &str = "mnist_onnx";
const MODEL_VERSION: i64 = 1;

impl Backend for MnistBackend {
    fn initialize(backend: &BackendHandle) -> Result<(), Error> {
        println!("Initializing model {}", backend.name()?);
        Ok(())
    }

    fn model_instance_execute(
        model: triton_ng::Model,
        requests: &[triton_ng::Request],
    ) -> Result<(), triton_ng::Error> {
        let server = model.get_server()?;

        for request in requests {
            let input = request.get_input("Input3")?;
            let properties = input.properties()?;

            // Must stay alive until infer_async returns — Triton holds a pointer, not a copy.
            let input_bytes: Vec<u8> = input
                .as_fp32_vec()?
                .iter()
                .flat_map(|&f| f.to_le_bytes())
                .collect();

            let mut inference_req = InferenceRequest::new(&server, MODEL_NAME, MODEL_VERSION)?;
            inference_req.add_input(
                "Input3",
                sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32,
                &properties.shape,
            )?;
            inference_req.append_input_data("Input3", &input_bytes)?;
            inference_req.add_requested_output("Plus214_Output_0")?;

            let result = server.infer_async(inference_req)?;
            let predictions: Vec<f32> = result.outputs[0]
                .data
                .chunks_exact(4)
                .filter_map(|c| c.try_into().ok().map(f32::from_le_bytes))
                .collect();

            let mut response = Response::new(request)?;
            response
                .create_output(
                    "Plus214_Output_0",
                    sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32,
                    &[10],
                )?
                .write_fp32_vec(&predictions)?;
            response.send()?;
        }

        Ok(())
    }
}

triton_ng::declare_backend!(MnistBackend);
