use triton_ng::backend::Backend;
use triton_ng::{InferenceRequest, Response, sys};

struct MnistBackend;

const MODEL_NAME: &str = "mnist_onnx";
const MODEL_VERSION: i64 = 1;

impl Backend for MnistBackend {
    fn initialize() -> Result<(), triton_ng::Error> {
        println!("[MNIST] initialize");
        Ok(())
    }

    fn model_instance_execute(
        model: triton_ng::Model,
        requests: &[triton_ng::Request],
    ) -> Result<(), triton_ng::Error> {
        println!("[MNIST] model_instance_execute");

        let server = model.get_server()?;

        println!("[MNIST] api version: {:?}", server.api_version()?);
        println!(
            "[MNIST] is_model_ready: {:?}",
            server.is_model_ready(MODEL_NAME, MODEL_VERSION)?
        );
        println!(
            "[MNIST] model metadata: {:?}",
            server.model_metadata(MODEL_NAME, MODEL_VERSION)?
        );

        for request in requests {
            let input = request.get_input("Input3")?;
            let properties = input.properties()?;
            let input_values = input.as_fp32_vec()?;

            println!("[MNIST] input values count: {}", input_values.len());

            let mut inference_req = InferenceRequest::new(&server, MODEL_NAME, MODEL_VERSION)?;

            inference_req.add_input(
                "Input3",
                sys::TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32,
                &properties.shape,
            )?;

            let input_bytes: Vec<u8> = input_values.iter().flat_map(|&f| f.to_le_bytes()).collect();

            inference_req.append_input_data("Input3", &input_bytes)?;
            inference_req.add_requested_output("Plus214_Output_0")?;

            println!("[MNIST] Running inference...");
            let inference_result = server.infer_async(inference_req)?;

            let output_tensor = &inference_result.outputs[0];
            println!("[MNIST] Got output: {} bytes", output_tensor.data.len());

            let predictions: Vec<f32> = output_tensor
                .data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();

            println!("[MNIST] Predictions: {:?}", predictions);

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
