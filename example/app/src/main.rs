use anyhow::{Context, Result, bail};
use triton_client::{InferenceOutput, TritonClient};
use triton_grpc_client::inference::model_infer_request::{
    InferInputTensor, InferRequestedOutputTensor,
};
use triton_grpc_client::inference::{InferTensorContents, ModelInferRequest};

const MODEL_NAME: &str = "mnist";
const MODEL_VERSION: &str = "1";

#[tokio::main]
async fn main() -> Result<()> {
    let mut client = TritonClient::new("http://localhost:8001").await?;

    println!("Server ready: {}", client.server_ready().await?);
    println!(
        "Model ready: {}",
        client.model_ready(MODEL_NAME, MODEL_VERSION).await?
    );

    let meta = client.model_metadata(MODEL_NAME, MODEL_VERSION).await?;

    let input_meta = meta.inputs.first().context("model has no inputs")?;
    let output_meta = meta.outputs.first().context("model has no outputs")?;

    let shape: Vec<i64> = input_meta.shape.clone();
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();

    if input_meta.datatype != "FP32" {
        bail!("expected FP32 input, got {}", input_meta.datatype);
    }

    let response = client
        .infer(ModelInferRequest {
            model_name: MODEL_NAME.to_string(),
            model_version: MODEL_VERSION.to_string(),
            id: String::new(),
            inputs: vec![InferInputTensor {
                name: input_meta.name.clone(),
                datatype: input_meta.datatype.clone(),
                shape,
                parameters: Default::default(),
                contents: Some(InferTensorContents {
                    fp32_contents: vec![0.123f32; n_elements],
                    ..Default::default()
                }),
            }],
            outputs: vec![InferRequestedOutputTensor {
                name: output_meta.name.clone(),
                parameters: Default::default(),
            }],
            parameters: Default::default(),
            raw_input_contents: vec![],
        })
        .await?;

    let outputs = InferenceOutput::from_response(&response)?;
    println!("Output: {:?}", outputs);

    Ok(())
}
