use app::app_config::AppConfig;
use app::error::{Error, Result};
use triton_ng_client::{InferInput, InferOptions, TritonClient, TritonClientConfig};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::init();

    let config = AppConfig::load()?;
    let model = config.model_name.as_str();
    let version = config.model_version.as_deref();

    let client = TritonClient::new(TritonClientConfig::new(&config.triton_url)).await?;

    let meta = client.model_metadata(model, version, None).await?;
    let input = meta.inputs.first().ok_or(Error::ModelHasNoInputs)?;
    let output = meta.outputs.first().ok_or(Error::ModelHasNoOutputs)?;
    let n_elements: usize = input.shape.iter().map(|&d| d as usize).product();

    let response = client
        .infer(
            model,
            version,
            [InferInput::fp32(
                &input.name,
                input.shape.clone(),
                vec![0.0f32; n_elements],
            )],
            [output.name.as_str()],
            InferOptions::default(),
        )
        .await?;

    for out in &response.outputs {
        println!("{}: shape={:?} values={:?}", out.name, out.shape, out.data);
    }

    Ok(())
}
