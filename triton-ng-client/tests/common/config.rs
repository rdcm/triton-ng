use anyhow::{Context, Result};

use super::dto::InferInput;

#[derive(serde::Deserialize)]
pub struct Config {
    pub triton_url: String,
    pub mnist_model: String,
    pub backend_model: String,
    pub input_name: String,
    pub output_name: String,
    pub backend_output_name: String,
}

impl Config {
    pub fn load() -> Result<Config> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("config.json");
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read test config from {}", path.display()))?;
        serde_json::from_str(&content).context("Failed to parse test config")
    }

    pub fn mnist_blank_input(&self) -> InferInput {
        InferInput {
            name: self.input_name.clone(),
            shape: vec![1, 1, 28, 28],
            data: vec![0.0_f32; 784],
        }
    }
}
