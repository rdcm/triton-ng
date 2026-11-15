use anyhow::{Context, Result};

use super::dto::InferInput;

#[derive(Clone, serde::Deserialize)]
pub struct Config {
    #[serde(skip)]
    models: Option<std::sync::Arc<super::models::Models>>,
    pub triton_url: String,
    pub metrics_address: String,
    pub trace_dir: String,
    pub mnist_model: String,
    pub backend_model: String,
    pub stream_model: String,
    pub sequence_state_model: String,
    pub sequence_state_other_model: String,
    pub batch_strategy_model: String,
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

    /// Creates private model directories and instances for one test.
    pub async fn isolated(test: &str, backend: &str, load: &[&str]) -> Result<Self> {
        let mut config = Self::load()?;
        let models = super::models::Models::new(test, &config.triton_url)?;
        config.mnist_model = models.name(&config.mnist_model).to_string();
        config.backend_model = models.name(backend).to_string();
        config.stream_model = models.name(&config.stream_model).to_string();
        config.sequence_state_model = models.name(&config.sequence_state_model).to_string();
        config.sequence_state_other_model =
            models.name(&config.sequence_state_other_model).to_string();
        config.batch_strategy_model = models.name(&config.batch_strategy_model).to_string();
        let sut = super::sut::Sut::new(&config.triton_url).await?;
        let models = std::sync::Arc::new(models);
        config.models = Some(std::sync::Arc::clone(&models));
        for model in load {
            sut.repository_model_load(models.name(model)).await?;
        }
        Ok(config)
    }

    /// The private name of a model in this test's repository.
    pub fn model<'a>(&'a self, name: &'a str) -> &'a str {
        self.models
            .as_ref()
            .map_or(name, |models| models.name(name))
    }

    pub fn trace_dir(&self) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(&self.trace_dir)
    }

    pub fn mnist_blank_input(&self) -> InferInput {
        self.mnist_blank_images(1)
    }

    pub fn mnist_blank_images(&self, count: usize) -> InferInput {
        InferInput {
            name: self.input_name.clone(),
            shape: vec![count as i64, 1, 28, 28],
            data: vec![0.0_f32; 784 * count],
        }
    }
}
