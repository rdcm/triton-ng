use crate::error::Result;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct AppConfig {
    pub triton_url: String,
    pub model_name: String,
    pub model_version: Option<String>,
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        let cfg = config::Config::builder()
            .add_source(config::File::with_name("example/app/src/app_config"))
            .add_source(config::Environment::default())
            .build()?;
        Ok(cfg.try_deserialize()?)
    }
}
