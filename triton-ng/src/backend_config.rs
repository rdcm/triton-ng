use crate::error::Error;
use crate::utils::parse_parameter;
use std::collections::BTreeMap;
use std::str::FromStr;

/// The backend config: the `key=value` pairs Triton was started with as
/// `--backend-config=<backend>,<key>=<value>`, read with
/// [`crate::Backend::config`].
///
/// Triton adds its own keys for every backend (`backend-directory`,
/// `default-max-batch-size` and others) next to the ones given on the
/// command line. Values are strings, parse them with
/// [`BackendConfig::parameter_or`].
///
/// A plain copy: owns its strings and holds nothing of Triton. `Send + Sync`.
///
/// # Examples
///
/// ```ignore
/// // tritonserver --backend-config=asr,pool-size=8
/// fn initialize(backend: &Backend) -> Result<(), Error> {
///     let config = backend.config()?;
///     let pool_size: usize = config.parameter_or("pool-size", 4)?;
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BackendConfig {
    parameters: BTreeMap<String, String>,
}

impl BackendConfig {
    /// Parses the JSON of `TRITONBACKEND_BackendConfig`:
    /// `{"cmdline": {"<key>": "<value>", ...}}`, where `cmdline` is left out
    /// when there is nothing in it.
    pub(crate) fn from_json(json: &str) -> Result<Self, String> {
        if json.is_empty() {
            return Ok(Self::default());
        }
        let value = serde_json::from_str::<serde_json::Value>(json)
            .map_err(|e| format!("backend config is not JSON: {e}"))?;
        let object = value
            .as_object()
            .ok_or_else(|| format!("backend config is not a JSON object: {value}"))?;
        let cmdline = match object.get("cmdline") {
            None => return Ok(Self::default()),
            Some(cmdline) => cmdline
                .as_object()
                .ok_or_else(|| format!("backend config cmdline is not a JSON object: {cmdline}"))?,
        };

        let mut parameters = BTreeMap::new();
        for (key, value) in cmdline {
            let value = value.as_str().ok_or_else(|| {
                format!("backend config parameter '{key}' is not a string: {value}")
            })?;
            parameters.insert(key.clone(), value.to_string());
        }
        Ok(Self { parameters })
    }

    /// The value of the parameter `key`, `None` when it is not set.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Some(directory) = backend.config()?.parameter("cache-dir") {
    ///     log::info(&format!("caching in {directory}"));
    /// }
    /// ```
    pub fn parameter(&self, key: &str) -> Option<&str> {
        self.parameters.get(key).map(String::as_str)
    }

    /// The value of the parameter `key` parsed as `T`, `default` when it is
    /// not set. Fails when the value does not parse.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let pool_size: usize = backend.config()?.parameter_or("pool-size", 4)?;
    /// ```
    pub fn parameter_or<T>(&self, key: &str, default: T) -> Result<T, Error>
    where
        T: FromStr,
        T::Err: std::fmt::Display,
    {
        match self.parameter(key) {
            None => Ok(default),
            Some(raw) => Ok(parse_parameter("backend config", key, raw)?),
        }
    }

    /// The value of the parameter `key`. Fails when it is not set.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let cache = backend.config()?.required_parameter("cache-dir")?.to_string();
    /// ```
    pub fn required_parameter(&self, key: &str) -> Result<&str, Error> {
        self.parameter(key)
            .ok_or_else(|| format!("missing '{key}' backend config parameter").into())
    }

    /// Every parameter as `(key, value)`, sorted by key.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// for (key, value) in backend.config()?.parameters() {
    ///     log::info(&format!("{key}={value}"));
    /// }
    /// ```
    pub fn parameters(&self) -> impl Iterator<Item = (&str, &str)> {
        self.parameters
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::BackendConfig;

    #[test]
    fn cmdline_parameters_are_read() -> Result<(), String> {
        let config = BackendConfig::from_json(
            r#"{"cmdline":{"pool-size":"8","default-max-batch-size":"4"}}"#,
        )?;
        assert_eq!(config.parameter("pool-size"), Some("8"));
        assert_eq!(config.parameter("missing"), None);
        assert_eq!(
            config.parameters().collect::<Vec<_>>(),
            vec![("default-max-batch-size", "4"), ("pool-size", "8")]
        );
        Ok(())
    }

    #[test]
    fn config_without_cmdline_is_empty() -> Result<(), String> {
        assert_eq!(BackendConfig::from_json("")?, BackendConfig::default());
        assert_eq!(BackendConfig::from_json("{}")?, BackendConfig::default());
        Ok(())
    }

    #[test]
    fn values_are_parsed_or_defaulted() -> Result<(), String> {
        let config = BackendConfig::from_json(r#"{"cmdline":{"pool-size":"8","name":"x"}}"#)?;
        assert_eq!(config.parameter_or("pool-size", 4).ok(), Some(8));
        assert_eq!(config.parameter_or("missing", 4).ok(), Some(4));
        assert!(config.parameter_or("name", 4).is_err());
        assert!(config.required_parameter("missing").is_err());
        assert_eq!(config.required_parameter("name").ok(), Some("x"));
        Ok(())
    }

    #[test]
    fn malformed_config_is_rejected() {
        assert!(BackendConfig::from_json("not json").is_err());
        assert!(BackendConfig::from_json("[]").is_err());
        assert!(BackendConfig::from_json(r#"{"cmdline":[]}"#).is_err());
        assert!(BackendConfig::from_json(r#"{"cmdline":{"pool-size":8}}"#).is_err());
    }
}
