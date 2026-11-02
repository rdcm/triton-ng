use crate::error::Error;
use crate::types::DataType;
use serde_json::{Map, Value};

/// The config of a model being loaded, given to
/// [`crate::CustomBackend::auto_complete_config`] to fill in what its
/// `config.pbtxt` leaves out, like `pb_utils.ModelConfig` of python_backend.
///
/// Triton takes back only the max batch size, the inputs and outputs, the
/// scheduler when the config names none, and `decoupled`, so these are the
/// only parts that can be changed. A part the config already sets can only
/// be confirmed, not changed: a conflicting value fails, like in
/// python_backend.
///
/// A plain copy of the config: holds nothing of Triton. `Send + Sync`.
///
/// # Examples
///
/// ```ignore
/// fn auto_complete_config(_model: &Model, config: &mut AutoCompleteConfig) -> Result<(), Error> {
///     config.set_max_batch_size(16)?;
///     config.set_dynamic_batching()?;
///     config.add_input("AUDIO", DataType::Fp32, &[-1])?;
///     config.add_output("TEXT", DataType::Bytes, &[1])?;
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct AutoCompleteConfig {
    model: String,
    config: Map<String, Value>,
}

impl AutoCompleteConfig {
    /// Reads the JSON of `TRITONBACKEND_ModelConfig`.
    pub(crate) fn from_json(json: &str) -> Result<Self, String> {
        let value = serde_json::from_str::<Value>(json)
            .map_err(|e| format!("the model config is not JSON: {e}"))?;
        let Value::Object(config) = value else {
            return Err(format!("the model config is not a JSON object: {value}"));
        };
        let model = match config.get("name") {
            None => String::new(),
            Some(Value::String(name)) => name.clone(),
            Some(other) => return Err(format!("the model name is not a string: {other}")),
        };
        Ok(Self { model, config })
    }

    /// The JSON for `TRITONBACKEND_ModelSetConfig`.
    pub(crate) fn to_json(&self) -> String {
        Value::Object(self.config.clone()).to_string()
    }

    /// The max batch size of the config, `0` when it does not batch.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if config.max_batch_size()? == 0 {
    ///     config.set_max_batch_size(8)?;
    /// }
    /// ```
    pub fn max_batch_size(&self) -> Result<u32, Error> {
        match self.config.get("max_batch_size") {
            None => Ok(0),
            Some(value) => Ok(u32::try_from(integer(value)?)
                .map_err(|_| format!("model '{}' has max_batch_size {value}", self.model))?),
        }
    }

    /// Whether the config has the input `name`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if !config.has_input("AUDIO") {
    ///     config.add_input("AUDIO", DataType::Fp32, &[-1])?;
    /// }
    /// ```
    pub fn has_input(&self, name: &str) -> bool {
        self.has_tensor("input", name)
    }

    /// Whether the config has the output `name`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if !config.has_output("TEXT") {
    ///     config.add_output("TEXT", DataType::Bytes, &[1])?;
    /// }
    /// ```
    pub fn has_output(&self, name: &str) -> bool {
        self.has_tensor("output", name)
    }

    fn has_tensor(&self, kind: &str, name: &str) -> bool {
        self.config
            .get(kind)
            .and_then(Value::as_array)
            .is_some_and(|tensors| {
                tensors
                    .iter()
                    .any(|tensor| tensor.get("name").and_then(Value::as_str) == Some(name))
            })
    }

    /// Sets the max batch size. Fails when the config sets a larger one, or
    /// `max_batch_size` does not fit the `int32` of the model config.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// config.set_max_batch_size(16)?;
    /// ```
    pub fn set_max_batch_size(&mut self, max_batch_size: u32) -> Result<(), Error> {
        if i32::try_from(max_batch_size).is_err() {
            return Err(format!("max_batch_size {max_batch_size} does not fit int32").into());
        }
        let configured = self.max_batch_size()?;
        if configured > max_batch_size {
            return Err(format!(
                "the config of model '{}' sets max_batch_size {configured}, larger than {max_batch_size}",
                self.model
            )
            .into());
        }
        self.config
            .insert("max_batch_size".to_string(), Value::from(max_batch_size));
        Ok(())
    }

    /// Makes the model use the dynamic batcher. Does nothing when the config
    /// already does; fails when it names another scheduler.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// config.set_dynamic_batching()?;
    /// ```
    pub fn set_dynamic_batching(&mut self) -> Result<(), Error> {
        for other in ["sequence_batching", "ensemble_scheduling"] {
            if self.config.contains_key(other) {
                return Err(format!(
                    "the config of model '{}' sets {other}, dynamic batching cannot be added",
                    self.model
                )
                .into());
            }
        }
        self.config
            .entry("dynamic_batching")
            .or_insert_with(|| Value::Object(Map::new()));
        Ok(())
    }

    /// Adds the input `name`. When the config has it, fills in its data type
    /// and dims if it leaves them out, and fails when they differ.
    ///
    /// Fails when `name` is empty, `data_type` is INVALID or a dim is neither
    /// positive nor `-1` (any size).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// config.add_input("AUDIO", DataType::Fp32, &[-1])?;
    /// ```
    pub fn add_input(
        &mut self,
        name: &str,
        data_type: DataType,
        dims: &[i64],
    ) -> Result<(), Error> {
        self.add_tensor("input", name, data_type, dims, None)
    }

    /// Adds the input `name` that requests may leave out, like
    /// [`AutoCompleteConfig::add_input`]. Fails also when the config has the
    /// input and says it is required.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// config.add_optional_input("LANGUAGE", DataType::Bytes, &[1])?;
    /// ```
    pub fn add_optional_input(
        &mut self,
        name: &str,
        data_type: DataType,
        dims: &[i64],
    ) -> Result<(), Error> {
        self.add_tensor("input", name, data_type, dims, Some(true))
    }

    /// Adds the output `name`, like [`AutoCompleteConfig::add_input`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// config.add_output("TEXT", DataType::Bytes, &[1])?;
    /// ```
    pub fn add_output(
        &mut self,
        name: &str,
        data_type: DataType,
        dims: &[i64],
    ) -> Result<(), Error> {
        self.add_tensor("output", name, data_type, dims, None)
    }

    /// Sets whether the model is decoupled, see [`crate::Request::into_sender`].
    /// Fails when the config sets it the other way.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// config.set_decoupled(true)?;
    /// ```
    pub fn set_decoupled(&mut self, decoupled: bool) -> Result<(), Error> {
        let policy = self
            .config
            .entry("model_transaction_policy")
            .or_insert_with(|| Value::Object(Map::new()));
        let Value::Object(policy) = policy else {
            return Err(format!(
                "model_transaction_policy of model '{}' is not an object",
                self.model
            )
            .into());
        };
        match policy.get("decoupled") {
            None => {
                policy.insert("decoupled".to_string(), Value::Bool(decoupled));
                Ok(())
            }
            Some(Value::Bool(configured)) if *configured == decoupled => Ok(()),
            Some(configured) => Err(format!(
                "the config of model '{}' sets decoupled to {configured}",
                self.model
            )
            .into()),
        }
    }

    fn add_tensor(
        &mut self,
        kind: &str,
        name: &str,
        data_type: DataType,
        dims: &[i64],
        optional: Option<bool>,
    ) -> Result<(), Error> {
        if name.is_empty() {
            return Err(format!("an {kind} of model '{}' has no name", self.model).into());
        }
        let type_name = data_type
            .config_name()
            .ok_or_else(|| format!("{kind} '{name}' has the INVALID data type"))?;
        if let Some(dim) = dims.iter().find(|&&dim| dim < -1 || dim == 0) {
            return Err(format!("{kind} '{name}' has dim {dim}, not positive or -1").into());
        }

        let model = self.model.clone();
        let tensors = self
            .config
            .entry(kind)
            .or_insert_with(|| Value::Array(Vec::new()));
        let Value::Array(tensors) = tensors else {
            return Err(format!("{kind} of model '{model}' is not a list").into());
        };

        for tensor in tensors.iter_mut() {
            let Value::Object(tensor) = tensor else {
                return Err(format!("an {kind} of model '{model}' is not an object").into());
            };
            if tensor.get("name").and_then(Value::as_str) != Some(name) {
                continue;
            }

            match tensor.get("data_type").and_then(Value::as_str) {
                None | Some("TYPE_INVALID") => {}
                Some(configured) if configured == type_name => {}
                Some(configured) => {
                    return Err(format!(
                        "the config of model '{model}' gives {kind} '{name}' the type \
                         {configured}, the backend {type_name}"
                    )
                    .into());
                }
            }
            let configured_dims = match tensor.get("dims") {
                None => Vec::new(),
                Some(Value::Array(configured)) => configured.iter().map(integer).collect::<Result<
                    Vec<_>,
                    _,
                >>(
                )?,
                Some(other) => {
                    return Err(format!("dims of {kind} '{name}' are not a list: {other}").into());
                }
            };
            if !configured_dims.is_empty() && configured_dims != dims {
                return Err(format!(
                    "the config of model '{model}' gives {kind} '{name}' the dims \
                     {configured_dims:?}, the backend {dims:?}"
                )
                .into());
            }
            if let (Some(configured), Some(optional)) =
                (tensor.get("optional").and_then(Value::as_bool), optional)
                && configured != optional
            {
                return Err(format!(
                    "the config of model '{model}' makes {kind} '{name}' required"
                )
                .into());
            }

            tensor.insert("data_type".to_string(), Value::from(type_name));
            tensor.insert("dims".to_string(), Value::from(dims.to_vec()));
            if let Some(optional) = optional {
                tensor.insert("optional".to_string(), Value::Bool(optional));
            }
            return Ok(());
        }

        let mut tensor = Map::new();
        tensor.insert("name".to_string(), Value::from(name));
        tensor.insert("data_type".to_string(), Value::from(type_name));
        tensor.insert("dims".to_string(), Value::from(dims.to_vec()));
        if let Some(optional) = optional {
            tensor.insert("optional".to_string(), Value::Bool(optional));
        }
        tensors.push(Value::Object(tensor));
        Ok(())
    }
}

/// An integer of the config JSON: Triton writes 64-bit integers as numbers or,
/// like protobuf JSON, as strings.
fn integer(value: &Value) -> Result<i64, String> {
    match value {
        Value::Number(number) => number.as_i64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
    .ok_or_else(|| format!("{value} is not an integer"))
}

#[cfg(test)]
mod tests {
    use super::AutoCompleteConfig;
    use crate::error::Error;
    use crate::types::DataType;
    use serde_json::{Value, json};

    fn config(value: Value) -> Result<AutoCompleteConfig, Error> {
        Ok(AutoCompleteConfig::from_json(&value.to_string())?)
    }

    fn json_of(config: &AutoCompleteConfig) -> Result<Value, Error> {
        Ok(serde_json::from_str(&config.to_json())?)
    }

    #[test]
    fn empty_config_is_filled_in() -> Result<(), Error> {
        let mut config = config(json!({"name": "m", "max_batch_size": 0}))?;
        config.set_max_batch_size(8)?;
        config.set_dynamic_batching()?;
        config.add_input("IMAGE", DataType::Fp32, &[1, 28, 28])?;
        config.add_optional_input("LANGUAGE", DataType::Bytes, &[1])?;
        config.add_output("DIGIT", DataType::Int32, &[1])?;
        config.set_decoupled(false)?;
        assert_eq!(
            json_of(&config)?,
            json!({
                "name": "m",
                "max_batch_size": 8,
                "dynamic_batching": {},
                "input": [
                    {"name": "IMAGE", "data_type": "TYPE_FP32", "dims": [1, 28, 28]},
                    {"name": "LANGUAGE", "data_type": "TYPE_STRING", "dims": [1], "optional": true}
                ],
                "output": [{"name": "DIGIT", "data_type": "TYPE_INT32", "dims": [1]}],
                "model_transaction_policy": {"decoupled": false}
            })
        );
        Ok(())
    }

    #[test]
    fn configured_tensor_gets_what_it_leaves_out() -> Result<(), Error> {
        let mut config = config(json!({
            "name": "m",
            "input": [{"name": "IMAGE", "data_type": "TYPE_INVALID", "dims": []}],
            "output": [{"name": "DIGIT", "data_type": "TYPE_INT32", "dims": ["1"]}]
        }))?;
        config.add_input("IMAGE", DataType::Fp32, &[-1])?;
        config.add_output("DIGIT", DataType::Int32, &[1])?;
        assert_eq!(
            json_of(&config)?["input"],
            json!([{"name": "IMAGE", "data_type": "TYPE_FP32", "dims": [-1]}])
        );
        Ok(())
    }

    #[test]
    fn configured_values_are_not_changed() -> Result<(), Error> {
        let mut config = config(json!({
            "name": "m",
            "max_batch_size": 16,
            "sequence_batching": {},
            "model_transaction_policy": {"decoupled": true},
            "input": [{"name": "IMAGE", "data_type": "TYPE_FP32", "dims": [28, 28], "optional": false}]
        }))?;
        assert!(config.set_max_batch_size(8).is_err());
        assert!(config.set_dynamic_batching().is_err());
        assert!(config.set_decoupled(false).is_err());
        assert!(
            config
                .add_input("IMAGE", DataType::Fp16, &[28, 28])
                .is_err()
        );
        assert!(config.add_input("IMAGE", DataType::Fp32, &[784]).is_err());
        assert!(
            config
                .add_optional_input("IMAGE", DataType::Fp32, &[28, 28])
                .is_err()
        );
        config.set_max_batch_size(32)?;
        config.set_decoupled(true)?;
        config.add_input("IMAGE", DataType::Fp32, &[28, 28])?;
        Ok(())
    }

    #[test]
    fn invalid_values_are_rejected() -> Result<(), Error> {
        let mut config = config(json!({"name": "m"}))?;
        assert!(config.set_max_batch_size(u32::MAX).is_err());
        assert!(config.add_input("", DataType::Fp32, &[1]).is_err());
        assert!(config.add_input("X", DataType::Invalid, &[1]).is_err());
        assert!(config.add_input("X", DataType::Fp32, &[0]).is_err());
        assert!(config.add_output("Y", DataType::Fp32, &[-2]).is_err());
        assert_eq!(json_of(&config)?, json!({"name": "m"}));
        Ok(())
    }

    #[test]
    fn tensors_are_found_by_name() -> Result<(), Error> {
        let mut config = config(json!({"name": "m", "input": [{"name": "IMAGE"}]}))?;
        assert!(config.has_input("IMAGE"));
        assert!(!config.has_output("IMAGE"));
        config.add_output("DIGIT", DataType::Int32, &[1])?;
        assert!(config.has_output("DIGIT"));
        Ok(())
    }

    #[test]
    fn malformed_config_is_rejected() {
        assert!(AutoCompleteConfig::from_json("[]").is_err());
        assert!(AutoCompleteConfig::from_json(r#"{"name": 1}"#).is_err());
    }
}
