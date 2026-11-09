use std::collections::BTreeMap;

/// The host policy Triton selected for a model instance: its name and
/// settings, read with [`crate::ModelInstance::host_policy`].
///
/// Triton always names the policy: the `host_policy` field of the
/// `instance_group` when the config sets it, and otherwise `cpu`,
/// `gpu_<device-id>` or `model` after the kind of the group.
///
/// A plain copy: owns its strings and holds nothing of Triton. `Send + Sync`.
///
/// # Examples
///
/// ```ignore
/// let policy = instance.host_policy()?;
/// log::info(&format!("host policy {}", policy.name()));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostPolicy {
    name: String,
    settings: BTreeMap<String, String>,
}

impl HostPolicy {
    /// Parses the JSON of `TRITONBACKEND_ModelInstanceHostPolicy`:
    /// `{"<policy-name>": {"<setting>": "<value>", ...}}`, built by the core in
    /// `backend_model_instance.cc` from the policy name and its settings.
    pub(crate) fn from_json(json: &str) -> Result<Self, String> {
        let value = serde_json::from_str::<serde_json::Value>(json)
            .map_err(|e| format!("host policy is not JSON: {e}"))?;
        let object = value
            .as_object()
            .ok_or_else(|| format!("host policy is not a JSON object: {value}"))?;
        if object.len() != 1 {
            return Err(format!(
                "host policy must contain exactly one policy: {value}"
            ));
        }
        let (name, value) = object
            .iter()
            .next()
            .ok_or_else(|| "host policy contains no policy".to_string())?;
        if name.is_empty() {
            return Err(format!("host policy has no name: {value}"));
        }
        let raw_settings = value
            .as_object()
            .ok_or_else(|| format!("host policy '{name}' is not a JSON object: {value}"))?;

        let mut settings = BTreeMap::new();
        for (key, value) in raw_settings {
            let value = value
                .as_str()
                .ok_or_else(|| format!("host policy setting '{key}' is not a string: {value}"))?;
            settings.insert(key.clone(), value.to_string());
        }
        Ok(Self {
            name: name.clone(),
            settings,
        })
    }

    /// The name of the host policy, as in Triton's `--host-policy` option.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let policy = instance.host_policy()?;
    /// log::info(&format!("host policy: {}", policy.name()));
    /// ```
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The value of the setting `key`, `None` when it is not set.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Some(node) = instance.host_policy()?.setting("numa-node") {
    ///     log::info(&format!("NUMA node: {node}"));
    /// }
    /// ```
    pub fn setting(&self, key: &str) -> Option<&str> {
        self.settings.get(key).map(String::as_str)
    }

    /// Every setting as `(key, value)`, sorted by key. Empty when no settings
    /// were configured for the selected policy name; a default name such as
    /// `cpu` can carry settings too, as in `--host-policy=cpu,cpu-cores=0-3`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// for (key, value) in instance.host_policy()?.settings() {
    ///     log::info(&format!("{key}={value}"));
    /// }
    /// ```
    pub fn settings(&self) -> &BTreeMap<String, String> {
        &self.settings
    }
}

#[cfg(test)]
mod tests {
    use super::HostPolicy;

    #[test]
    fn name_and_settings_are_read() -> Result<(), String> {
        let policy =
            HostPolicy::from_json(r#"{"numa":{"numa-node":"0","cpu-node":"1","strict":"yes"}}"#)?;
        assert_eq!(policy.name(), "numa");
        assert_eq!(policy.setting("numa-node"), Some("0"));
        assert_eq!(policy.setting("missing"), None);
        assert_eq!(
            policy
                .settings()
                .iter()
                .map(|(key, value)| (key.as_str(), value.as_str()))
                .collect::<Vec<_>>(),
            vec![("cpu-node", "1"), ("numa-node", "0"), ("strict", "yes")]
        );
        Ok(())
    }

    #[test]
    fn a_policy_without_settings_is_read() -> Result<(), String> {
        let policy = HostPolicy::from_json(r#"{"gpu_0":{}}"#)?;
        assert_eq!(policy.name(), "gpu_0");
        assert!(policy.settings().is_empty());
        Ok(())
    }

    #[test]
    fn names_are_taken_verbatim() -> Result<(), String> {
        let policy = HostPolicy::from_json(r#"{"numa \"node\" \\ ☃":{"a":"b"}}"#)?;
        assert_eq!(policy.name(), "numa \"node\" \\ ☃");
        assert_eq!(policy.setting("a"), Some("b"));
        Ok(())
    }

    #[test]
    fn malformed_json_and_policy_counts_are_rejected() {
        assert!(HostPolicy::from_json("").is_err());
        assert!(HostPolicy::from_json("not json").is_err());
        assert!(HostPolicy::from_json("[]").is_err());
        assert!(HostPolicy::from_json("{}").is_err());
        assert!(HostPolicy::from_json(r#"{"first":{},"second":{}}"#).is_err());
    }

    #[test]
    fn malformed_policies_and_settings_are_rejected() {
        assert!(HostPolicy::from_json(r#"{"":{}}"#).is_err());
        assert!(HostPolicy::from_json(r#"{"":{"numa-node":"0"}}"#).is_err());
        assert!(HostPolicy::from_json(r#"{"numa":[]}"#).is_err());
        assert!(HostPolicy::from_json(r#"{"numa":{"numa-node":0}}"#).is_err());
    }
}
