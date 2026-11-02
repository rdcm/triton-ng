use std::cell::Cell;
use std::path::{Component, Path};

/// Prefix of the name of a file parameter of a model load.
pub(crate) const FILE_PREFIX: &str = "file:";

thread_local! {
    /// The hook running on this thread in which models must not be loaded or
    /// unloaded, see [`forbid_model_control`].
    static FORBIDDEN_IN: Cell<Option<&'static str>> = const { Cell::new(None) };
}

/// Runs `f`, a hook of a backend, refusing [`crate::Server::load_model`] and
/// [`crate::Server::unload_model`] on this thread while it runs, like
/// python_backend does in `finalize` and `auto_complete_config`: these hooks
/// run inside a load or unload of Triton.
pub(crate) fn forbid_model_control<T>(hook: &'static str, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<&'static str>);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORBIDDEN_IN.set(self.0);
        }
    }

    let _restore = Restore(FORBIDDEN_IN.replace(Some(hook)));
    f()
}

/// Fails when the thread runs a hook given to [`forbid_model_control`].
pub(crate) fn check_model_control_allowed(action: &str) -> Result<(), String> {
    match FORBIDDEN_IN.get() {
        Some(hook) => Err(format!("models cannot be {action} in {hook}")),
        None => Ok(()),
    }
}

/// Checks the arguments of [`crate::Server::load_model`] before they reach
/// Triton.
pub(crate) fn check_load_arguments(
    model_name: &str,
    config: Option<&str>,
    files: &[(&str, &[u8])],
) -> Result<(), String> {
    if model_name.is_empty() {
        return Err("the model name is empty".to_string());
    }
    if let Some(config) = config {
        let value = serde_json::from_str::<serde_json::Value>(config)
            .map_err(|e| format!("the config of model '{model_name}' is not JSON: {e}"))?;
        if !value.is_object() {
            return Err(format!(
                "the config of model '{model_name}' is not a JSON object"
            ));
        }
    }
    if !files.is_empty() && config.is_none() {
        return Err(format!(
            "the files of model '{model_name}' are given without its config"
        ));
    }
    for (i, (name, _)) in files.iter().enumerate() {
        check_file_name(name)?;
        if files.iter().take(i).any(|(seen, _)| seen == name) {
            return Err(format!("file '{name}' is given twice"));
        }
    }
    Ok(())
}

/// A file is named `file:<path>`, with a relative `<path>` inside the model
/// directory.
fn check_file_name(name: &str) -> Result<(), String> {
    let path = name
        .strip_prefix(FILE_PREFIX)
        .ok_or_else(|| format!("file name '{name}' does not start with '{FILE_PREFIX}'"))?;
    if path.is_empty() {
        return Err(format!("file name '{name}' has no path"));
    }
    let inside = Path::new(path)
        .components()
        .all(|component| matches!(component, Component::Normal(_) | Component::CurDir));
    if !inside {
        return Err(format!(
            "file '{name}' is not a relative path inside the model directory"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{check_load_arguments, check_model_control_allowed, forbid_model_control};

    const CONFIG: Option<&str> = Some(r#"{"backend":"onnxruntime"}"#);

    #[test]
    fn model_from_the_repository_needs_only_a_name() {
        assert_eq!(check_load_arguments("m", None, &[]), Ok(()));
        assert!(check_load_arguments("", None, &[]).is_err());
    }

    #[test]
    fn config_is_a_json_object() {
        assert_eq!(check_load_arguments("m", CONFIG, &[]), Ok(()));
        assert!(check_load_arguments("m", Some("not json"), &[]).is_err());
        assert!(check_load_arguments("m", Some("[]"), &[]).is_err());
        assert!(check_load_arguments("m", Some(""), &[]).is_err());
    }

    #[test]
    fn files_need_a_config() {
        let files: &[(&str, &[u8])] = &[("file:1/model.onnx", b"onnx")];
        assert_eq!(check_load_arguments("m", CONFIG, files), Ok(()));
        assert!(check_load_arguments("m", None, files).is_err());
    }

    #[test]
    fn files_stay_inside_the_model_directory() {
        for name in [
            "1/model.onnx",
            "file:",
            "file:/etc/passwd",
            "file:../other/1/model.onnx",
            "file:1/../../model.onnx",
        ] {
            let files: &[(&str, &[u8])] = &[(name, b"onnx")];
            assert!(check_load_arguments("m", CONFIG, files).is_err(), "{name}");
        }
    }

    #[test]
    fn repeated_file_is_rejected() {
        let files: &[(&str, &[u8])] = &[("file:1/model.onnx", b"a"), ("file:1/model.onnx", b"b")];
        assert!(check_load_arguments("m", CONFIG, files).is_err());
    }

    #[test]
    fn model_control_is_refused_only_inside_a_forbidding_hook() {
        assert_eq!(check_model_control_allowed("loaded"), Ok(()));
        let inside = forbid_model_control("finalize", || check_model_control_allowed("loaded"));
        assert_eq!(
            inside,
            Err("models cannot be loaded in finalize".to_string())
        );
        assert_eq!(check_model_control_allowed("loaded"), Ok(()));
    }
}
