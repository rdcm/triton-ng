use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

pub struct Models {
    names: HashMap<String, String>,
    url: String,
    cleanup: Vec<String>,
}

impl Models {
    pub fn new(url: &str, fixtures: &[&str]) -> Result<Self> {
        let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models");
        let run =
            std::env::var("TRITON_NG_TEST_RUN").context("run integration tests with make tests")?;
        let prefix = format!(
            "test_{run}_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let templates = std::fs::read_dir(&repository)?.collect::<std::io::Result<Vec<_>>>()?;
        let mut models = Self {
            names: HashMap::new(),
            url: url.to_string(),
            cleanup: Vec::new(),
        };
        for entry in &templates {
            if entry.path().join("config.pbtxt").is_file()
                && !entry.file_name().to_string_lossy().starts_with("test_")
            {
                let name = entry.file_name().to_string_lossy().into_owned();
                models
                    .names
                    .insert(name.clone(), format!("{prefix}_{name}"));
            }
        }
        for name in ["mnist_from_files", "mnist_escaped", "no_such_model"] {
            models
                .names
                .insert(name.to_string(), format!("{prefix}_{name}"));
        }
        for name in
            fixtures
                .iter()
                .copied()
                .chain(["mnist_from_files", "mnist_escaped", "no_such_model"])
        {
            models.cleanup.push(models.name(name).to_string());
        }
        for entry in templates {
            let original = entry.file_name().to_string_lossy().into_owned();
            if !fixtures.contains(&original.as_str()) {
                continue;
            }
            let name = models
                .names
                .get(&original)
                .with_context(|| format!("unknown fixture {original}"))?;
            let directory = repository.join(name);
            std::fs::create_dir(&directory)
                .with_context(|| format!("creating model fixture {}", directory.display()))?;
            let mut config = std::fs::read_to_string(entry.path().join("config.pbtxt"))?;
            config = config.replacen(
                &format!("name: \"{original}\""),
                &format!("name: \"{name}\""),
                1,
            );
            // Read-only inference targets are shared. Stateful BLS streams belong to this test.
            if let Some(stream) = models.names.get("mnist_stream") {
                config = config.replace(
                    "string_value: \"mnist_stream\"",
                    &format!("string_value: \"{stream}\""),
                );
            }
            std::fs::write(directory.join("config.pbtxt"), config)?;
            std::fs::create_dir(directory.join("1"))?;
            if entry.path().join("1/model.onnx").is_file() {
                std::os::unix::fs::symlink(
                    format!("../../{original}/1/model.onnx"),
                    directory.join("1/model.onnx"),
                )?;
            }
        }
        Ok(models)
    }

    pub fn name<'a>(&'a self, name: &'a str) -> &'a str {
        self.names.get(name).map_or(name, String::as_str)
    }
}

impl Drop for Models {
    fn drop(&mut self) {
        let url = self.url.clone();
        let names = self.cleanup.clone();
        // Use a separate runtime so cleanup also runs on an early test failure, without
        // trying to enter or drive the test's current-thread Tokio runtime recursively.
        let cleanup = std::thread::spawn(move || -> Result<()> {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(async {
                    let client = triton_ng_client::TritonClient::new(
                        triton_ng_client::TritonClientConfig::new(&url),
                    )
                    .await?;
                    for name in names {
                        if let Err(error) = client.repository_model_unload(&name, None).await {
                            eprintln!("cannot unload test model {name}: {error}");
                        }
                    }
                    Ok(())
                })
        });
        match cleanup.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => eprintln!("cannot clean up test models: {error}"),
            Err(_) => eprintln!("test model cleanup thread panicked"),
        }
        // make tests removes directories after all tests have stopped scanning.
    }
}
