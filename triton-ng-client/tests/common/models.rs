use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

pub struct Models {
    names: HashMap<String, String>,
    url: String,
}

impl Models {
    pub fn new(test: &str, url: &str) -> Result<Self> {
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
        for entry in templates {
            let original = entry.file_name().to_string_lossy().into_owned();
            let Some(name) = models.names.get(&original) else {
                continue;
            };
            let directory = repository.join(name);
            std::fs::create_dir(&directory)
                .with_context(|| format!("creating model fixture for {test}"))?;
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
        let names: Vec<_> = self.names.values().cloned().collect();
        // Use a separate runtime so cleanup also runs on an early test failure, without
        // trying to enter or drive the test's current-thread Tokio runtime recursively.
        let cleanup = std::thread::spawn(move || -> Result<()> {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(async {
                    let sut = super::sut::Sut::new(&url).await?;
                    for name in names {
                        if let Err(error) = sut.repository_model_unload(&name).await {
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
        // tools/run-tests.py removes directories after all tests have stopped scanning.
    }
}
