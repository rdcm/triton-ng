use triton_ng::{BatchStrategy, Error, Model, Parameter, RequestView};

const BATCH_KEY: &str = "batch_key";

// A failure and its recovery belong to one model, even when tests load models concurrently.
static FAILED_INITIALIZATIONS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashSet<String>>,
> = std::sync::LazyLock::new(Default::default);

/// Groups the requests of a batch by `batch_key`, the key of the request at
/// the head of the queue first.
struct ByKey {
    model: String,
    mode: String,
}

/// What the strategy keeps about the batch Triton is forming: the key of the
/// request that heads it.
struct Batch {
    /// Whether the head of the batch has been seen, which is what tells a
    /// request without `batch_key` from the state before the first one.
    model: String,
    mode: String,
    started: bool,
    key: Option<String>,
}

#[triton_ng::batch_strategy]
impl BatchStrategy for ByKey {
    type Batch = Batch;

    fn initialize(model: &Model) -> Result<Self, Error> {
        let mode = model
            .config_parameter("strategy_failure")?
            .unwrap_or_default();
        if mode == "init_error" || mode == "init_panic" {
            let name = model.name()?;
            let fail = {
                let mut failed = FAILED_INITIALIZATIONS
                    .lock()
                    .map_err(|_| "initialization history is poisoned")?;
                if failed.remove(&name) {
                    false
                } else {
                    failed.insert(name);
                    true
                }
            };
            if fail {
                if mode == "init_error" {
                    return Err("strategy initialization failed".into());
                }
                std::panic::resume_unwind(Box::new("strategy initialization panicked"));
            }
        }
        let strategy = Self {
            model: model.name()?,
            mode,
        };
        event(&strategy.model, "strategy_created");
        Ok(strategy)
    }

    fn batch(&self) -> Batch {
        if self.mode == "batch_panic" {
            std::panic::resume_unwind(Box::new("batch creation panicked"));
        }
        event(&self.model, "batch_created");
        Batch {
            model: self.model.clone(),
            mode: self.mode.clone(),
            started: false,
            key: None,
        }
    }

    fn include(&self, batch: &mut Batch, request: &RequestView<'_>) -> Result<bool, Error> {
        if self.mode.starts_with("include_") {
            batch.started = true;
        }
        if self.mode == "include_error" {
            return Err("include failed after mutation".into());
        }
        if self.mode == "include_panic" {
            std::panic::resume_unwind(Box::new("include panicked after mutation"));
        }
        if self.mode == "refuse" {
            return Ok(false);
        }
        if request.parameter("inspect_policy_input")? == Some(Parameter::Bool(true)) {
            let input = request.get_input("Input3")?;
            let properties = input.properties_for_host_policy("pinned")?;
            let bytes = input.as_bytes_for_host_policy("pinned")?;
            if properties.byte_size != bytes.len() as u64 || bytes != input.as_bytes()? {
                return Err("queued policy input differs from ordinary input".into());
            }
            let mut pieces = Vec::new();
            for index in 0..properties.buffer_count {
                pieces.extend(input.buffer_for_host_policy("pinned", index)?);
            }
            if pieces != bytes {
                return Err("queued policy input buffers differ".into());
            }
        }
        let key = match request.parameter(BATCH_KEY)? {
            Some(Parameter::String(key)) => Some(key),
            _ => None,
        };
        if !batch.started {
            // The request at the head of the queue joins the batch whatever
            // its key: a strategy that refuses it forms no batch at all.
            batch.started = true;
            batch.key = key;
            return Ok(true);
        }
        Ok(key == batch.key)
    }
}

/// Appends one lifecycle event to the test trace volume. A single write keeps
/// events from concurrently loaded models distinguishable by their name.
#[derive(Default)]
struct Events {
    strategies_created: u64,
    strategies_dropped: u64,
    batches_created: u64,
    batches_dropped: u64,
}

static EVENTS: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<String, Events>>> =
    std::sync::LazyLock::new(Default::default);

fn event(model: &str, kind: &str) {
    let Ok(mut models) = EVENTS.lock() else {
        eprintln!("strategy lifecycle counters are poisoned");
        return;
    };
    let counts = models.entry(model.to_string()).or_default();
    match kind {
        "strategy_created" => counts.strategies_created += 1,
        "strategy_dropped" => counts.strategies_dropped += 1,
        "batch_created" => counts.batches_created += 1,
        "batch_dropped" => counts.batches_dropped += 1,
        _ => return,
    }
    // Publish a bounded snapshot on teardown, including batches released after the strategy.
    // Active queue retries only update counters; they never append an unbounded log.
    if counts.strategies_created == counts.strategies_dropped && counts.strategies_created > 0 {
        let snapshot = format!(
            "[{},{},{},{}]",
            counts.strategies_created,
            counts.strategies_dropped,
            counts.batches_created,
            counts.batches_dropped
        );
        if let Err(error) = std::fs::write(format!("/traces/{model}.lifecycle.json"), snapshot) {
            eprintln!("cannot record strategy lifecycle: {error}");
        }
    }
}

impl Drop for ByKey {
    fn drop(&mut self) {
        event(&self.model, "strategy_dropped");
        if self.mode == "drop_panic" {
            std::panic::resume_unwind(Box::new("strategy drop panicked"));
        }
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        event(&self.model, "batch_dropped");
        if self.mode == "drop_panic" {
            std::panic::resume_unwind(Box::new("batch drop panicked"));
        }
    }
}
