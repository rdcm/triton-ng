//! The batching strategy of the `batch_strategy` fixture model: requests of
//! one batch share the value of their `batch_key` request parameter.
//!
//! The model config names this library through `TRITON_BATCH_STRATEGY_PATH`,
//! and the integration test reads back how it grouped the requests from the
//! `batch_keys` response parameter the example backend answers with.

use triton_ng::{BatchStrategy, Error, Model, Parameter, RequestView};

/// The request parameter whose value groups the requests of a batch.
const BATCH_KEY: &str = "batch_key";

// Failure fixtures alternate a failed load and a successful recovery. This
// lets the test clear Triton's retained failed-version status without changing
// repository files, and remains repeatable in one running server.
static INIT_ERROR_NEXT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static INIT_PANIC_NEXT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

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
        if mode == "init_error"
            && INIT_ERROR_NEXT.fetch_xor(true, std::sync::atomic::Ordering::SeqCst)
        {
            return Err("strategy initialization failed".into());
        }
        if mode == "init_panic"
            && INIT_PANIC_NEXT.fetch_xor(true, std::sync::atomic::Ordering::SeqCst)
        {
            std::panic::resume_unwind(Box::new("strategy initialization panicked"));
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
fn event(model: &str, kind: &str) {
    use std::io::Write;
    let result = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/traces/batch_strategy_lifecycle.log")
        .and_then(|mut file| writeln!(file, "{model} {kind}"));
    if let Err(error) = result {
        eprintln!("cannot record strategy lifecycle: {error}");
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
