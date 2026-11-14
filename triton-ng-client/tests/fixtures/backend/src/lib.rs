//! Shared hooks for isolated model fixtures. See ../README.md.

mod backend_config;
use backend_config::{echo_backend_config, execution_policy};
mod backend_state;
mod batching;
mod bls_mutation;
use bls_mutation::wait_for_input_release;
mod bls_streaming;
mod cancellation;
use cancellation::{hold_until_cancelled, work_step};
mod common;
use common::{answer_done, argmax, host_inference, infer_image, respond};
mod concurrency;
mod errors;
mod host_policy;
mod inference;
mod input_forward;
use input_forward::forward_to_identity;
mod instance_details;
mod iterative_sequence;
use iterative_sequence::stream_steps;
mod memory_buffers;
mod memory_usage;
mod metadata;
mod model_control;
mod policy_append;
mod reschedule_refusal;
mod response_statistics;
mod sender_refusal;
mod sequence;
use sequence::count_sequence;
mod shared_inputs;
use shared_inputs::kept_inputs;
mod streaming;
use streaming::{answer_with_streamed_digits, cancel_stream, stream_digits, stream_inference};
mod tracing;

use std::collections::HashMap;
use std::future::Future;
use std::mem::size_of;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use triton_ng::{
    ArtifactType, AutoCompleteConfig, Backend, BackendAttributes, BackendConfig, BufferAttributes,
    Context, CorrelationId, Counter, CounterFamily, CustomBackend, DataType, Error, ErrorCode,
    ExecutionPolicy, Gauge, GaugeFamily, GpuBuffer, Histogram, HistogramFamily, HostPolicy,
    InferenceRequest, InferenceResult, InstanceGroupKind, MemoryManager, MemoryType, Model,
    ModelInstance, Parameter, PinnedRamBuffer, RamBuffer, Request, RequestFlags, Response,
    ResponseSender, SharedInput, TritonError, UninitGpuBuffer,
};

const IMAGE: &str = "Input3";
/// The two names of the sequence state of the `sequence_state` model, from the
/// `state` section of its configuration: Triton hands the input state to the
/// request under `COUNTER_IN`, and the backend writes the output state under
/// `COUNTER_OUT`.
const COUNTER_IN: &str = "counter_in";
const COUNTER_OUT: &str = "counter_out";
const COUNTER_OUTPUT: &str = "COUNTER";
const IMAGE_SHAPE: [i64; 4] = [1, 1, 28, 28];
const IMAGE_BYTES: usize = 28 * 28 * 4;
/// The request parameter the batching strategy of the `batch_strategy` model
/// groups by.
const BATCH_KEY: &str = "batch_key";
const LOGITS: &str = "Plus214_Output_0";
const LOGITS_BYTES: usize = 10 * 4;
const IDENTITY_INPUT: &str = "INPUT0";
const IDENTITY_OUTPUT: &str = "OUTPUT0";
/// The two entries the memory usage fixture reports for its model; the report
/// sums them into one.
const CHECK_MODEL_BYTES: [u64; 2] = [111, 222];
/// The pinned buffer the memory usage fixture keeps per instance.
const CHECK_PINNED_BYTES: usize = 4096;

/// The states every model of the fixture declares, by model name, so that a test
/// can ask one model to write the state of another.
static DECLARED_STATES: LazyLock<Mutex<HashMap<String, Vec<triton_ng::StateConfig>>>> =
    LazyLock::new(Default::default);

#[derive(Clone, Copy)]
enum OutputMemory {
    Ram,
    Pinned,
    Gpu,
    GpuInput,
    GpuForward,
    WrongSize,
}

impl FromStr for OutputMemory {
    type Err = Error;

    fn from_str(memory: &str) -> Result<Self, Error> {
        match memory {
            "ram" => Ok(Self::Ram),
            "pinned" => Ok(Self::Pinned),
            "gpu" => Ok(Self::Gpu),
            "gpu_input" => Ok(Self::GpuInput),
            "gpu_forward" => Ok(Self::GpuForward),
            "wrong_size" => Ok(Self::WrongSize),
            other => Err(format!("unknown memory '{other}'").into()),
        }
    }
}

struct BackendState {
    config: BackendConfig,
    artifact_type: ArtifactType,
    artifact_directory: PathBuf,
    /// The policy Triton holds for this backend, read back after it was set.
    execution_policy: ExecutionPolicy,
    /// Counted by `count_backend_state`, to show that the state of the backend
    /// is one value for every model that uses it.
    requests: AtomicU64,
}

struct State {
    fixture: String,
    model: String,
    repository: PathBuf,
    steps: Mutex<HashMap<CorrelationId, i32>>,
    target_model: String,
    identity_model: String,
    stream_model: Option<String>,
    forward_model: Option<String>,
    /// The states the model configuration declares, from the configuration.
    states: Vec<triton_ng::StateConfig>,
    /// What every instance reported in its own hook, answered by
    /// `echo_instance_details`.
    instances: Mutex<Vec<String>>,
    /// How many execute calls of this model run at once, and the most that ever
    /// did. A model whose instances share an execution thread under
    /// `ExecutionPolicy::DeviceBlocking` never goes past one.
    in_flight: AtomicI64,
    max_in_flight: AtomicI64,
    held: AtomicI64,
    cancelled: AtomicI64,
    kept_inputs: Mutex<Vec<Arc<SharedInput>>>,
    metrics: Metrics,
}

struct Metrics {
    answered: Counter,
    held: Gauge,
    bls_seconds: Histogram,
}

/// Counts the execute calls of a model that run at once.
///
/// Dropped when the call returns, so `in_flight` goes back down however the
/// call ends.
struct InFlight<'a> {
    state: &'a State,
}

impl<'a> InFlight<'a> {
    fn enter(state: &'a State) -> Self {
        let in_flight = state.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        state.max_in_flight.fetch_max(in_flight, Ordering::SeqCst);
        Self { state }
    }
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.state.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Metrics {
    fn new(model: &str) -> Result<Self, Error> {
        let labels = [("model", model)];
        Ok(Self {
            answered: CounterFamily::new(
                "triton_ng_example_answered_total",
                "Requests answered with a digit",
            )?
            .counter(&labels)?,
            held: GaugeFamily::new(
                "triton_ng_example_held_requests",
                "Requests an instance holds until the client cancels them or hold_micros runs out",
            )?
            .gauge(&labels)?,
            bls_seconds: HistogramFamily::new(
                "triton_ng_example_bls_seconds",
                "Duration of the BLS calls to the target model",
                &[0.001, 0.01, 0.1, 1.0],
            )?
            .histogram(&labels)?,
        })
    }
}

#[derive(Default)]
struct Buffers {
    ram: Mutex<Vec<RamBuffer>>,
    pinned: Mutex<Vec<PinnedRamBuffer>>,
    gpu: Mutex<Vec<GpuBuffer>>,
    ram_allocated: AtomicI64,
    pinned_allocated: AtomicI64,
    gpu_allocated: AtomicI64,
    /// The host policy Triton gave this instance, read in the instance hook and
    /// answered by `echo_host_policy`.
    host_policy: Option<HostPolicy>,
    /// The pinned buffer of the memory usage fixture, kept for as long as the
    /// instance lives so that the report describes an allocation that exists.
    #[expect(dead_code, reason = "kept alive for the memory usage report")]
    check: Option<PinnedRamBuffer>,
}

impl Buffers {
    fn ram(&self, memory: &MemoryManager, len: usize) -> Result<RamBuffer, Error> {
        let mut cached = self.ram.lock().map_err(|_| "buffer cache is poisoned")?;
        match cached.iter().position(|buffer| buffer.len() == len) {
            Some(index) => Ok(cached.swap_remove(index)),
            None => {
                self.ram_allocated.fetch_add(1, Ordering::SeqCst);
                memory.ram(len)
            }
        }
    }

    fn pinned(&self, memory: &MemoryManager, len: usize) -> Result<PinnedRamBuffer, Error> {
        let mut cached = self.pinned.lock().map_err(|_| "buffer cache is poisoned")?;
        match cached.iter().position(|buffer| buffer.len() == len) {
            Some(index) => Ok(cached.swap_remove(index)),
            None => {
                self.pinned_allocated.fetch_add(1, Ordering::SeqCst);
                memory.pinned(len)
            }
        }
    }

    fn gpu(
        &self,
        memory: &MemoryManager,
        device_id: i32,
        len: usize,
    ) -> Result<UninitGpuBuffer, Error> {
        let mut cached = self.gpu.lock().map_err(|_| "buffer cache is poisoned")?;
        match cached
            .iter()
            .position(|buffer| buffer.len() == len && buffer.device_id() == device_id)
        {
            Some(index) => Ok(cached.swap_remove(index).into_uninit()),
            None => {
                self.gpu_allocated.fetch_add(1, Ordering::SeqCst);
                memory.gpu(device_id, len)
            }
        }
    }

    fn put_ram(&self, buffer: RamBuffer) -> Result<(), Error> {
        self.ram
            .lock()
            .map_err(|_| "buffer cache is poisoned")?
            .push(buffer);
        Ok(())
    }

    fn put_pinned(&self, buffer: PinnedRamBuffer) -> Result<(), Error> {
        self.pinned
            .lock()
            .map_err(|_| "buffer cache is poisoned")?
            .push(buffer);
        Ok(())
    }

    fn put_gpu(&self, buffer: GpuBuffer) -> Result<(), Error> {
        self.gpu
            .lock()
            .map_err(|_| "buffer cache is poisoned")?
            .push(buffer);
        Ok(())
    }
}

struct MnistBackend;

#[triton_ng::backend]
impl CustomBackend for MnistBackend {
    fn initialize(backend: &Backend) -> Result<(), Error> {
        println!("Initializing backend: {}", backend.name()?);
        let (artifact_type, artifact_directory) = backend.artifacts()?;
        let policy = execution_policy(backend)?;
        backend.set_execution_policy(policy)?;
        let held = backend.execution_policy()?;
        println!("Backend execution policy: set {policy:?}, Triton holds {held:?}");
        backend.set_state(BackendState {
            config: backend.config()?,
            artifact_type,
            artifact_directory,
            execution_policy: held,
            requests: AtomicU64::new(0),
        })?;
        let state = backend.state::<BackendState>()?;
        println!(
            "Backend artifacts: {:?} at {}",
            state.artifact_type,
            state.artifact_directory.display()
        );
        Ok(())
    }

    fn attributes(backend: &Backend) -> Result<BackendAttributes, Error> {
        // Instances that share an execution thread under
        // `ExecutionPolicy::DeviceBlocking` are created one after another: the
        // core gives an instance the thread of a sibling it has already
        // created, and loading them in parallel hides that sibling.
        let parallel = execution_policy(backend)? == ExecutionPolicy::Blocking;
        Ok(BackendAttributes::new()
            .preferred_instance_group(InstanceGroupKind::Cpu, 2, &[])?
            .parallel_model_instance_loading(parallel))
    }

    fn auto_complete_config(_model: &Model, config: &mut AutoCompleteConfig) -> Result<(), Error> {
        if config.has_input(IMAGE) || config.has_input(IDENTITY_INPUT) {
            return Ok(());
        }
        config.set_max_batch_size(8)?;
        config.set_dynamic_batching()?;
        config.add_input(IMAGE, DataType::Fp32, &IMAGE_SHAPE[1..])?;
        config.add_output("DIGIT", DataType::Int32, &[1])?;
        Ok(())
    }

    fn model_initialize(model: &Model) -> Result<(), Error> {
        let name = model.name()?;
        let states = model.sequence_states()?;
        DECLARED_STATES
            .lock()
            .map_err(|_| "the declared states are poisoned")?
            .insert(name.clone(), states.clone());
        model.set_state(State {
            fixture: model
                .config_parameter("fixture")?
                .unwrap_or_else(|| "inference".to_string()),
            metrics: Metrics::new(&name)?,
            model: name,
            repository: model
                .directory()?
                .parent()
                .ok_or("the model directory has no parent")?
                .to_path_buf(),
            steps: Mutex::new(HashMap::new()),
            target_model: model
                .config_parameter("target_model")?
                .ok_or("missing 'target_model' config parameter")?,
            identity_model: model
                .config_parameter("identity_model")?
                .ok_or("missing 'identity_model' config parameter")?,
            stream_model: model.config_parameter("stream_model")?,
            forward_model: model.config_parameter("forward_model")?,
            states,
            instances: Mutex::new(Vec::new()),
            in_flight: AtomicI64::new(0),
            max_in_flight: AtomicI64::new(0),
            held: AtomicI64::new(0),
            cancelled: AtomicI64::new(0),
            kept_inputs: Mutex::new(Vec::new()),
        })?;
        memory_usage::report_model(model)
    }

    fn model_instance_initialize(instance: &ModelInstance) -> Result<(), Error> {
        let host_policy = instance.host_policy()?;
        let details = format!(
            "{} passive={} profiles={:?} secondary_devices={:?} host_policy={:?} settings={:?}",
            instance.name()?,
            instance.is_passive()?,
            instance.profile_names()?,
            instance.secondary_devices()?,
            host_policy.name(),
            host_policy.settings()
        );
        println!("Initializing instance {details}");
        let model = instance.model()?;
        model
            .state::<State>()?
            .instances
            .lock()
            .map_err(|_| "the instance list is poisoned")?
            .push(details);
        memory_usage::initialize_buffers(instance, host_policy)
    }

    async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
        let state = ctx.state::<State>()?;
        let _in_flight = InFlight::enter(state);
        match state.fixture.as_str() {
            "errors" => errors::execute(&ctx, requests).await,
            "cancellation" => cancellation::execute(&ctx, requests).await,
            "tracing" => tracing::execute(&ctx, requests).await,
            "bls_streaming" => bls_streaming::execute(&ctx, requests).await,
            "sender_refusal" => sender_refusal::execute(&ctx, requests).await,
            "shared_inputs" => shared_inputs::execute(&ctx, requests).await,
            "response_statistics" => response_statistics::execute(&ctx, requests).await,
            "model_control" => model_control::execute(&ctx, requests).await,
            "reschedule_refusal" => reschedule_refusal::execute(&ctx, requests).await,
            "backend_config" => backend_config::execute(&ctx, requests).await,
            "bls_mutation" => bls_mutation::execute(&ctx, requests).await,
            "policy_append" => policy_append::execute(&ctx, requests).await,
            "host_policy" => host_policy::execute(&ctx, requests).await,
            "metadata" => metadata::execute(&ctx, requests).await,
            "instance_details" => instance_details::execute(&ctx, requests).await,
            "backend_state" => backend_state::execute(&ctx, requests).await,
            "concurrency" => concurrency::execute(&ctx, requests).await,
            "memory_buffers" => memory_buffers::execute(&ctx, requests).await,
            "batching" => batching::execute(&ctx, requests).await,
            "streaming" => stream_digits(&ctx, state, requests).await,
            "sequence" => count_sequence(state, requests),
            "iterative_sequence" => stream_steps(state, requests),
            "input_forward" => forward_to_identity(&ctx, state, requests).await,
            "inference" | "memory_usage" => inference::execute(&ctx, requests).await,
            other => Err(format!("unknown test fixture '{other}'").into()),
        }
    }
}
