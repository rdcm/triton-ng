use std::collections::HashMap;
use std::future::Future;
use std::mem::size_of;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
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
const IMAGE_SHAPE: [i64; 4] = [1, 1, 28, 28];
const IMAGE_BYTES: usize = 28 * 28 * 4;
const LOGITS: &str = "Plus214_Output_0";
const LOGITS_BYTES: usize = 10 * 4;
const IDENTITY_INPUT: &str = "INPUT0";
const IDENTITY_OUTPUT: &str = "OUTPUT0";
/// The two entries the memory usage fixture reports for its model; the report
/// sums them into one.
const CHECK_MODEL_BYTES: [u64; 2] = [111, 222];
/// The pinned buffer the memory usage fixture keeps per instance.
const CHECK_PINNED_BYTES: usize = 4096;

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
    model: String,
    repository: PathBuf,
    decoupled: bool,
    iterative_sequence: bool,
    steps: Mutex<HashMap<CorrelationId, i32>>,
    target_model: String,
    identity_model: String,
    stream_model: Option<String>,
    forward_model: Option<String>,
    forwards_input: bool,
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
                "Requests held until the client cancels them",
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
        model.set_state(State {
            metrics: Metrics::new(&name)?,
            model: name,
            repository: model
                .directory()?
                .parent()
                .ok_or("the model directory has no parent")?
                .to_path_buf(),
            decoupled: model.is_decoupled()?,
            iterative_sequence: model.is_iterative_sequence()?,
            steps: Mutex::new(HashMap::new()),
            target_model: model
                .config_parameter("target_model")?
                .ok_or("missing 'target_model' config parameter")?,
            identity_model: model
                .config_parameter("identity_model")?
                .ok_or("missing 'identity_model' config parameter")?,
            stream_model: model.config_parameter("stream_model")?,
            forward_model: model.config_parameter("forward_model")?,
            forwards_input: model.config_parameter("forwards_input")?.as_deref() == Some("true"),
            instances: Mutex::new(Vec::new()),
            in_flight: AtomicI64::new(0),
            max_in_flight: AtomicI64::new(0),
            held: AtomicI64::new(0),
            cancelled: AtomicI64::new(0),
            kept_inputs: Mutex::new(Vec::new()),
        })?;
        if reports_memory_check(model)? {
            model.report_memory_usage(&[
                BufferAttributes::new(MemoryType::Cpu, CHECK_MODEL_BYTES[0])?,
                BufferAttributes::new(MemoryType::Cpu, CHECK_MODEL_BYTES[1])?,
            ])?;
        } else {
            model.report_memory_usage(&[BufferAttributes::new(
                MemoryType::Cpu,
                size_of::<State>() as u64,
            )?])?;
        }
        Ok(())
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
        if !reports_memory_check(&model)? {
            instance.set_state(Buffers {
                host_policy: Some(host_policy),
                ..Default::default()
            })?;
            instance.report_memory_usage(&[BufferAttributes::new(
                MemoryType::Cpu,
                size_of::<Buffers>() as u64,
            )?])?;
            return Ok(());
        }
        // A pinned buffer that really exists, unlike the size of the Rust state.
        let pinned = model.memory_manager()?.pinned(CHECK_PINNED_BYTES)?;
        instance.set_state(Buffers {
            host_policy: Some(host_policy),
            check: Some(pinned),
            ..Default::default()
        })?;
        instance.report_memory_usage(&[BufferAttributes::new(
            MemoryType::CpuPinned,
            CHECK_PINNED_BYTES as u64,
        )?])?;
        Ok(())
    }

    async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
        let state = ctx.state::<State>()?;
        let _in_flight = InFlight::enter(state);
        if state.iterative_sequence {
            return stream_steps(state, requests);
        }
        if state.decoupled {
            return stream_digits(&ctx, state, requests).await;
        }
        if state.forwards_input {
            return forward_to_identity(&ctx, state, requests).await;
        }
        let mut held = Vec::new();
        let mut buffered = Vec::new();
        let mut pending = Vec::new();

        for request in requests {
            if request.parameter("hold_until_cancelled")? == Some(Parameter::Bool(true)) {
                held.push(request);
                continue;
            }
            if let Some(Parameter::String(code)) = request.parameter("error_code")? {
                match code.parse::<ErrorCode>() {
                    Ok(code) => {
                        request.respond_error_with_code(code, "error requested by the client")?
                    }
                    Err(e) => request.respond_error(&e)?,
                }
                continue;
            }
            if let Some(Parameter::String(model)) = request.parameter("infer_model")? {
                let image = request.get_input(IMAGE)?.as_bytes()?;
                match infer_image(&ctx, &model, image).await {
                    Ok(_) => request.respond_error("the inference succeeded")?,
                    Err(e) => request.respond_error_with_code(e.code(), &e.to_string())?,
                }
                continue;
            }
            if request.parameter("answer_before_bls")? == Some(Parameter::Bool(true)) {
                let image = request.get_input(IMAGE)?.as_bytes()?;
                let mut inference = host_inference(&ctx, state, image)?;
                inference.set_parent_trace(&request)?;
                let unawaited = ctx.server().infer_async(inference);
                respond(&ctx, state, request, 0)?;
                drop(unawaited);
                continue;
            }
            if let Some(Parameter::Int(images)) = request.parameter("stream_images")? {
                let via_infer_async =
                    request.parameter("via_infer_async")? == Some(Parameter::Bool(true));
                let image = request.get_input(IMAGE)?.as_bytes()?;
                let inference = stream_inference(&ctx, state, &request, image, images, false)?;
                if via_infer_async {
                    match ctx.server().infer_async(inference).await {
                        Ok(_) => request.respond_error("the inference succeeded")?,
                        Err(e) => request.respond_error_with_code(e.code(), &e.to_string())?,
                    }
                } else {
                    answer_with_streamed_digits(&ctx, request, inference).await?;
                }
                continue;
            }
            if let Some(Parameter::String(how)) = request.parameter("cancel_stream")? {
                let image = request.get_input(IMAGE)?.as_bytes()?;
                let inference = stream_inference(&ctx, state, &request, image, 1, true)?;
                cancel_stream(&ctx, request, inference, &how).await?;
                continue;
            }
            if request.parameter("into_sender")? == Some(Parameter::Bool(true)) {
                if request.into_sender().is_ok() {
                    return Err("a request of a model that is not decoupled got a sender".into());
                }
                continue;
            }
            if request.parameter("enable_statistics")? == Some(Parameter::Bool(true)) {
                let mut response = Response::new(request)?;
                response
                    .create_output("DIGIT", DataType::Int32, &[1, 1])?
                    .write_i32_vec(&[0])?;
                response.enable_statistics();
                response.send()?;
                continue;
            }
            if let Some(Parameter::String(model)) = request.parameter("load_model")? {
                let file = match request.parameter("load_file")? {
                    Some(Parameter::String(file)) => file,
                    _ => "file:1/model.onnx".to_string(),
                };
                let onnx = state
                    .repository
                    .join(&state.target_model)
                    .join("1/model.onnx");
                let from_repository = request.parameter("load_from")?
                    == Some(Parameter::String("repository".to_string()));
                let server = ctx.server();
                let loaded = if from_repository {
                    server.load_model(&model, None, &[])
                } else {
                    std::fs::read(onnx)
                        .map_err(|e| TritonError::from_message(e.to_string()))
                        .and_then(|onnx| {
                            server.load_model(
                                &model,
                                Some(r#"{"backend": "onnxruntime"}"#),
                                &[(&file, &onnx)],
                            )
                        })
                };
                match loaded {
                    Ok(()) => answer_done(request)?,
                    Err(e) => request.respond_error_with_code(e.code(), &e.to_string())?,
                }
                continue;
            }
            if let Some(Parameter::String(model)) = request.parameter("unload_model")? {
                match ctx.server().unload_model(&model, false) {
                    Ok(()) => answer_done(request)?,
                    Err(e) => request.respond_error_with_code(e.code(), &e.to_string())?,
                }
                continue;
            }
            if request.parameter("reschedule")? == Some(Parameter::Bool(true)) {
                if let Err(e) = request.reschedule() {
                    println!("refused to reschedule: {e}");
                }
                continue;
            }
            if request.parameter("echo_backend_config")? == Some(Parameter::Bool(true)) {
                echo_backend_config(&ctx, request)?;
                continue;
            }
            if request.parameter("echo_host_policy")? == Some(Parameter::Bool(true)) {
                echo_host_policy(&ctx, request)?;
                continue;
            }
            if request.parameter("echo_metadata")? == Some(Parameter::Bool(true)) {
                echo_metadata(request)?;
                continue;
            }
            if request.parameter("echo_instance_details")? == Some(Parameter::Bool(true)) {
                echo_instance_details(&ctx, request)?;
                continue;
            }
            if request.parameter("count_backend_state")? == Some(Parameter::Bool(true)) {
                echo_backend_state_count(&ctx, request)?;
                continue;
            }
            if request.parameter("echo_concurrency")? == Some(Parameter::Bool(true)) {
                echo_concurrency(state, request)?;
                continue;
            }
            if let Some(Parameter::Int(micros)) = request.parameter("hold_micros")? {
                let micros = u64::try_from(micros).map_err(|_| "hold_micros is negative")?;
                // Holds the execution thread of the instance so that a test can
                // observe how many of them run at once.
                std::thread::park_timeout(Duration::from_micros(micros));
            }
            if request.parameter("forward_metadata")? == Some(Parameter::Bool(true)) {
                let image = request.get_input(IMAGE)?.as_bytes()?;
                let result = match metadata_inference(&ctx, state, &request, image) {
                    Ok(inference) => ctx.server().infer_async(inference).await,
                    Err(e) => Err(e),
                };
                match result {
                    Ok(result) => answer_with_parameters_of(request, &result)?,
                    Err(e) => request.respond_error_with_code(e.code(), &e.to_string())?,
                }
                continue;
            }
            if request.parameter("release_shared_inputs")? == Some(Parameter::Bool(true)) {
                kept_inputs(state)?.clear();
            }
            if let Some(Parameter::Bool(keep)) = request.parameter("shared_input")? {
                let (image, copy) = {
                    let input = request.get_input(IMAGE)?;
                    (Arc::new(input.share()?), input.as_bytes()?)
                };
                match digit_of_shared_input(&ctx, state, image, copy, keep).await {
                    Ok(digit) => respond(&ctx, state, request, digit)?,
                    Err(e) => request.respond_error(&e.to_string())?,
                }
                continue;
            }
            if let Some(Parameter::String(memory)) = request.parameter("memory")? {
                match memory.parse::<OutputMemory>() {
                    Ok(memory) => buffered.push((request, memory)),
                    Err(e) => request.respond_error(&e.to_string())?,
                }
                continue;
            }

            let image = request.get_input(IMAGE)?.as_bytes()?;
            let mut inference = host_inference(&ctx, state, image)?;
            inference.set_parent_trace(&request)?;
            pending.push((request, Instant::now(), ctx.server().infer_async(inference)));
        }

        for (request, started, inference) in pending {
            let result = inference.await?;
            state
                .metrics
                .bls_seconds
                .observe(started.elapsed().as_secs_f64())?;
            let digit = argmax(&result.output(LOGITS)?.as_fp32_vec()?)?;
            respond(&ctx, state, request, digit)?;
        }

        for (request, memory) in buffered {
            let image = request.get_input(IMAGE)?.as_bytes()?;
            match digit_with_buffers(&ctx, state, image, memory).await {
                Ok(digit) => respond(&ctx, state, request, digit)?,
                Err(e) => request.respond_error(&e.to_string())?,
            }
        }

        for request in held {
            hold_until_cancelled(state, respond_when_cancelled(&ctx, state, request)).await?;
        }

        Ok(())
    }
}

/// Answers every request of the decoupled model with a stream of the digits
/// of its images, one response per image.
/// Streams the number of every step of a request and reschedules it until
/// it has made the `steps` it asks for (2 by default). Triton r25.09
/// reschedules a request only once, so more steps end with an error.
///
/// The sequence batcher fills the slots of other sequences with null
/// requests, which have no correlation ID: they are dropped unanswered.
/// Whether the model is the fixture that checks the memory usage report, see
/// `models/memory_usage/config.pbtxt`.
fn reports_memory_check(model: &Model) -> Result<bool, Error> {
    Ok(model.config_parameter("report_memory_check")?.as_deref() == Some("true"))
}

fn stream_steps(state: &State, requests: Vec<Request>) -> Result<(), Error> {
    for request in requests {
        let Some(id) = request.correlation_id()? else {
            continue;
        };
        let steps = match request.parameter("steps")? {
            Some(Parameter::Int(steps)) => steps,
            _ => 2,
        };
        let step = {
            let mut done = state.steps.lock().map_err(|_| "steps are poisoned")?;
            let step = done.entry(id.clone()).or_insert(0);
            *step += 1;
            *step
        };

        let mut sender = request.into_sender()?;
        let mut response = sender.response()?;
        response
            .create_output("DIGIT", DataType::Int32, &[1, 1])?
            .write_i32_vec(&[step])?;
        response.send()?;

        let finished = i64::from(step) >= steps;
        let result = if finished {
            sender.finish()
        } else {
            sender.reschedule()
        };
        if finished || result.is_err() {
            state
                .steps
                .lock()
                .map_err(|_| "steps are poisoned")?
                .remove(&id);
        }
        if let Err(e) = result {
            println!("step {step} of {id:?} failed: {e}");
        }
    }
    Ok(())
}

async fn stream_digits(ctx: &Context, state: &State, requests: Vec<Request>) -> Result<(), Error> {
    for request in requests {
        let images = request.get_input(IMAGE)?.as_bytes()?;
        let inferences = images
            .chunks(IMAGE_BYTES)
            .map(|image| {
                let inference = host_inference(ctx, state, image.to_vec())?;
                Ok(ctx.server().infer_async(inference))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let hold = request.parameter("hold_until_cancelled")? == Some(Parameter::Bool(true));
        let unfinished = request.parameter("unfinished")? == Some(Parameter::Bool(true));
        let statistics = request.parameter("enable_statistics")? == Some(Parameter::Bool(true));

        let mut sender = request.into_sender()?;
        for inference in inferences {
            let digit = argmax(&inference.await?.output(LOGITS)?.as_fp32_vec()?)?;
            let mut response = sender.response()?;
            response
                .create_output("DIGIT", DataType::Int32, &[1, 1])?
                .write_i32_vec(&[digit])?;
            response.set_parameter("held_requests", state.held.load(Ordering::SeqCst))?;
            response.set_parameter("cancelled_requests", state.cancelled.load(Ordering::SeqCst))?;
            if statistics {
                response.enable_statistics();
            }
            response.send()?;
        }

        if hold {
            hold_until_cancelled(state, finish_when_cancelled(ctx, state, sender, &images)).await?;
        } else if unfinished {
            drop(sender);
        } else {
            sender.finish()?;
        }
    }
    Ok(())
}

/// A request of `images` copies of `image` to the decoupled model.
fn stream_inference(
    ctx: &Context,
    state: &State,
    request: &Request,
    image: Vec<u8>,
    images: i64,
    hold_until_cancelled: bool,
) -> Result<InferenceRequest, Error> {
    let model = state
        .stream_model
        .as_deref()
        .ok_or("missing 'stream_model' config parameter")?;
    let mut inference = InferenceRequest::new(ctx.server(), model, -1)?;
    let shape = [images, 1, 28, 28];
    inference.add_input_with_data(
        IMAGE,
        DataType::Fp32,
        &shape,
        image.repeat(images.try_into()?),
    )?;
    inference.add_requested_output("DIGIT")?;
    if hold_until_cancelled {
        inference.set_parameter("hold_until_cancelled", true)?;
    }
    inference.set_parent_trace(request)?;
    Ok(inference)
}

/// Answers with the first streamed digit and the number of digits.
async fn answer_with_streamed_digits(
    ctx: &Context,
    request: Request,
    inference: InferenceRequest,
) -> Result<(), Error> {
    let mut stream = ctx.server().infer_stream(inference);
    let mut digits = Vec::new();
    while let Some(result) = stream.next().await {
        match result {
            Ok(result) => digits.extend(result.output("DIGIT")?.as_i32_vec()?),
            Err(e) => return Ok(request.respond_error_with_code(e.code(), &e.to_string())?),
        }
    }

    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[digits.first().copied().unwrap_or_default()])?;
    response.set_parameter("streamed_digits", i64::try_from(digits.len())?)?;
    response.send()?;
    Ok(())
}

/// Cancels a stream after its first result, `how` is `cancel` or `drop`,
/// and answers with how the stream ended.
async fn cancel_stream(
    ctx: &Context,
    request: Request,
    inference: InferenceRequest,
    how: &str,
) -> Result<(), Error> {
    let mut stream = ctx.server().infer_stream(inference);
    stream.next().await.ok_or("the stream is empty")??;
    let ended_with = match how {
        "cancel" => {
            stream.cancel()?;
            let mut ended_with = "final flag".to_string();
            while let Some(result) = stream.next().await {
                if let Err(e) = result {
                    ended_with = e.code().to_string();
                }
            }
            ended_with
        }
        "drop" => {
            drop(stream);
            "dropped".to_string()
        }
        other => return Ok(request.respond_error(&format!("unknown cancel_stream '{other}'"))?),
    };

    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[0])?;
    response.set_parameter("stream_ended_with", ended_with)?;
    response.send()?;
    Ok(())
}

/// Answers with the metadata of the request in response parameters.
fn echo_metadata(request: Request) -> Result<(), Error> {
    let id = request.id()?;
    let correlation_id = request.correlation_id()?;
    let flags = request.flags()?;
    let timeout = request.timeout()?;
    let forwarded = request.parameter("forwarded")?;

    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[0])?;
    if let Some(id) = id {
        response.set_parameter("request_id", id)?;
    }
    match correlation_id {
        Some(CorrelationId::Int(id)) => response.set_parameter("correlation_id", id.to_string())?,
        Some(CorrelationId::String(id)) => response.set_parameter("correlation_id", id)?,
        None => {}
    }
    response.set_parameter("sequence_start", flags.sequence_start)?;
    response.set_parameter("sequence_end", flags.sequence_end)?;
    if let Some(timeout) = timeout {
        response.set_parameter("timeout_us", i64::try_from(timeout.as_micros())?)?;
    }
    if let Some(forwarded) = forwarded {
        response.set_parameter("forwarded", forwarded)?;
    }
    response.send()?;
    Ok(())
}

fn answer_done(request: Request) -> Result<(), Error> {
    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[0])?;
    response.send()?;
    Ok(())
}

/// The policy the backend asks Triton for, `device-blocking` when the model
/// repository runs a second copy of this backend that is configured for it.
fn execution_policy(backend: &Backend) -> Result<ExecutionPolicy, Error> {
    match backend.config()?.parameter("execution-policy") {
        None | Some("blocking") => Ok(ExecutionPolicy::Blocking),
        Some("device-blocking") => Ok(ExecutionPolicy::DeviceBlocking),
        Some(other) => Err(format!("unknown execution-policy '{other}'").into()),
    }
}

/// Answers the host policy the instance hook stored, with its settings joined
/// as `key=value,key=value`.
fn echo_host_policy(ctx: &Context, request: Request) -> Result<(), Error> {
    let policy = ctx
        .instance_state::<Buffers>()?
        .host_policy
        .as_ref()
        .ok_or("the instance hook stored no host policy")?;
    let settings = policy
        .settings()
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(",");
    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[0])?;
    response.set_parameter("host_policy_name", policy.name())?;
    response.set_parameter("host_policy_settings", settings)?;
    response.send()?;
    Ok(())
}

/// Answers how many execute calls of the model ran at once since the last
/// read, and resets the maximum.
fn echo_concurrency(state: &State, request: Request) -> Result<(), Error> {
    let max = state.max_in_flight.swap(0, Ordering::SeqCst);
    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[0])?;
    response.set_parameter("max_in_flight", Parameter::Int(max))?;
    response.set_parameter(
        "in_flight",
        Parameter::Int(state.in_flight.load(Ordering::SeqCst)),
    )?;
    response.send()?;
    Ok(())
}

/// Answers what every instance of the model reported about itself, sorted by
/// instance name.
fn echo_instance_details(ctx: &Context, request: Request) -> Result<(), Error> {
    let mut details = ctx
        .state::<State>()?
        .instances
        .lock()
        .map_err(|_| "the instance list is poisoned")?
        .clone();
    details.sort();
    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[0])?;
    for (index, detail) in details.iter().enumerate() {
        response.set_parameter(&format!("instance_{index}"), detail.as_str())?;
    }
    response.send()?;
    Ok(())
}

/// Counts the call in the state of the backend and answers the new value, so a
/// client can see that two models share one state.
fn echo_backend_state_count(ctx: &Context, request: Request) -> Result<(), Error> {
    let counted = ctx
        .backend_state::<BackendState>()?
        .requests
        .fetch_add(1, Ordering::SeqCst)
        + 1;
    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[0])?;
    response.set_parameter("backend_state_count", Parameter::Int(counted as i64))?;
    response.send()?;
    Ok(())
}

fn echo_backend_config(ctx: &Context, request: Request) -> Result<(), Error> {
    let state = ctx.backend_state::<BackendState>()?;
    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[0])?;
    for (key, value) in state.config.parameters() {
        response.set_parameter(key, value)?;
    }
    response.set_parameter("artifact_type", format!("{:?}", state.artifact_type))?;
    response.set_parameter("execution_policy", format!("{:?}", state.execution_policy))?;
    response.set_parameter(
        "artifact_directory",
        state.artifact_directory.to_string_lossy().as_ref(),
    )?;
    response.send()?;
    Ok(())
}

/// A request sending the image to this model again with metadata derived
/// from `request`, so that the called request echoes it.
fn metadata_inference(
    ctx: &Context,
    state: &State,
    request: &Request,
    image: Vec<u8>,
) -> Result<InferenceRequest, TritonError> {
    let mut inference = InferenceRequest::new(ctx.server(), &state.model, -1)?;
    inference.add_input_with_data(IMAGE, DataType::Fp32, &IMAGE_SHAPE, image)?;
    inference.add_requested_output("DIGIT")?;
    let id = request.id()?.unwrap_or_default();
    inference.set_id(&format!("{id}/bls"))?;
    inference.set_correlation_id(match request.correlation_id()? {
        Some(CorrelationId::Int(id)) => CorrelationId::Int(id + 1),
        Some(CorrelationId::String(id)) => CorrelationId::String(format!("{id}/bls")),
        None => CorrelationId::Int(1),
    })?;
    let flags = request.flags()?;
    inference.set_flags(RequestFlags {
        sequence_start: !flags.sequence_start,
        sequence_end: !flags.sequence_end,
    })?;
    inference.set_timeout(request.timeout()?.unwrap_or(Duration::from_secs(10)) * 2)?;
    inference.set_priority(1)?;
    inference.set_parameter("echo_metadata", true)?;
    inference.set_parameter("forwarded", "by the backend")?;
    inference.set_parent_trace(request)?;
    Ok(inference)
}

fn answer_with_parameters_of(request: Request, result: &InferenceResult) -> Result<(), Error> {
    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&result.output("DIGIT")?.as_i32_vec()?)?;
    for name in [
        "request_id",
        "correlation_id",
        "sequence_start",
        "sequence_end",
        "timeout_us",
        "forwarded",
    ] {
        if let Some(value) = result.parameter(name)? {
            response.set_parameter(name, value)?;
        }
    }
    response.send()?;
    Ok(())
}

async fn infer_image(
    ctx: &Context,
    model: &str,
    image: Vec<u8>,
) -> Result<InferenceResult, TritonError> {
    let mut inference = InferenceRequest::new(ctx.server(), model, -1)?;
    inference.add_input_with_data(IMAGE, DataType::Fp32, &IMAGE_SHAPE, image)?;
    ctx.server().infer_async(inference).await
}

fn host_inference(ctx: &Context, state: &State, image: Vec<u8>) -> Result<InferenceRequest, Error> {
    let mut inference = InferenceRequest::new(ctx.server(), &state.target_model, -1)?;
    inference.add_input_with_data(IMAGE, DataType::Fp32, &IMAGE_SHAPE, image)?;
    inference.add_requested_output(LOGITS)?;
    Ok(inference)
}

async fn digit_with_buffers(
    ctx: &Context,
    state: &State,
    image: Vec<u8>,
    memory: OutputMemory,
) -> Result<i32, Error> {
    let buffers = ctx.instance_state::<Buffers>()?;
    let mut inference = InferenceRequest::new(ctx.server(), &state.target_model, -1)?;
    if let OutputMemory::Ram = memory {
        let mut ram_image = ctx.memory().ram(image.len())?;
        ram_image.copy_from_slice(&image);
        inference.add_input_with_buffer(IMAGE, DataType::Fp32, &IMAGE_SHAPE, ram_image)?;
    } else {
        let mut pinned_image = ctx.memory().pinned(image.len())?;
        pinned_image.copy_from_slice(&image);
        inference.add_input_with_buffer(IMAGE, DataType::Fp32, &IMAGE_SHAPE, pinned_image)?;
    }

    match memory {
        OutputMemory::Ram => {
            inference
                .add_requested_output_into(LOGITS, buffers.ram(ctx.memory(), LOGITS_BYTES)?)?;
            let mut result = ctx.server().infer_async(inference).await?;
            let digit = argmax(&result.output(LOGITS)?.as_fp32_vec()?)?;
            buffers.put_ram(result.take_ram(LOGITS)?)?;
            Ok(digit)
        }
        OutputMemory::Pinned => {
            inference
                .add_requested_output_into(LOGITS, buffers.pinned(ctx.memory(), LOGITS_BYTES)?)?;
            let mut result = ctx.server().infer_async(inference).await?;
            let digit = argmax(&result.output(LOGITS)?.as_fp32_vec()?)?;
            buffers.put_pinned(result.take_pinned(LOGITS)?)?;
            Ok(digit)
        }
        OutputMemory::Gpu => {
            let logits = buffers.gpu(ctx.memory(), ctx.device_id(), LOGITS_BYTES)?;
            inference.add_requested_output_into(LOGITS, logits)?;
            let mut result = ctx.server().infer_async(inference).await?;
            buffers.put_gpu(result.take_gpu(LOGITS)?)?;

            let inference = host_inference(ctx, state, image)?;
            let host = ctx.server().infer_async(inference).await?;
            argmax(&host.output(LOGITS)?.as_fp32_vec()?)
        }
        OutputMemory::GpuInput => {
            inference.add_requested_output_into(
                LOGITS,
                ctx.memory().gpu(ctx.device_id(), LOGITS_BYTES)?,
            )?;
            let mut result = ctx.server().infer_async(inference).await?;
            let logits = Arc::new(result.take_gpu(LOGITS)?);
            let copies = [
                identity_of(ctx, state, &logits)?,
                identity_of(ctx, state, &logits)?,
            ];
            drop(logits);

            let inference = host_inference(ctx, state, image)?;
            let host = ctx.server().infer_async(inference).await?;
            let expected = host.output(LOGITS)?.bytes()?;
            for copy in copies {
                if copy.await?.output(IDENTITY_OUTPUT)?.bytes()? != expected {
                    return Err("the GPU buffer read as an input differs from the logits".into());
                }
            }
            argmax(&host.output(LOGITS)?.as_fp32_vec()?)
        }
        OutputMemory::GpuForward => {
            inference.add_requested_output_into(
                LOGITS,
                ctx.memory().gpu(ctx.device_id(), LOGITS_BYTES)?,
            )?;
            let mut result = ctx.server().infer_async(inference).await?;
            let logits = result.take_gpu(LOGITS)?;
            let model = state
                .forward_model
                .as_deref()
                .ok_or("missing 'forward_model' config parameter")?;
            let mut forward = InferenceRequest::new(ctx.server(), model, -1)?;
            forward.add_input_with_buffer(IDENTITY_INPUT, DataType::Fp32, &[1, 10], logits)?;
            forward.add_requested_output(IDENTITY_OUTPUT)?;
            let forwarded = ctx.server().infer_async(forward).await?;

            let inference = host_inference(ctx, state, image)?;
            let host = ctx.server().infer_async(inference).await?;
            if forwarded.parameter("input_on_gpu")? != Some(Parameter::Bool(true)) {
                return Err("the forwarding model did not get its input on the GPU".into());
            }
            if forwarded.output(IDENTITY_OUTPUT)?.bytes()? != host.output(LOGITS)?.bytes()? {
                return Err("the forwarded GPU input differs from the logits".into());
            }
            argmax(&host.output(LOGITS)?.as_fp32_vec()?)
        }
        OutputMemory::WrongSize => {
            inference.add_requested_output_into(LOGITS, ctx.memory().pinned(4)?)?;
            ctx.server().infer_async(inference).await?;
            Err("output written into a buffer of the wrong size".into())
        }
    }
}

/// Infers the digit of a shared `image` of a request, compared with a `copy`
/// of it. With `keep` the shared image stays in the state of the model until
/// a request with `release_shared_inputs`, which holds the response of its
/// request back that long.
async fn digit_of_shared_input(
    ctx: &Context,
    state: &State,
    image: Arc<SharedInput>,
    copy: Vec<u8>,
    keep: bool,
) -> Result<i32, Error> {
    let mut inference = InferenceRequest::new(ctx.server(), &state.target_model, -1)?;
    let shape = image.shape().to_vec();
    inference.add_input_with_buffer(IMAGE, image.datatype(), &shape, Arc::clone(&image))?;
    inference.add_requested_output(LOGITS)?;
    let shared = ctx.server().infer_async(inference).await?;

    let copied = host_inference(ctx, state, copy)?;
    let copied = ctx.server().infer_async(copied).await?;
    if shared.output(LOGITS)?.bytes()? != copied.output(LOGITS)?.bytes()? {
        return Err("the shared input differs from its copy".into());
    }

    if keep {
        kept_inputs(state)?.push(image);
    }
    argmax(&shared.output(LOGITS)?.as_fp32_vec()?)
}

fn kept_inputs(state: &State) -> Result<MutexGuard<'_, Vec<Arc<SharedInput>>>, Error> {
    state
        .kept_inputs
        .lock()
        .map_err(|_| "the kept inputs are poisoned".into())
}

/// Answers every request with its input as it comes back from the identity
/// model, which reads it where the request keeps it, and tells whether it
/// was in GPU memory.
async fn forward_to_identity(
    ctx: &Context,
    state: &State,
    requests: Vec<Request>,
) -> Result<(), Error> {
    for request in requests {
        let (input_on_gpu, shape, result) = {
            let input = request.get_input(IDENTITY_INPUT)?;
            let input_on_gpu = input.bytes().is_err();
            let shared = input.share()?;
            let shape = shared.shape().to_vec();
            let mut inference = InferenceRequest::new(ctx.server(), &state.identity_model, -1)?;
            inference.add_input_with_buffer(IDENTITY_INPUT, shared.datatype(), &shape, shared)?;
            inference.add_requested_output(IDENTITY_OUTPUT)?;
            (input_on_gpu, shape, ctx.server().infer_async(inference))
        };
        match result.await {
            Ok(result) => {
                let mut response = Response::new(request)?;
                response
                    .create_output(IDENTITY_OUTPUT, DataType::Fp32, &shape)?
                    .write_bytes(result.output(IDENTITY_OUTPUT)?.bytes()?)?;
                response.set_parameter("input_on_gpu", input_on_gpu)?;
                response.send()?;
            }
            Err(e) => request.respond_error_with_code(e.code(), &e.to_string())?,
        }
    }
    Ok(())
}

fn identity_of(
    ctx: &Context,
    state: &State,
    logits: &Arc<GpuBuffer>,
) -> Result<impl Future<Output = Result<InferenceResult, TritonError>> + use<>, Error> {
    let mut inference = InferenceRequest::new(ctx.server(), &state.identity_model, -1)?;
    inference.add_input_with_buffer(
        IDENTITY_INPUT,
        DataType::Fp32,
        &[1, 10],
        Arc::clone(logits),
    )?;
    inference.add_requested_output(IDENTITY_OUTPUT)?;
    Ok(ctx.server().infer_async(inference))
}

fn argmax(logits: &[f32]) -> Result<i32, Error> {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(digit, _)| digit as i32)
        .ok_or_else(|| "mnist returned no logits".into())
}

fn respond(ctx: &Context, state: &State, request: Request, digit: i32) -> Result<(), Error> {
    let buffers = ctx.instance_state::<Buffers>()?;
    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[digit])?;
    response.set_parameter("held_requests", state.held.load(Ordering::SeqCst))?;
    response.set_parameter("cancelled_requests", state.cancelled.load(Ordering::SeqCst))?;
    response.set_parameter("kept_inputs", i64::try_from(kept_inputs(state)?.len())?)?;
    response.set_parameter(
        "ram_buffers_allocated",
        buffers.ram_allocated.load(Ordering::SeqCst),
    )?;
    response.set_parameter(
        "pinned_buffers_allocated",
        buffers.pinned_allocated.load(Ordering::SeqCst),
    )?;
    response.set_parameter(
        "gpu_buffers_allocated",
        buffers.gpu_allocated.load(Ordering::SeqCst),
    )?;
    state.metrics.answered.increment(1.0)?;
    response.send()?;
    Ok(())
}

async fn hold_until_cancelled(
    state: &State,
    answer: impl Future<Output = Result<(), Error>>,
) -> Result<(), Error> {
    state.held.fetch_add(1, Ordering::SeqCst);
    state.metrics.held.increment(1.0)?;
    let result = answer.await;
    if result.is_ok() {
        state.cancelled.fetch_add(1, Ordering::SeqCst);
    }
    state.held.fetch_sub(1, Ordering::SeqCst);
    state.metrics.held.decrement(1.0)?;
    result
}

/// One step of a long request. Triton only marks a request as cancelled, so a
/// backend checks that between the steps of its work.
async fn work_step(ctx: &Context, state: &State, images: &[u8]) -> Result<(), Error> {
    let image = images
        .get(..IMAGE_BYTES)
        .ok_or("the request has no image")?;
    let inference = host_inference(ctx, state, image.to_vec())?;
    ctx.server().infer_async(inference).await?;
    Ok(())
}

async fn respond_when_cancelled(
    ctx: &Context,
    state: &State,
    request: Request,
) -> Result<(), Error> {
    let images = request.get_input(IMAGE)?.as_bytes()?;
    while !request.is_cancelled()? {
        work_step(ctx, state, &images).await?;
    }
    request.respond_cancelled()?;
    Ok(())
}

async fn finish_when_cancelled(
    ctx: &Context,
    state: &State,
    sender: ResponseSender,
    images: &[u8],
) -> Result<(), Error> {
    while !sender.is_cancelled()? {
        work_step(ctx, state, images).await?;
    }
    sender.respond_cancelled()?;
    Ok(())
}
