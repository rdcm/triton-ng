use std::str::FromStr;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use triton_ng::backend::Backend;
use triton_ng::{
    BackendHandle, Context, CorrelationId, Counter, CounterFamily, DataType, Error, ErrorCode,
    Gauge, GaugeFamily, GpuBuffer, Histogram, HistogramFamily, InferenceFuture, InferenceRequest,
    InferenceResult, MemoryManager, Model, ModelInstance, Parameter, PinnedRamBuffer, RamBuffer,
    Request, RequestFlags, Response, TritonError, UninitGpuBuffer,
};

const IMAGE: &str = "Input3";
const IMAGE_SHAPE: [i64; 4] = [1, 1, 28, 28];
const LOGITS: &str = "Plus214_Output_0";
const LOGITS_BYTES: usize = 10 * 4;
const IDENTITY_INPUT: &str = "INPUT0";
const IDENTITY_OUTPUT: &str = "OUTPUT0";

#[derive(Clone, Copy)]
enum OutputMemory {
    Ram,
    Pinned,
    Gpu,
    GpuInput,
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
            "wrong_size" => Ok(Self::WrongSize),
            other => Err(format!("unknown memory '{other}'").into()),
        }
    }
}

struct State {
    model: String,
    target_model: String,
    identity_model: String,
    held: AtomicI64,
    cancelled: AtomicI64,
    metrics: Metrics,
}

struct Metrics {
    answered: Counter,
    held: Gauge,
    bls_seconds: Histogram,
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
impl Backend for MnistBackend {
    fn initialize(backend: &BackendHandle) -> Result<(), Error> {
        println!("Initializing backend: {}", backend.name()?);
        Ok(())
    }

    fn model_initialize(model: &Model) -> Result<(), Error> {
        let name = model.name()?;
        model.set_state(State {
            metrics: Metrics::new(&name)?,
            model: name,
            target_model: model
                .config_parameter("target_model")?
                .ok_or("missing 'target_model' config parameter")?,
            identity_model: model
                .config_parameter("identity_model")?
                .ok_or("missing 'identity_model' config parameter")?,
            held: AtomicI64::new(0),
            cancelled: AtomicI64::new(0),
        })?;
        Ok(())
    }

    fn model_instance_initialize(instance: &ModelInstance) -> Result<(), Error> {
        instance.set_state(Buffers::default())
    }

    async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
        let state = ctx.state::<State>()?;
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
            if request.parameter("echo_metadata")? == Some(Parameter::Bool(true)) {
                echo_metadata(request)?;
                continue;
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
            hold_until_cancelled(state, request).await?;
        }

        Ok(())
    }
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
        OutputMemory::WrongSize => {
            inference.add_requested_output_into(LOGITS, ctx.memory().pinned(4)?)?;
            ctx.server().infer_async(inference).await?;
            Err("output written into a buffer of the wrong size".into())
        }
    }
}

fn identity_of(
    ctx: &Context,
    state: &State,
    logits: &Arc<GpuBuffer>,
) -> Result<InferenceFuture, Error> {
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
    response.send()?;
    state.metrics.answered.increment(1.0)?;
    Ok(())
}

async fn hold_until_cancelled(state: &State, request: Request) -> Result<(), Error> {
    state.held.fetch_add(1, Ordering::SeqCst);
    state.metrics.held.increment(1.0)?;
    let result = respond_when_cancelled(request).await;
    if result.is_ok() {
        state.cancelled.fetch_add(1, Ordering::SeqCst);
    }
    state.held.fetch_sub(1, Ordering::SeqCst);
    state.metrics.held.decrement(1.0)?;
    result
}

async fn respond_when_cancelled(request: Request) -> Result<(), Error> {
    while !request.is_cancelled()? {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    request.respond_cancelled()?;
    Ok(())
}
