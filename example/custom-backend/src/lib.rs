use std::str::FromStr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;
use triton_ng::backend::Backend;
use triton_ng::{
    BackendHandle, Context, DataType, Error, GpuBuffer, InferenceRequest, MemoryManager, Model,
    ModelInstance, Parameter, PinnedRamBuffer, Request, Response, UninitGpuBuffer,
};

const IMAGE: &str = "Input3";
const IMAGE_SHAPE: [i64; 4] = [1, 1, 28, 28];
const LOGITS: &str = "Plus214_Output_0";
const LOGITS_BYTES: usize = 10 * 4;

#[derive(Clone, Copy)]
enum OutputMemory {
    Pinned,
    Gpu,
    WrongSize,
}

impl FromStr for OutputMemory {
    type Err = Error;

    fn from_str(memory: &str) -> Result<Self, Error> {
        match memory {
            "pinned" => Ok(Self::Pinned),
            "gpu" => Ok(Self::Gpu),
            "wrong_size" => Ok(Self::WrongSize),
            other => Err(format!("unknown memory '{other}'").into()),
        }
    }
}

struct State {
    target_model: String,
    held: AtomicI64,
    cancelled: AtomicI64,
}

#[derive(Default)]
struct Buffers {
    pinned: Mutex<Vec<PinnedRamBuffer>>,
    gpu: Mutex<Vec<GpuBuffer>>,
    pinned_allocated: AtomicI64,
    gpu_allocated: AtomicI64,
}

impl Buffers {
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
        model.set_state(State {
            target_model: model
                .config_parameter("target_model")?
                .ok_or("missing 'target_model' config parameter")?,
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
            if let Some(Parameter::String(memory)) = request.parameter("memory")? {
                match memory.parse::<OutputMemory>() {
                    Ok(memory) => buffered.push((request, memory)),
                    Err(e) => request.respond_error(&e.to_string())?,
                }
                continue;
            }

            let image = request.get_input(IMAGE)?.as_bytes()?;
            pending.push((
                request,
                ctx.server()
                    .infer_async(host_inference(&ctx, state, image)?),
            ));
        }

        for (request, inference) in pending {
            let digit = argmax(&inference.await?.output(LOGITS)?.as_fp32_vec()?)?;
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
    let mut pinned_image = ctx.memory().pinned(image.len())?;
    pinned_image.copy_from_slice(&image);

    let mut inference = InferenceRequest::new(ctx.server(), &state.target_model, -1)?;
    inference.add_input_with_buffer(IMAGE, DataType::Fp32, &IMAGE_SHAPE, pinned_image)?;

    match memory {
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
        OutputMemory::WrongSize => {
            inference.add_requested_output_into(LOGITS, ctx.memory().pinned(4)?)?;
            ctx.server().infer_async(inference).await?;
            Err("output written into a buffer of the wrong size".into())
        }
    }
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
        "pinned_buffers_allocated",
        buffers.pinned_allocated.load(Ordering::SeqCst),
    )?;
    response.set_parameter(
        "gpu_buffers_allocated",
        buffers.gpu_allocated.load(Ordering::SeqCst),
    )?;
    response.send()?;
    Ok(())
}

async fn hold_until_cancelled(state: &State, request: Request) -> Result<(), Error> {
    state.held.fetch_add(1, Ordering::SeqCst);
    let result = respond_when_cancelled(request).await;
    if result.is_ok() {
        state.cancelled.fetch_add(1, Ordering::SeqCst);
    }
    state.held.fetch_sub(1, Ordering::SeqCst);
    result
}

async fn respond_when_cancelled(request: Request) -> Result<(), Error> {
    while !request.is_cancelled()? {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    request.respond_cancelled()?;
    Ok(())
}
