use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::{Handle, Runtime};
use tokio::sync::Semaphore;
use triton_ng::backend::Backend;
use triton_ng::{
    BackendHandle, DataType, Error, InferenceRequest, InferenceResult, Model, Request, Response,
    Server, log,
};

/// Requests processed at once by the backend. When all of them are busy,
/// `model_instance_execute` waits and Triton keeps the rest in its queue.
const MAX_IN_FLIGHT: usize = 64;

struct Executor {
    runtime: Runtime,
    permits: Arc<Semaphore>,
}

static EXECUTOR: Mutex<Option<Executor>> = Mutex::new(None);

fn executor() -> Result<(Handle, Arc<Semaphore>), Error> {
    let executor = EXECUTOR.lock().map_err(|_| "executor lock is poisoned")?;
    let executor = executor.as_ref().ok_or("backend is not initialized")?;
    Ok((
        executor.runtime.handle().clone(),
        Arc::clone(&executor.permits),
    ))
}

struct Target {
    server: Server,
    model: String,
    version: i64,
}

struct MnistBackend;

impl Backend for MnistBackend {
    fn initialize(backend: &BackendHandle) -> Result<(), Error> {
        println!("Initializing backend: {}", backend.name()?);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()?;
        *EXECUTOR.lock().map_err(|_| "executor lock is poisoned")? = Some(Executor {
            runtime,
            permits: Arc::new(Semaphore::new(MAX_IN_FLIGHT)),
        });
        Ok(())
    }

    fn finalize(_backend: &BackendHandle) -> Result<(), Error> {
        let Some(executor) = EXECUTOR
            .lock()
            .map_err(|_| "executor lock is poisoned")?
            .take()
        else {
            return Ok(());
        };
        // Every task holds a permit until it ends, so all permits back means no task is left.
        executor
            .runtime
            .block_on(executor.permits.acquire_many(MAX_IN_FLIGHT as u32))?
            .forget();
        executor.runtime.shutdown_timeout(Duration::from_secs(5));
        Ok(())
    }

    fn model_instance_execute(model: Model, requests: Vec<Request>) -> Result<(), Error> {
        let target = Arc::new(Target {
            server: model.get_server()?,
            model: model
                .config_parameter("target_model")?
                .ok_or("missing 'target_model' config parameter")?,
            version: model.version()? as i64,
        });
        let (handle, permits) = executor()?;

        for request in requests {
            let permit = handle.block_on(Arc::clone(&permits).acquire_owned())?;
            let target = Arc::clone(&target);
            handle.spawn(async move {
                process(request, &target).await;
                drop(permit);
            });
        }

        Ok(())
    }
}

/// Answers `request` with the result of the target model, or with the error
/// that prevented it. Other requests are not affected either way.
async fn process(request: Request, target: &Target) {
    let output_names = match request.output_names() {
        Ok(names) => names,
        Err(e) => return respond_error(request, &e.to_string()),
    };
    let inference = match bls_request(&request, target, &output_names) {
        Ok(inference) => inference,
        Err(e) => return respond_error(request, &e.to_string()),
    };

    match target.server.infer_async(inference).await {
        Ok(result) => {
            if let Err(e) = respond(request, &output_names, &result) {
                log::error(&format!("failed to send response: {e}"));
            }
        }
        Err(e) => respond_error(request, &e.to_string()),
    }
}

fn bls_request(
    request: &Request,
    target: &Target,
    output_names: &[String],
) -> Result<InferenceRequest, Error> {
    let mut inference = InferenceRequest::new(&target.server, &target.model, target.version)?;

    for name in request.input_names()? {
        let input = request.get_input(&name)?;
        let shape = input.properties()?.shape;
        inference.add_input_with_data(&name, DataType::Fp32, &shape, input.as_bytes()?)?;
    }

    for name in output_names {
        inference.add_requested_output(name)?;
    }

    Ok(inference)
}

fn respond(
    request: Request,
    output_names: &[String],
    result: &InferenceResult,
) -> Result<(), Error> {
    let mut response = Response::new(request)?;
    for name in output_names {
        let output = result.output(name)?;
        response
            .create_output(name, DataType::Fp32, &output.shape)?
            .write_fp32_vec(&output.as_fp32_vec()?)?;
    }
    response.send()?;
    Ok(())
}

fn respond_error(request: Request, message: &str) {
    if let Err(e) = request.respond_error(message) {
        log::error(&format!("failed to send error response: {e}"));
    }
}

triton_ng::declare_backend!(MnistBackend);
