use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::{Handle, Runtime};
use tokio::sync::Semaphore;
use triton_ng::backend::Backend;
use triton_ng::{
    BackendHandle, DataType, Error, InferenceFuture, InferenceRequest, InferenceResult, Model,
    Request, Response, Server, log,
};

/// Batches processed at once by the backend. When all of them are busy,
/// `model_instance_execute` waits and Triton keeps collecting the next batch.
const MAX_BATCHES_IN_FLIGHT: usize = 4;

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
            permits: Arc::new(Semaphore::new(MAX_BATCHES_IN_FLIGHT)),
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
            .block_on(executor.permits.acquire_many(MAX_BATCHES_IN_FLIGHT as u32))?
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

        // One task per batch: the batch Triton collected is processed as a whole, while
        // this instance returns and accepts the next one.
        let permit = handle.block_on(Arc::clone(&permits).acquire_owned())?;
        handle.spawn(async move {
            process_batch(requests, &target).await;
            drop(permit);
        });

        Ok(())
    }
}

/// Answers every request of the batch with the result of the target model,
/// or with the error that prevented it. Requests do not affect each other.
///
/// `mnist_onnx` takes one image per call, so every request gets its own BLS
/// call, all of them in flight at once. A target model with a batch dimension
/// would get the whole batch in one BLS call instead.
async fn process_batch(requests: Vec<Request>, target: &Target) {
    let pending: Vec<_> = requests
        .into_iter()
        .filter_map(|request| submit(request, target))
        .collect();

    for (request, output_names, inference) in pending {
        match inference.await {
            Ok(result) => {
                if let Err(e) = respond(request, &output_names, &result) {
                    log::error(&format!("failed to send response: {e}"));
                }
            }
            Err(e) => respond_error(request, &e.to_string()),
        }
    }
}

/// Submits the BLS call of `request`. On error the request is answered and dropped.
fn submit(request: Request, target: &Target) -> Option<(Request, Vec<String>, InferenceFuture)> {
    let prepared = request
        .output_names()
        .map_err(Error::from)
        .and_then(|output_names| {
            let inference = bls_request(&request, target, &output_names)?;
            Ok((output_names, inference))
        });
    match prepared {
        Ok((output_names, inference)) => {
            let inference = target.server.infer_async(inference);
            Some((request, output_names, inference))
        }
        Err(e) => {
            respond_error(request, &e.to_string());
            None
        }
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
