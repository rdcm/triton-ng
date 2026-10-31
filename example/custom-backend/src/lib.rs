use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use triton_ng::backend::Backend;
use triton_ng::{
    BackendHandle, Context, DataType, Error, InferenceFuture, InferenceRequest, InferenceResult,
    Model, Request, Response, log,
};

/// How often a waiting request checks whether it was cancelled.
const CANCEL_POLL: Duration = Duration::from_millis(10);

enum Mode {
    /// Forwards every request to `target` and returns its outputs.
    Proxy { target: String, version: i64 },
    /// Holds every request for `WAIT_MS` milliseconds or until it is cancelled,
    /// and answers with the number of requests cancelled so far.
    WaitForCancel,
}

struct State {
    mode: Mode,
    cancelled: AtomicU64,
}

struct MnistBackend;

impl Backend for MnistBackend {
    type State = State;

    fn initialize(backend: &BackendHandle) -> Result<(), Error> {
        println!("Initializing backend: {}", backend.name()?);
        Ok(())
    }

    fn model_initialize(model: &Model) -> Result<State, Error> {
        let mode = match model
            .config_parameter_or("mode", "proxy".to_string())?
            .as_str()
        {
            "proxy" => Mode::Proxy {
                target: model
                    .config_parameter("target_model")?
                    .ok_or("missing 'target_model' config parameter")?,
                version: model.version()? as i64,
            },
            "wait_for_cancel" => Mode::WaitForCancel,
            other => return Err(format!("unknown mode '{other}'").into()),
        };
        Ok(State {
            mode,
            cancelled: AtomicU64::new(0),
        })
    }

    async fn model_instance_execute(
        ctx: Context<State>,
        requests: Vec<Request>,
    ) -> Result<(), Error> {
        match &ctx.state().mode {
            Mode::Proxy { target, version } => proxy_batch(&ctx, target, *version, requests).await,
            Mode::WaitForCancel => wait_for_cancel(ctx.state(), requests).await,
        }
        Ok(())
    }
}

/// Answers every request of the batch with the result of the target model,
/// or with the error that prevented it. Requests do not affect each other.
///
/// `mnist_onnx` takes one image per call, so every request gets its own BLS
/// call, all of them in flight at once. A target model with a batch dimension
/// would get the whole batch in one BLS call instead.
async fn proxy_batch(ctx: &Context<State>, target: &str, version: i64, requests: Vec<Request>) {
    let pending: Vec<_> = requests
        .into_iter()
        .filter_map(|request| submit(ctx, target, version, request))
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
fn submit(
    ctx: &Context<State>,
    target: &str,
    version: i64,
    request: Request,
) -> Option<(Request, Vec<String>, InferenceFuture)> {
    match bls_request(ctx, target, version, &request) {
        Ok((output_names, inference)) => {
            let inference = ctx.server().infer_async(inference);
            Some((request, output_names, inference))
        }
        Err(e) => {
            respond_error(request, &e.to_string());
            None
        }
    }
}

fn bls_request(
    ctx: &Context<State>,
    target: &str,
    version: i64,
    request: &Request,
) -> Result<(Vec<String>, InferenceRequest), Error> {
    let output_names = request.output_names()?;
    let mut inference = InferenceRequest::new(ctx.server(), target, version)?;

    for name in request.input_names()? {
        let input = request.get_input(&name)?;
        let shape = input.properties()?.shape;
        inference.add_input_with_data(&name, DataType::Fp32, &shape, input.as_bytes()?)?;
    }

    for name in &output_names {
        inference.add_requested_output(name)?;
    }

    Ok((output_names, inference))
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

/// Holds the requests of the batch until their `WAIT_MS` runs out or they are
/// cancelled. A cancelled request gets the CANCELLED status, the others get
/// the number of requests cancelled so far.
async fn wait_for_cancel(state: &State, requests: Vec<Request>) {
    let started = Instant::now();
    let mut waiting: Vec<(Request, Instant)> = requests
        .into_iter()
        .filter_map(|request| match wait_ms(&request) {
            Ok(wait) => Some((request, started + wait)),
            Err(e) => {
                respond_error(request, &e.to_string());
                None
            }
        })
        .collect();

    while !waiting.is_empty() {
        let mut still_waiting = Vec::with_capacity(waiting.len());
        for (request, deadline) in waiting {
            match request.is_cancelled() {
                Ok(true) => {
                    state.cancelled.fetch_add(1, Ordering::SeqCst);
                    if let Err(e) = request.respond_cancelled() {
                        log::error(&format!("failed to send cancelled response: {e}"));
                    }
                }
                Ok(false) if Instant::now() >= deadline => {
                    let cancelled = state.cancelled.load(Ordering::SeqCst);
                    if let Err(e) = respond_cancelled_count(request, cancelled) {
                        log::error(&format!("failed to send response: {e}"));
                    }
                }
                Ok(false) => still_waiting.push((request, deadline)),
                Err(e) => respond_error(request, &e.to_string()),
            }
        }
        waiting = still_waiting;
        if !waiting.is_empty() {
            tokio::time::sleep(CANCEL_POLL).await;
        }
    }
}

fn wait_ms(request: &Request) -> Result<Duration, Error> {
    let values = request.get_input("WAIT_MS")?.as_fp32_vec()?;
    let [wait] = values[..] else {
        return Err(format!("WAIT_MS must hold one value, got {}", values.len()).into());
    };
    Ok(Duration::from_millis(wait.max(0.0) as u64))
}

fn respond_cancelled_count(request: Request, cancelled: u64) -> Result<(), Error> {
    let mut response = Response::new(request)?;
    response
        .create_output("CANCELLED", DataType::Fp32, &[1, 1])?
        .write_fp32_vec(&[cancelled as f32])?;
    response.send()?;
    Ok(())
}

fn respond_error(request: Request, message: &str) {
    if let Err(e) = request.respond_error(message) {
        log::error(&format!("failed to send error response: {e}"));
    }
}

triton_ng::declare_backend!(MnistBackend);
