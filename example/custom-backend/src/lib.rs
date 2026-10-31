use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use triton_ng::backend::Backend;
use triton_ng::{
    BackendHandle, Context, DataType, Error, InferenceRequest, Model, Request, Response,
};

struct State {
    target_model: Option<String>,
    version: i64,
    cancelled: AtomicU64,
}

struct MnistBackend;

impl Backend for MnistBackend {
    fn initialize(backend: &BackendHandle) -> Result<(), Error> {
        println!("Initializing backend: {}", backend.name()?);
        Ok(())
    }

    fn model_initialize(model: &Model) -> Result<(), Error> {
        model.set_state(State {
            target_model: model.config_parameter("target_model")?,
            version: model.version()? as i64,
            cancelled: AtomicU64::new(0),
        })?;
        Ok(())
    }

    async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
        let state = ctx.state::<State>()?;
        match &state.target_model {
            Some(target_model) => proxy(&ctx, state, target_model, requests).await,
            None => wait_for_cancel(state, requests).await,
        }
    }
}

async fn proxy(
    ctx: &Context,
    state: &State,
    target_model: &str,
    requests: Vec<Request>,
) -> Result<(), Error> {
    for request in requests {
        let output_names = request.output_names()?;

        let mut inference_req = InferenceRequest::new(ctx.server(), target_model, state.version)?;
        for name in request.input_names()? {
            let input = request.get_input(&name)?;
            let shape = input.properties()?.shape;
            inference_req.add_input_with_data(&name, DataType::Fp32, &shape, input.as_bytes()?)?;
        }
        for name in &output_names {
            inference_req.add_requested_output(name)?;
        }

        let result = ctx.server().infer_async(inference_req).await?;

        let mut response = Response::new(request)?;
        for name in &output_names {
            let output = result.output(name)?;
            response
                .create_output(name, DataType::Fp32, &output.shape)?
                .write_fp32_vec(&output.as_fp32_vec()?)?;
        }
        response.send()?;
    }

    Ok(())
}

async fn wait_for_cancel(state: &State, requests: Vec<Request>) -> Result<(), Error> {
    let started = Instant::now();

    for request in requests {
        let wait_ms = request.get_input("WAIT_MS")?.as_fp32_vec()?[0];
        let deadline = started + Duration::from_millis(wait_ms as u64);

        while !request.is_cancelled()? && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        if request.is_cancelled()? {
            state.cancelled.fetch_add(1, Ordering::SeqCst);
            request.respond_cancelled()?;
            continue;
        }

        let cancelled = state.cancelled.load(Ordering::SeqCst) as f32;
        let mut response = Response::new(request)?;
        response
            .create_output("CANCELLED", DataType::Fp32, &[1, 1])?
            .write_fp32_vec(&[cancelled])?;
        response.send()?;
    }

    Ok(())
}

triton_ng::declare_backend!(MnistBackend);
