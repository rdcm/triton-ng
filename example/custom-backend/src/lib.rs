use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;
use triton_ng::backend::Backend;
use triton_ng::{
    BackendHandle, Context, DataType, Error, InferenceRequest, Model, Parameter, Request, Response,
};

struct State {
    target_model: String,
    version: i64,
    held: AtomicI64,
    cancelled: AtomicI64,
}

struct MnistBackend;

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
            version: model.version()? as i64,
            held: AtomicI64::new(0),
            cancelled: AtomicI64::new(0),
        })?;
        Ok(())
    }

    async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
        let state = ctx.state::<State>()?;
        let mut held = Vec::new();

        for request in requests {
            if request.parameter("hold_until_cancelled")? == Some(Parameter::Bool(true)) {
                held.push(request);
                continue;
            }

            let output_names = request.output_names()?;

            let mut inference_req =
                InferenceRequest::new(ctx.server(), &state.target_model, state.version)?;
            for name in request.input_names()? {
                let input = request.get_input(&name)?;
                let shape = input.properties()?.shape;
                inference_req.add_input_with_data(
                    &name,
                    DataType::Fp32,
                    &shape,
                    input.as_bytes()?,
                )?;
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
            response.set_parameter("held_requests", state.held.load(Ordering::SeqCst))?;
            response.set_parameter("cancelled_requests", state.cancelled.load(Ordering::SeqCst))?;
            response.send()?;
        }

        for request in held {
            hold_until_cancelled(state, request).await?;
        }

        Ok(())
    }
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

triton_ng::declare_backend!(MnistBackend);
