use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;
use triton_ng::backend::Backend;
use triton_ng::{
    BackendHandle, Context, DataType, Error, InferenceRequest, Model, Parameter, Request, Response,
};

struct State {
    target_model: String,
    held: AtomicI64,
    cancelled: AtomicI64,
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

    async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
        let state = ctx.state::<State>()?;
        let mut held = Vec::new();
        let mut pending = Vec::new();

        for request in requests {
            if request.parameter("hold_until_cancelled")? == Some(Parameter::Bool(true)) {
                held.push(request);
                continue;
            }

            let image = request.get_input("Input3")?.as_bytes()?;
            let mut inference_req = InferenceRequest::new(ctx.server(), &state.target_model, -1)?;
            inference_req.add_input_with_data("Input3", DataType::Fp32, &[1, 1, 28, 28], image)?;
            inference_req.add_requested_output("Plus214_Output_0")?;
            pending.push((request, ctx.server().infer_async(inference_req)));
        }

        for (request, inference) in pending {
            let logits = inference.await?.output("Plus214_Output_0")?.as_fp32_vec()?;
            let digit = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(digit, _)| digit as i32)
                .ok_or("mnist returned no logits")?;

            let mut response = Response::new(request)?;
            response
                .create_output("DIGIT", DataType::Int32, &[1, 1])?
                .write_i32_vec(&[digit])?;
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
