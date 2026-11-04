use futures::future::join_all;
use std::str::FromStr;
use triton_ng::{
    Context, CustomBackend, DataType, Error, InferenceRequest, InferenceResult, Model, Request,
    Response, TritonError,
};

const IMAGE: &str = "Input3";
const IMAGE_SHAPE: [i64; 4] = [1, 1, 28, 28];
const IMAGE_BYTES: usize = 28 * 28 * 4;
const LOGITS: &str = "Plus214_Output_0";
const DIGIT: &str = "DIGIT";

#[derive(Clone, Copy)]
enum Mode {
    Argmax,
    Infer,
    InferConcurrent,
    Decoupled,
    Stream,
}

impl FromStr for Mode {
    type Err = Error;

    fn from_str(mode: &str) -> Result<Self, Error> {
        match mode {
            "argmax" => Ok(Self::Argmax),
            "infer" => Ok(Self::Infer),
            "infer_concurrent" => Ok(Self::InferConcurrent),
            "decoupled" => Ok(Self::Decoupled),
            "stream" => Ok(Self::Stream),
            other => Err(format!("unknown mode '{other}'").into()),
        }
    }
}

struct State {
    mode: Mode,
    target: String,
}

struct BenchBackend;

#[triton_ng::backend]
impl CustomBackend for BenchBackend {
    fn model_initialize(model: &Model) -> Result<(), Error> {
        model.set_state(State {
            mode: model.required_config_parameter("mode")?.parse()?,
            target: model.config_parameter_or("target", String::new())?,
        })?;
        Ok(())
    }

    async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
        let state = ctx.state::<State>()?;
        match state.mode {
            Mode::Argmax => argmax_of_images(requests),
            Mode::Infer => infer(&ctx, state, requests).await,
            Mode::InferConcurrent => infer_concurrent(&ctx, state, requests).await,
            Mode::Decoupled => {
                for request in requests {
                    decoupled(&ctx, state, request).await?;
                }
                Ok(())
            }
            Mode::Stream => {
                for request in requests {
                    stream(&ctx, state, request).await?;
                }
                Ok(())
            }
        }
    }
}

fn argmax_of_images(requests: Vec<Request>) -> Result<(), Error> {
    for request in requests {
        let image = request.get_input(IMAGE)?.as_fp32_vec()?;
        respond(request, argmax(&image)?)?;
    }
    Ok(())
}

async fn infer(ctx: &Context, state: &State, requests: Vec<Request>) -> Result<(), Error> {
    for request in requests {
        let image = request.get_input(IMAGE)?.as_bytes()?;
        let result = ctx.server().infer_async(classify(ctx, state, image)?).await;
        answer(request, result)?;
    }
    Ok(())
}

async fn infer_concurrent(
    ctx: &Context,
    state: &State,
    requests: Vec<Request>,
) -> Result<(), Error> {
    let mut inferences = Vec::with_capacity(requests.len());
    for request in &requests {
        let image = request.get_input(IMAGE)?.as_bytes()?;
        inferences.push(ctx.server().infer_async(classify(ctx, state, image)?));
    }
    for (request, result) in requests.into_iter().zip(join_all(inferences).await) {
        answer(request, result)?;
    }
    Ok(())
}

async fn decoupled(ctx: &Context, state: &State, request: Request) -> Result<(), Error> {
    let images = request.get_input(IMAGE)?.as_bytes()?;
    let mut sender = request.into_sender()?;
    for image in images.chunks(IMAGE_BYTES) {
        let inference = classify(ctx, state, image.to_vec())?;
        let digit = match ctx.server().infer_async(inference).await {
            Ok(result) => digit_of(&result)?,
            Err(e) => return Ok(sender.respond_error_with_code(e.code(), &e.to_string())?),
        };
        let mut response = sender.response()?;
        response
            .create_output(DIGIT, DataType::Int32, &[1, 1])?
            .write_i32_vec(&[digit])?;
        response.send()?;
    }
    Ok(sender.finish()?)
}

async fn stream(ctx: &Context, state: &State, request: Request) -> Result<(), Error> {
    let images = request.get_input(IMAGE)?.as_bytes()?;
    let count = i64::try_from(images.len() / IMAGE_BYTES)?;
    let mut inference = InferenceRequest::new(ctx.server(), &state.target, -1)?;
    inference.add_input_with_data(IMAGE, DataType::Fp32, &[count, 1, 28, 28], images)?;
    inference.add_requested_output(DIGIT)?;

    let mut stream = ctx.server().infer_stream(inference);
    let mut digits = Vec::new();
    while let Some(result) = stream.next().await {
        match result {
            Ok(result) => digits.extend(result.output(DIGIT)?.as_i32_vec()?),
            Err(e) => return Ok(request.respond_error_with_code(e.code(), &e.to_string())?),
        }
    }

    let mut response = Response::new(request)?;
    response
        .create_output(DIGIT, DataType::Int32, &[i64::try_from(digits.len())?])?
        .write_i32_vec(&digits)?;
    Ok(response.send()?)
}

fn classify(ctx: &Context, state: &State, image: Vec<u8>) -> Result<InferenceRequest, Error> {
    let mut inference = InferenceRequest::new(ctx.server(), &state.target, -1)?;
    inference.add_input_with_data(IMAGE, DataType::Fp32, &IMAGE_SHAPE, image)?;
    inference.add_requested_output(LOGITS)?;
    Ok(inference)
}

fn answer(request: Request, result: Result<InferenceResult, TritonError>) -> Result<(), Error> {
    match result {
        Ok(result) => respond(request, digit_of(&result)?),
        Err(e) => Ok(request.respond_error_with_code(e.code(), &e.to_string())?),
    }
}

fn digit_of(result: &InferenceResult) -> Result<i32, Error> {
    argmax(&result.output(LOGITS)?.as_fp32_vec()?)
}

fn argmax(values: &[f32]) -> Result<i32, Error> {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| i32::try_from(index))
        .ok_or("argmax of nothing")?
        .map_err(Into::into)
}

fn respond(request: Request, digit: i32) -> Result<(), Error> {
    let mut response = Response::new(request)?;
    response
        .create_output(DIGIT, DataType::Int32, &[1, 1])?
        .write_i32_vec(&[digit])?;
    Ok(response.send()?)
}
