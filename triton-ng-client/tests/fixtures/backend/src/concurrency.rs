//! Concurrency integration-test fixture.

use super::*;

pub(super) fn echo_concurrency(state: &State, request: Request) -> Result<(), Error> {
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

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let state = ctx.state::<State>()?;
    let mut remaining = Vec::new();
    for request in requests {
        if request.parameter("echo_concurrency")? == Some(Parameter::Bool(true)) {
            echo_concurrency(state, request)?;
            continue;
        }
        if request.parameter("echo_backend_config")? == Some(Parameter::Bool(true)) {
            echo_backend_config(ctx, request)?;
            continue;
        }
        if let Some(Parameter::Int(micros)) = request.parameter("hold_micros")? {
            let micros = u64::try_from(micros).map_err(|_| "hold_micros is negative")?;
            // Holds the execution thread of the instance so that a test can
            // observe how many of them run at once, or fill the queue of a
            // single instance and see which batches Triton forms from it.
            state.metrics.held.increment(1.0)?;
            std::thread::park_timeout(Duration::from_micros(micros));
            state.metrics.held.decrement(1.0)?;
        }
        remaining.push(request);
    }
    inference::execute(ctx, remaining).await?;
    Ok(())
}
