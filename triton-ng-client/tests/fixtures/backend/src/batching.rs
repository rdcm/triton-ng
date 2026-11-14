//! Batching integration-test fixture.

use super::*;

pub(super) fn echo_batch_keys(requests: Vec<Request>) -> Result<(), Error> {
    let mut keys = Vec::with_capacity(requests.len());
    for request in &requests {
        match request.parameter(BATCH_KEY)? {
            Some(Parameter::String(key)) => keys.push(key),
            _ => keys.push("none".to_string()),
        }
    }
    let keys = keys.join(",");
    let size = i64::try_from(requests.len()).map_err(|_| "the batch is too large")?;
    for request in requests {
        let mut response = Response::new(request)?;
        response
            .create_output("DIGIT", DataType::Int32, &[1, 1])?
            .write_i32_vec(&[0])?;
        response.set_parameter("batch_keys", Parameter::String(keys.clone()))?;
        response.set_parameter("batch_size", Parameter::Int(size))?;
        response.send()?;
    }
    Ok(())
}

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let state = ctx.state::<State>()?;
    if let Some(first) = requests.first()
        && first.parameter("echo_batch_keys")? == Some(Parameter::Bool(true))
    {
        return echo_batch_keys(requests);
    }
    let mut remaining = Vec::new();
    for request in requests {
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
