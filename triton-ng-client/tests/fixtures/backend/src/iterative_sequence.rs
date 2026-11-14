//! Iterative sequence integration-test fixture.

use super::*;

pub(super) fn stream_steps(state: &State, requests: Vec<Request>) -> Result<(), Error> {
    for request in requests {
        let Some(id) = request.correlation_id()? else {
            continue;
        };
        let steps = match request.parameter("steps")? {
            Some(Parameter::Int(steps)) => steps,
            _ => 2,
        };
        let step = {
            let mut done = state.steps.lock().map_err(|_| "steps are poisoned")?;
            let step = done.entry(id.clone()).or_insert(0);
            *step += 1;
            *step
        };

        let mut sender = request.into_sender()?;
        let mut response = sender.response()?;
        response
            .create_output("DIGIT", DataType::Int32, &[1, 1])?
            .write_i32_vec(&[step])?;
        response.send()?;

        let finished = i64::from(step) >= steps;
        let result = if finished {
            sender.finish()
        } else {
            sender.reschedule()
        };
        if finished || result.is_err() {
            state
                .steps
                .lock()
                .map_err(|_| "steps are poisoned")?
                .remove(&id);
        }
        if let Err(e) = result {
            println!("step {step} of {id:?} failed: {e}");
        }
    }
    Ok(())
}
