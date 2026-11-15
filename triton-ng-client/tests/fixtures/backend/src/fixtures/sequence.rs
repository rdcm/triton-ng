use super::*;

pub(super) fn count_sequence(state: &State, requests: Vec<Request>) -> Result<(), Error> {
    for mut request in requests {
        if request.parameter("bad_state_shape")? == Some(Parameter::Bool(true)) {
            let counter = state
                .states
                .iter()
                .find(|declared| declared.name() == COUNTER_OUT)
                .ok_or("the model configuration declares no counter_out state")?;
            match triton_ng::State::new(&mut request, counter, &[2]) {
                Ok(_) => request.respond_error("a state of the wrong shape was created")?,
                Err(error) => request.respond_error(&error.to_string())?,
            }
            continue;
        }
        if request.parameter("missing_state")? == Some(Parameter::Bool(true)) {
            let name = "no_such_state";
            match state.states.iter().find(|declared| declared.name() == name) {
                Some(_) => request.respond_error("the configuration declares no_such_state")?,
                None => request.respond_error(&format!(
                    "the model configuration declares no state '{name}'"
                ))?,
            }
            continue;
        }

        if let Some(Parameter::String(model)) = request.parameter("foreign_state_model")? {
            // The configuration of another model: the datatype and the dimensions it
            // declares are not the ones this model keeps the state with, so the
            // library refuses it before Triton changes the metadata of the state.
            let foreign = DECLARED_STATES
                .lock()
                .map_err(|_| "the declared states are poisoned")?
                .get(&model)
                .cloned()
                .ok_or_else(|| format!("the model '{model}' declared no state"))?;
            let counter = foreign
                .iter()
                .find(|declared| declared.name() == COUNTER_OUT)
                .ok_or("the other model declares no counter_out state")?;
            let message = match triton_ng::State::new(&mut request, counter, &[2]) {
                Ok(_) => "a state of another model was created".to_string(),
                Err(error) => error.to_string(),
            };
            request.respond_error(&message)?;
            continue;
        }
        if request.parameter("shared_state_input")? == Some(Parameter::Bool(true)) {
            // The input the state is paired with, handed to another request before
            // the state is created: writing the state may write the bytes the
            // shared input reads.
            let shared = request.get_input(COUNTER_IN)?.share()?;
            let counter = state
                .states
                .iter()
                .find(|declared| declared.name() == COUNTER_OUT)
                .ok_or("the model configuration declares no counter_out state")?;
            let message = match triton_ng::State::new(&mut request, counter, &[1]) {
                Ok(_) => "a state of a shared input was created".to_string(),
                Err(error) => error.to_string(),
            };
            // The final response is held back until the last shared input is
            // dropped, so the shared input goes before the error is answered.
            drop(shared);
            request.respond_error(&message)?;
            continue;
        }

        if request.parameter("reallocate_state_buffer")? == Some(Parameter::Bool(true)) {
            let counter = state
                .states
                .iter()
                .find(|entry| entry.name() == COUNTER_OUT)
                .ok_or("the model configuration declares no counter_out state")?;
            let mut output = triton_ng::State::new(&mut request, counter, &[1])?;
            output.write_bytes(&1.0_f32.to_ne_bytes())?;
            // Switching from host memory replaces the allocation, including when
            // GPU allocation falls back to host memory and device_buffer fails.
            let _allocation = output.device_buffer(MemoryType::Gpu { device_id: 0 });
            let message = match output.update() {
                Ok(()) => "an unwritten replacement buffer was published".to_string(),
                Err(error) => error.to_string(),
            };
            request.respond_error(&message)?;
            continue;
        }

        let counted = count_one(&mut request, &state.states);
        match counted {
            Ok(next) => {
                let request_batched =
                    request.parameter("check_batched_state")? == Some(Parameter::Bool(true));
                let mut response = Response::new(request)?;
                response
                    .create_output(
                        COUNTER_OUTPUT,
                        DataType::Fp32,
                        if request_batched { &[1, 1] } else { &[1] },
                    )?
                    .write_fp32_vec(&[next])?;
                response.send()?;
            }
            Err(error) => request.respond_error(&error.to_string())?,
        }
    }
    Ok(())
}

pub(super) fn count_one(
    request: &mut Request,
    states: &[triton_ng::StateConfig],
) -> Result<f32, Error> {
    let counter = states
        .iter()
        .find(|declared| declared.name() == COUNTER_OUT)
        .ok_or("the model configuration declares no counter_out state")?;
    let input = request.get_input(COUNTER_IN)?;
    let input_shape = input.properties()?.shape;
    let expected_shape: &[i64] =
        if request.parameter("check_batched_state")? == Some(Parameter::Bool(true)) {
            &[1, 1]
        } else {
            &[1]
        };
    if input_shape != expected_shape {
        return Err(format!(
            "unexpected state input shape: {input_shape:?}, expected {expected_shape:?}"
        )
        .into());
    }
    let previous = input.as_fp32_vec()?;
    let previous = previous.first().ok_or("the counter state is empty")?;
    let next = previous + 1.0;

    let mut state = triton_ng::State::new(request, counter, &[1])?;
    // Triton allocates the buffer of a state on demand, so the attributes are
    // read after the write that asks for it. Its allocator hands out pinned
    // host memory where it has room, so the buffer is host memory either way.
    state.write_bytes(&next.to_ne_bytes())?;
    let attributes = state.buffer_attributes()?;
    if attributes.byte_size != 4 || matches!(attributes.memory_type, MemoryType::Gpu { .. }) {
        return Err(format!(
            "the state buffer is {:?} with {} bytes",
            attributes.memory_type, attributes.byte_size
        )
        .into());
    }
    state.update()?;
    Ok(next)
}
