use super::*;

pub(super) fn check_specific_policy(request: &Request) -> Result<(), Error> {
    let input = request.get_input(IMAGE)?;
    let properties = input.properties_for_host_policy("pinned")?;
    let bytes = input.as_bytes_for_host_policy("pinned")?;
    let expected = 1.25_f32.to_ne_bytes().repeat(784);
    let first = input.buffer_for_host_policy("pinned", 0)?;
    let second = input.buffer_for_host_policy("pinned", 1)?;
    if properties.buffer_count != 2 || bytes != expected || [first, second].concat() != expected {
        return Err("the policy-specific input did not preserve its two buffers".into());
    }
    if bytes == input.as_bytes()?
        || input.as_bytes_for_host_policy("missing")? != input.as_bytes()?
    {
        return Err("the policy-specific input did not stay separate from default storage".into());
    }
    Ok(())
}

pub(super) fn echo_policy_input(ctx: &Context, request: Request) -> Result<(), Error> {
    let instance = ctx.instance_state::<Buffers>()?;
    let declared = instance
        .host_policy
        .as_ref()
        .ok_or("the instance has no host policy")?;
    let mut policy = match request.parameter("input_policy")? {
        Some(Parameter::String(name)) => name,
        _ => declared.name().to_owned(),
    };
    let input = request.get_input(IMAGE)?;
    // Construct NUL in Rust: Triton's NUL-terminated C parameter API cannot
    // preserve an embedded NUL sent through gRPC.
    if request.parameter("invalid_policy_name")? == Some(Parameter::Bool(true)) {
        policy.push('\0');
        policy.push_str("suffix");
        let results = [
            input.properties_for_host_policy(&policy).map(|_| ()),
            input.buffer_for_host_policy(&policy, 0).map(|_| ()),
            input.as_bytes_for_host_policy(&policy).map(|_| ()),
        ];
        let message = if results.iter().any(Result::is_ok) {
            "an invalid policy name was accepted".to_string()
        } else {
            results
                .into_iter()
                .filter_map(Result::err)
                .map(|error| error.to_string())
                .collect::<Vec<_>>()
                .join("; ")
        };
        request.respond_error(&message)?;
        return Ok(());
    }
    let properties = input.properties_for_host_policy(&policy)?;
    if request.parameter("invalid_policy_index")? == Some(Parameter::Bool(true)) {
        let message = match input.buffer_for_host_policy(&policy, properties.buffer_count) {
            Ok(_) => "an out-of-range buffer was accepted".to_string(),
            Err(error) => error.to_string(),
        };
        request.respond_error(&message)?;
        return Ok(());
    }
    let bytes = input.as_bytes_for_host_policy(&policy)?;
    let mut pieces = Vec::new();
    for index in 0..properties.buffer_count {
        pieces.extend(input.buffer_for_host_policy(&policy, index)?);
    }
    if pieces != bytes || bytes != input.as_bytes()? {
        return Err("policy input copies do not match the original client input".into());
    }
    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[0])?;
    response.set_parameter("input_policy", policy.as_str())?;
    response.set_parameter("input_name", properties.name.as_str())?;
    response.set_parameter("input_datatype", format!("{:?}", properties.datatype))?;
    response.set_parameter("input_shape", format!("{:?}", properties.shape))?;
    response.set_parameter(
        "input_byte_size",
        Parameter::Int(i64::try_from(properties.byte_size)?),
    )?;
    response.set_parameter("policy_input_matches", Parameter::Bool(true))?;
    response.send()?;
    Ok(())
}

pub(super) fn echo_host_policy(ctx: &Context, request: Request) -> Result<(), Error> {
    let policy = ctx
        .instance_state::<Buffers>()?
        .host_policy
        .as_ref()
        .ok_or("the instance hook stored no host policy")?;
    let settings = policy
        .settings()
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(",");
    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[0])?;
    response.set_parameter("host_policy_name", policy.name())?;
    response.set_parameter("host_policy_settings", settings)?;
    response.send()?;
    Ok(())
}

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let mut remaining = Vec::new();
    for request in requests {
        if request.parameter("check_specific_policy")? == Some(Parameter::Bool(true)) {
            let checked = check_specific_policy(&request);
            match checked {
                Ok(()) => answer_done(request)?,
                Err(error) => request.respond_error(&error.to_string())?,
            }
            continue;
        }
        if request.parameter("echo_policy_input")? == Some(Parameter::Bool(true)) {
            echo_policy_input(ctx, request)?;
            continue;
        }
        if request.parameter("echo_host_policy")? == Some(Parameter::Bool(true)) {
            echo_host_policy(ctx, request)?;
            continue;
        }
        remaining.push(request);
    }
    inference::execute(ctx, remaining).await?;
    Ok(())
}
