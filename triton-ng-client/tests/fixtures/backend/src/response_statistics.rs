//! Response statistics integration-test fixture.

use super::*;

pub(super) async fn execute(ctx: &Context, requests: Vec<Request>) -> Result<(), Error> {
    let mut remaining = Vec::new();
    for mut request in requests {
        if let Some(Parameter::String(code)) = request.parameter("error_code")? {
            if request.parameter("enable_statistics")? == Some(Parameter::Bool(true)) {
                request.enable_statistics();
            }
            match code.parse::<ErrorCode>() {
                Ok(code) => {
                    request.respond_error_with_code(code, "error requested by the client")?
                }
                Err(e) => request.respond_error(&e)?,
            }
            continue;
        }
        if request.parameter("enable_statistics")? == Some(Parameter::Bool(true)) {
            let hold = match request.parameter("hold_micros")? {
                Some(Parameter::Int(micros)) => {
                    let micros = u64::try_from(micros).map_err(|_| "hold_micros is negative")?;
                    Some(Duration::from_micros(micros))
                }
                _ => None,
            };
            let mut response = Response::new(request)?;
            response
                .create_output("DIGIT", DataType::Int32, &[1, 1])?
                .write_i32_vec(&[0])?;
            // The interval starts when the response was created, so work done
            // before the opt-in is inside it.
            if let Some(hold) = hold {
                std::thread::park_timeout(hold);
            }
            // Enabling twice must neither move the start nor report twice.
            response.enable_statistics();
            response.enable_statistics();
            response.send()?;
            continue;
        }
        remaining.push(request);
    }
    inference::execute(ctx, remaining).await?;
    Ok(())
}
