use crate::Output;
use crate::error::TritonError;
use crate::request::Request;
use crate::types::{DataType, Parameter};
use crate::utils::cstring_from_str;
use std::ptr;
use triton_ng_macros::triton_call;

/// Final response of a [`Request`].
///
/// The response takes the request over and keeps it until it is sent, so the
/// request gets no other response. A response dropped without being sent, or
/// whose sending fails, is deleted and its request gets an error response.
///
/// Owns the `TRITONBACKEND_Response` and its request, and deletes the
/// response on drop unless it was sent. `Send`, not `Sync`.
///
/// # Examples
///
/// ```ignore
/// let scores = request.get_input("SCORES")?.as_fp32_vec()?;
///
/// let mut response = Response::new(request)?;
/// response
///     .create_output("SCORES", DataType::Fp32, &[1, scores.len() as i64])?
///     .write_fp32_vec(&scores)?;
/// response.send()?;
/// ```
pub struct Response {
    /// Null once the response was handed over to `TRITONBACKEND_ResponseSend`,
    /// which takes ownership whatever it returns.
    ptr: *mut triton_ng_sys::TRITONBACKEND_Response,
    /// Dropped after the response, see `Drop`: an unsent response must be
    /// deleted before the request gets its error response.
    request: Request,
    /// Outputs created but not written yet. Triton frontends read every
    /// output of a sent response, so one without a buffer crashes the server.
    pending_outputs: usize,
}

// SAFETY: see `Request`. Outputs are created through `&mut self`, so they are
// never added from several threads at once, as tritonbackend.h requires.
unsafe impl Send for Response {}

impl Response {
    /// Creates the response of `request`. On error the request is dropped and
    /// gets an error response.
    pub fn new(request: Request) -> Result<Self, TritonError> {
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseNew(
            &mut _,
            request.as_ptr()
        ))?;
        Ok(Self {
            ptr,
            request,
            pending_outputs: 0,
        })
    }

    /// The request this response answers, e.g. to read its inputs.
    pub fn request(&self) -> &Request {
        &self.request
    }

    /// Creates an output tensor. It must be written exactly once before
    /// [`Response::send`].
    pub fn create_output(
        &mut self,
        name: &str,
        datatype: DataType,
        shape: &[i64],
    ) -> Result<Output<'_>, TritonError> {
        if datatype == DataType::Invalid {
            return Err(TritonError::from_message(format!(
                "output '{name}' has invalid datatype"
            )));
        }
        let elements = element_count(shape)
            .map_err(|e| TritonError::from_message(format!("output '{name}': {e}")))?;

        let name_cstr = cstring_from_str(name)?;
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseOutput(
            self.ptr,
            &mut _,
            name_cstr.as_ptr(),
            datatype.to_sys(),
            shape.as_ptr(),
            shape.len() as u32,
        ))?;

        self.pending_outputs += 1;

        Ok(Output::new(
            ptr,
            name.to_string(),
            datatype,
            elements,
            &mut self.pending_outputs,
        ))
    }

    /// Adds the parameter `name` to the response. Clients read it from
    /// `parameters` of the gRPC `ModelInferResponse`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut response = Response::new(request)?;
    /// response.set_parameter("tokens", 42)?;
    /// response.set_parameter("language", "ru")?;
    /// response.send()?;
    /// ```
    pub fn set_parameter(
        &mut self,
        name: &str,
        value: impl Into<Parameter>,
    ) -> Result<(), TritonError> {
        let name_cstr = cstring_from_str(name)?;
        match value.into() {
            Parameter::Bool(value) => {
                triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSetBoolParameter(
                    self.ptr,
                    name_cstr.as_ptr(),
                    value
                ))
            }
            Parameter::Int(value) => {
                triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSetIntParameter(
                    self.ptr,
                    name_cstr.as_ptr(),
                    value
                ))
            }
            Parameter::Double(value) => {
                triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSetDoubleParameter(
                    self.ptr,
                    name_cstr.as_ptr(),
                    value
                ))
            }
            Parameter::String(value) => {
                let value_cstr = cstring_from_str(&value)?;
                triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSetStringParameter(
                    self.ptr,
                    name_cstr.as_ptr(),
                    value_cstr.as_ptr()
                ))
            }
        }
    }

    /// Sends the response as the final one for its request.
    pub fn send(mut self) -> Result<(), TritonError> {
        if self.pending_outputs > 0 {
            return Err(TritonError::from_message(format!(
                "{} output(s) of the response were created but not written",
                self.pending_outputs
            )));
        }

        let response = std::mem::replace(&mut self.ptr, ptr::null_mut());

        triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSend(
            response,
            triton_ng_sys::tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            ptr::null_mut(),
        ))?;

        self.request.mark_responded();

        Ok(())
    }
}

impl Drop for Response {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                triton_ng_sys::TRITONBACKEND_ResponseDelete(self.ptr);
            }
        }
    }
}

/// Number of elements of a tensor with `shape`.
pub(crate) fn element_count(shape: &[i64]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |count, &dim| {
        let dim = usize::try_from(dim)
            .map_err(|_| format!("invalid dimension {dim} in shape {shape:?}"))?;
        count
            .checked_mul(dim)
            .ok_or_else(|| format!("shape {shape:?} is too large"))
    })
}

#[cfg(test)]
mod tests {
    use super::element_count;

    #[test]
    fn element_count_multiplies_dimensions() {
        assert_eq!(element_count(&[2, 3, 4]), Ok(24));
        assert_eq!(element_count(&[1, 0, 2]), Ok(0));
        assert_eq!(element_count(&[]), Ok(1));
    }

    #[test]
    fn element_count_rejects_negative_dimensions() {
        assert!(element_count(&[1, -1]).is_err());
    }

    #[test]
    fn element_count_rejects_overflow() {
        assert!(element_count(&[i64::MAX, i64::MAX]).is_err());
    }
}
