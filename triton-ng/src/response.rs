use crate::Output;
use crate::error::TritonError;
use crate::final_response::Pending;
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
    /// Dropped before the request, see `Drop` of `ResponseBody`: an unsent
    /// response must be deleted before the request gets its error response.
    body: ResponseBody,
    request: Request,
}

impl Response {
    /// Creates the response of `request`. On error the request is dropped and
    /// gets an error response.
    pub fn new(request: Request) -> Result<Self, TritonError> {
        Ok(Self {
            body: ResponseBody::new(&request)?,
            request,
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
        self.body.create_output(name, datatype, shape)
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
        self.body.set_parameter(name, value)
    }

    /// Sends the response as the final one for its request. While a
    /// [`crate::SharedInput`] of the request is alive, the response is sent
    /// when the last one is dropped, and a failure to send it then is only
    /// logged.
    ///
    /// Fails without sending while an output is not written; the request
    /// gets an error response then.
    pub fn send(mut self) -> Result<(), TritonError> {
        let statistics = self.body.take_statistics();
        let response = self.body.take()?;
        self.request
            .send_final(Pending::Response(response, statistics))
    }

    /// Asks Triton to record the time this response took, which
    /// `/v2/models/<model>/stats` shows as `response_stats`.
    ///
    /// The clock starts when [`Response::new`] created the response, so the
    /// work that fills the outputs is inside it even when this is called just
    /// before [`Response::send`]. Without this call Triton records nothing for
    /// the response.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut response = Response::new(request)?;
    /// response
    ///     .create_output("DIGIT", DataType::Int32, &[1])?
    ///     .write_i32_vec(&[digit])?;
    /// response.enable_statistics();
    /// response.send()?;
    /// ```
    pub fn enable_statistics(&mut self) {
        self.body.enable_statistics();
    }
}

/// `TRITONBACKEND_Response` being built: its outputs and parameters, shared
/// by [`Response`] and [`crate::StreamResponse`].
///
/// Owns the response and deletes it on drop unless it was sent. `Send`, not
/// `Sync`.
pub(crate) struct ResponseBody {
    /// Null once the response was handed over to `TRITONBACKEND_ResponseSend`,
    /// which takes ownership whatever it returns.
    ptr: *mut triton_ng_sys::TRITONBACKEND_Response,
    /// Outputs created but not written yet. Triton frontends read every
    /// output of a sent response, so one without a buffer crashes the server.
    pending_outputs: usize,
    /// What Triton needs to time this response, when the backend enables it.
    statistics: ResponseStatistics,
}

// SAFETY: see `Request`. Outputs are created through `&mut self`, so they are
// never added from several threads at once, as tritonbackend.h requires.
unsafe impl Send for ResponseBody {}

impl ResponseBody {
    pub(crate) fn new(request: &Request) -> Result<Self, TritonError> {
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseNew(
            &mut _,
            request.as_ptr()
        ))?;
        Ok(Self {
            ptr,
            pending_outputs: 0,
            statistics: ResponseStatistics::new(request),
        })
    }

    /// Enables the per-response statistics of Triton for this response.
    pub(crate) fn enable_statistics(&mut self) {
        self.statistics.enabled = true;
    }

    /// The statistics to report with the final send, when they are enabled.
    pub(crate) fn take_statistics(&mut self) -> Option<ResponseStatistics> {
        if !self.statistics.enabled {
            return None;
        }
        self.statistics.enabled = false;
        Some(ResponseStatistics {
            request: self.statistics.request,
            instance: self.statistics.instance,
            start_ns: self.statistics.start_ns,
            output_start_ns: self.statistics.output_start_ns,
            enabled: false,
        })
    }

    pub(crate) fn create_output(
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
        self.statistics.output_started();
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

    pub(crate) fn set_parameter(
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

    /// Sends the response with `flags`, see [`ResponseBody::take`].
    pub(crate) fn send(&mut self, flags: u32) -> Result<(), TritonError> {
        let statistics = self.take_statistics();
        let response = self.take()?;

        // ResponseSend takes ownership of the response whatever it returns.
        let sent = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSend(
            response,
            flags,
            ptr::null_mut(),
        ));
        if let Some(statistics) = statistics
            && let Err(e) = statistics.report()
        {
            crate::log::error(&format!("failed to report response statistics: {e}"));
        }
        sent
    }

    /// Hands the response over to be sent: the caller owns it then and must
    /// pass it to `TRITONBACKEND_ResponseSend`. Fails while an output is not
    /// written; the response is deleted on drop then.
    pub(crate) fn take(
        &mut self,
    ) -> Result<*mut triton_ng_sys::TRITONBACKEND_Response, TritonError> {
        if self.pending_outputs > 0 {
            return Err(TritonError::from_message(format!(
                "{} output(s) of the response were created but not written",
                self.pending_outputs
            )));
        }
        if self.ptr.is_null() {
            return Err(TritonError::from_message("the response was already sent"));
        }
        Ok(std::mem::replace(&mut self.ptr, ptr::null_mut()))
    }
}

impl Drop for ResponseBody {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                triton_ng_sys::TRITONBACKEND_ResponseDelete(self.ptr);
            }
        }
    }
}

pub(crate) const FINAL: u32 =
    triton_ng_sys::tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL;

/// What Triton needs to record the time of one response, which is what
/// `/v2/models/<model>/stats` shows as `response_stats`.
///
/// Created with the response, so the start is the moment the backend began
/// building it; reported only when the backend enabled it, and only once.
#[derive(Debug)]
pub(crate) struct ResponseStatistics {
    /// The request the response answers, kept alive by the response.
    request: *mut triton_ng_sys::TRITONBACKEND_Request,
    instance: *mut triton_ng_sys::TRITONBACKEND_ModelInstance,
    /// Nanoseconds of the clock Triton stamps its statistics with.
    start_ns: u64,
    /// When the backend began to fill the outputs, which separates the time it
    /// spent computing from the time it spent writing. `None` when it created
    /// no output at all, which Triton counts as an empty response.
    output_start_ns: Option<u64>,
    enabled: bool,
}

// SAFETY: the pointers belong to the request and the instance that the
// response is built for, both of which outlive it.
unsafe impl Send for ResponseStatistics {}

impl ResponseStatistics {
    fn new(request: &Request) -> Self {
        Self {
            request: request.as_ptr(),
            instance: request.statistics().instance(),
            start_ns: crate::batch_statistics::now_ns(),
            output_start_ns: None,
            enabled: false,
        }
    }

    /// Records the moment the backend created its first output, once.
    fn output_started(&mut self) {
        if self.output_start_ns.is_none() {
            self.output_start_ns = Some(crate::batch_statistics::now_ns());
        }
    }

    /// Reports the response to Triton, with the factory of its request, which
    /// the report dereferences to number the response.
    pub(crate) fn report(&self) -> Result<(), TritonError> {
        let factory = crate::response_factory::ResponseFactory::new(self.request)?;
        let statistics =
            triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceResponseStatisticsNew(&mut _))?;
        if statistics.is_null() {
            return Err(TritonError::from_message(
                "TRITONBACKEND_ModelInstanceResponseStatisticsNew returned null",
            ));
        }

        let reported = (|| {
            triton_call!(
                triton_ng_sys::TRITONBACKEND_ModelInstanceResponseStatisticsSetModelInstance(
                    statistics,
                    self.instance
                )
            )?;
            triton_call!(
                triton_ng_sys::TRITONBACKEND_ModelInstanceResponseStatisticsSetResponseFactory(
                    statistics,
                    factory.as_ptr()
                )
            )?;
            triton_call!(
                triton_ng_sys::TRITONBACKEND_ModelInstanceResponseStatisticsSetResponseStart(
                    statistics,
                    self.start_ns
                )
            )?;
            if let Some(output_start_ns) = self.output_start_ns {
                triton_call!(
                    triton_ng_sys::TRITONBACKEND_ModelInstanceResponseStatisticsSetComputeOutputStart(
                        statistics,
                        output_start_ns
                    )
                )?;
            }
            triton_call!(
                triton_ng_sys::TRITONBACKEND_ModelInstanceResponseStatisticsSetResponseEnd(
                    statistics,
                    crate::batch_statistics::now_ns()
                )
            )?;
            triton_call!(
                triton_ng_sys::TRITONBACKEND_ModelInstanceReportResponseStatistics(statistics)
            )
        })();

        if let Err(e) = triton_call!(
            triton_ng_sys::TRITONBACKEND_ModelInstanceResponseStatisticsDelete(statistics)
        ) {
            crate::log::error(&format!("failed to delete response statistics: {e}"));
        }
        reported
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
