use crate::Output;
use crate::error::TritonError;
use crate::request::Request;
use crate::response::ResponseBody;
use crate::response_sender::ResponseSender;
use crate::types::{DataType, Parameter};
use std::marker::PhantomData;

/// A response of a stream that is not the final one, created by
/// [`ResponseSender::response`].
///
/// Borrows its sender mutably, so the stream cannot be finished while the
/// response is being built. A response dropped without being sent, or whose
/// sending fails, is deleted, and the stream goes on.
///
/// Owns the `TRITONBACKEND_Response` and deletes it on drop unless it was
/// sent. `Send`, not `Sync`.
///
/// # Examples
///
/// ```ignore
/// let mut response = sender.response()?;
/// response
///     .create_output("DIGIT", DataType::Int32, &[1])?
///     .write_i32_vec(&[digit])?;
/// response.set_parameter("index", index as i64)?;
/// response.send()?;
/// ```
pub struct StreamResponse<'a> {
    body: ResponseBody,
    _sender: PhantomData<&'a mut ResponseSender>,
}

impl StreamResponse<'_> {
    pub(crate) fn new(request: &Request) -> Result<Self, TritonError> {
        Ok(Self {
            body: ResponseBody::new(request)?,
            _sender: PhantomData,
        })
    }

    /// Creates an output tensor. It must be written exactly once before
    /// [`StreamResponse::send`].
    pub fn create_output(
        &mut self,
        name: &str,
        datatype: DataType,
        shape: &[i64],
    ) -> Result<Output<'_>, TritonError> {
        self.body.create_output(name, datatype, shape)
    }

    /// Adds the parameter `name` to the response, see
    /// [`crate::Response::set_parameter`].
    pub fn set_parameter(
        &mut self,
        name: &str,
        value: impl Into<Parameter>,
    ) -> Result<(), TritonError> {
        self.body.set_parameter(name, value)
    }

    /// Sends the response without the final flag.
    pub fn send(mut self) -> Result<(), TritonError> {
        self.body.send(0)
    }
}
