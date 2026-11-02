use crate::error::{ErrorCode, TritonError};
use crate::final_response::Pending;
use crate::request::Request;
use crate::response::Response;
use crate::stream_response::StreamResponse;

/// Sends any number of responses to a request of a decoupled model
/// (`model_transaction_policy { decoupled: true }`), then exactly one final
/// flag, see [`Request::into_sender`].
///
/// Clients get the responses as a stream (gRPC `ModelStreamInfer`). The
/// stream ends with
/// [`ResponseSender::finish`], [`ResponseSender::final_response`] or one of
/// the `respond_*` methods, all of which consume the sender, so nothing can
/// be sent after the final flag. A sender dropped without it sends a final
/// error response.
///
/// Owns its [`Request`] and releases it when it is dropped, so the inputs
/// stay readable for the whole stream. `Send`, not `Sync`: move it into the
/// task that produces the responses.
///
/// # Examples
///
/// ```ignore
/// let mut sender = request.into_sender()?;
/// for token in tokens {
///     if sender.is_cancelled()? {
///         return Ok(sender.respond_cancelled()?);
///     }
///     let mut response = sender.response()?;
///     response
///         .create_output("TOKEN", DataType::Int64, &[1])?
///         .write_i64_vec(&[token])?;
///     response.send()?;
/// }
/// sender.finish()?;
/// ```
pub struct ResponseSender {
    request: Request,
}

impl ResponseSender {
    pub(crate) fn new(request: Request) -> Self {
        Self { request }
    }

    /// The request the responses answer, e.g. to read its inputs or to make
    /// BLS calls traced as its children.
    pub fn request(&self) -> &Request {
        &self.request
    }

    /// Returns `true` if the client cancelled the request, see
    /// [`ResponseSender::respond_cancelled`].
    pub fn is_cancelled(&self) -> Result<bool, TritonError> {
        self.request.is_cancelled()
    }

    /// Creates the next, not final, response of the stream. It borrows the
    /// sender, so the stream cannot be finished while it is being built.
    pub fn response(&mut self) -> Result<StreamResponse<'_>, TritonError> {
        StreamResponse::new(&self.request)
    }

    /// Creates the last response of the stream, sent with the final flag by
    /// [`Response::send`]. On error the request gets an error response.
    pub fn final_response(self) -> Result<Response, TritonError> {
        Response::new(self.request)
    }

    /// Ends the stream without one more response. On error the request gets
    /// an error response. While a [`crate::SharedInput`] of the request is
    /// alive, the stream ends when the last one is dropped.
    pub fn finish(mut self) -> Result<(), TritonError> {
        self.request.send_final(Pending::Flags)
    }

    /// Ends the stream with `message` as an internal error, see
    /// [`Request::respond_error`].
    pub fn respond_error(self, message: &str) -> Result<(), TritonError> {
        self.request.respond_error(message)
    }

    /// Ends the stream with `message` as an error with `code`, see
    /// [`Request::respond_error_with_code`].
    pub fn respond_error_with_code(
        self,
        code: ErrorCode,
        message: &str,
    ) -> Result<(), TritonError> {
        self.request.respond_error_with_code(code, message)
    }

    /// Hands the request back to Triton without ending the stream, to send
    /// its next responses in a later batch, see [`Request::reschedule`]. The
    /// client keeps getting the responses of every step as one stream.
    ///
    /// Fails, and ends the stream with the error, when the model has no
    /// `iterative_sequence`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut sender = request.into_sender()?;
    /// let mut response = sender.response()?;
    /// response.create_output("TOKEN", DataType::Int64, &[1])?.write_i64_vec(&[token])?;
    /// response.send()?;
    /// if more_tokens {
    ///     sender.reschedule()?;
    /// } else {
    ///     sender.finish()?;
    /// }
    /// ```
    pub fn reschedule(self) -> Result<(), TritonError> {
        self.request.reschedule()
    }

    /// Ends the stream with the CANCELLED status, see
    /// [`ResponseSender::is_cancelled`].
    pub fn respond_cancelled(self) -> Result<(), TritonError> {
        self.request.respond_cancelled()
    }
}
