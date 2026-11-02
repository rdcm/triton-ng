use crate::inference_future::Failure;
use crate::request_cancel::RequestCancel;
use crate::{InferenceResult, TritonError};
use futures_core::Stream;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

/// Results of one inference by a decoupled model, see
/// [`crate::Server::infer_stream`]: one per response, in the order the model
/// sent them, each an error when the response is one. The stream ends with
/// the final flag of the model.
///
/// The request runs whether the stream is polled or not, and its responses
/// wait in the stream until they are read. Dropping the stream cancels the
/// request, see [`InferenceStream::cancel`].
///
/// The stream holds no Triton pointers, it is `Send + 'static` and works
/// with any executor. Triton wakes it from its own threads.
///
/// # Examples
///
/// ```ignore
/// let mut stream = ctx.server().infer_stream(inference);
/// while let Some(result) = stream.next().await {
///     let digit = result?.output("DIGIT")?.as_i32_vec()?;
///     // ...
/// }
/// ```
pub struct InferenceStream {
    results: UnboundedReceiver<Result<InferenceResult, Failure>>,
    /// `None` when the request was never submitted.
    cancel: Option<Arc<RequestCancel>>,
}

/// The sending side of an [`InferenceStream`], kept by the response callback
/// until the final flag.
pub(crate) type StreamResults = UnboundedSender<Result<InferenceResult, Failure>>;

impl InferenceStream {
    pub(crate) fn channel() -> (
        StreamResults,
        UnboundedReceiver<Result<InferenceResult, Failure>>,
    ) {
        unbounded_channel()
    }

    pub(crate) fn submitted(
        results: UnboundedReceiver<Result<InferenceResult, Failure>>,
        cancel: Arc<RequestCancel>,
    ) -> Self {
        Self {
            results,
            cancel: Some(cancel),
        }
    }

    /// A stream of the single error `failure`, for a request that was not
    /// submitted.
    pub(crate) fn failed(failure: Failure) -> Self {
        let (sender, results) = unbounded_channel();
        // The receiver is alive, so the send cannot fail.
        let _ = sender.send(Err(failure));
        Self {
            results,
            cancel: None,
        }
    }

    /// The next result, `None` once the model sent the final flag.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// while let Some(result) = stream.next().await {
    ///     forward(&mut sender, result?)?;
    /// }
    /// ```
    pub async fn next(&mut self) -> Option<Result<InferenceResult, TritonError>> {
        std::future::poll_fn(|cx| self.poll_result(cx))
            .await
            .map(|result| result.map_err(TritonError::from))
    }

    /// Asks Triton to cancel the request, so that the model stops producing
    /// responses. The responses sent before still arrive, and how the stream
    /// ends is up to the model, usually with a CANCELLED error. Does nothing
    /// when the request is already done.
    ///
    /// Fails when Triton cannot cancel the request.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if sender.is_cancelled()? {
    ///     stream.cancel()?;
    ///     return Ok(sender.respond_cancelled()?);
    /// }
    /// ```
    pub fn cancel(&self) -> Result<(), TritonError> {
        match &self.cancel {
            Some(cancel) => cancel.cancel(),
            None => Ok(()),
        }
    }

    fn poll_result(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<InferenceResult, Failure>>> {
        self.results.poll_recv(cx)
    }
}

impl Stream for InferenceStream {
    type Item = Result<InferenceResult, TritonError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut()
            .poll_result(cx)
            .map(|result| result.map(|result| result.map_err(TritonError::from)))
    }
}

impl Drop for InferenceStream {
    fn drop(&mut self) {
        if let Err(e) = self.cancel() {
            crate::log::error(&format!("failed to cancel the BLS stream: {e}"));
        }
    }
}
