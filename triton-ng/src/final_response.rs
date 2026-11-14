use crate::error::{ErrorCode, TritonError};
use crate::request_release::RequestRelease;
use crate::response::FINAL;
use crate::response_factory::ResponseFactory;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use triton_ng_macros::triton_call;

/// Sends the final response of a `TRITONBACKEND_Request`, held back while a
/// [`crate::SharedInput`] of the request is alive.
///
/// The data of a request input lives until the request gets its final
/// response and is released, whichever is later: the gRPC frontend reuses
/// the protobuf the inputs point into once the final response is written
/// (r25.09 `grpc/infer_handler.h`, `State::Reset` clears `request_`), and an
/// ensemble frees the memory of a step input when the step request is
/// deleted. So a request with a shared input in flight gets its final
/// response from the last shared input dropped, before the request is
/// released.
///
/// Holds the `TRITONBACKEND_Request` a third time, after [`crate::Request`]
/// and [`RequestRelease`], and keeps the request from being released until
/// the final response is sent. `Send + Sync`.
pub(crate) struct FinalResponse {
    request: *mut triton_ng_sys::TRITONBACKEND_Request,
    gate: Mutex<Gate<Pending>>,
    release: Arc<RequestRelease>,
    /// Whether the backend asked for the statistics of the responses of this
    /// request, see `Request::enable_statistics`.
    statistics: AtomicBool,
}

// SAFETY: Triton lets a request be answered from any thread, one thread at a
// time (tritonbackend.h, TRITONBACKEND_Response). The response is sent once,
// by the thread that takes it out of the mutex.
unsafe impl Send for FinalResponse {}
unsafe impl Sync for FinalResponse {}

/// A final response waiting for the last shared input.
pub(crate) enum Pending {
    /// A response ready to be sent, owned until it is handed to
    /// `TRITONBACKEND_ResponseSend`, with the statistics the backend enabled
    /// for it.
    Response(
        *mut triton_ng_sys::TRITONBACKEND_Response,
        Option<crate::response::ResponseStatistics>,
    ),
    /// An error response, created when it is sent.
    Error(ErrorCode, String),
    /// The final flag of a stream without one more response.
    Flags,
}

// SAFETY: see `FinalResponse`; a pending response is used by one thread at a
// time, behind the mutex.
unsafe impl Send for Pending {}

impl FinalResponse {
    pub(crate) fn new(
        request: *mut triton_ng_sys::TRITONBACKEND_Request,
        release: Arc<RequestRelease>,
    ) -> Self {
        Self {
            request,
            gate: Mutex::new(Gate::default()),
            release,
            statistics: AtomicBool::new(false),
        }
    }

    /// Records that the backend asked for the statistics of the error response
    /// Triton sends for this request, see `Request::enable_statistics`.
    pub(crate) fn enable_statistics(&self) {
        self.statistics.store(true, Ordering::Relaxed);
    }

    /// Sends `pending` now, or keeps it for the last shared input. A failure
    /// to send it now is returned, one after it was kept is only logged.
    pub(crate) fn send(&self, pending: Pending) -> Result<(), TritonError> {
        match self.lock().send(pending) {
            Some(pending) => self.send_now(pending),
            None => Ok(()),
        }
    }

    /// Counts one more shared input named `name`. Fails once the final response
    /// is given.
    pub(crate) fn share(&self, name: &str) -> Result<(), TritonError> {
        self.lock().share(name).map_err(TritonError::from_message)
    }

    /// Counts the shared input named `name` dropped, and sends the final
    /// response kept for the last one.
    pub(crate) fn unshare(&self, name: &str) {
        let Some(pending) = self.lock().unshare(name) else {
            return;
        };
        if let Err(e) = self.send_now(pending) {
            crate::log::error(&format!(
                "failed to send the final response held for a shared input: {e}"
            ));
        }
    }

    /// Whether the input named `name` is shared and not dropped yet.
    pub(crate) fn is_input_shared(&self, name: &str) -> bool {
        self.lock().is_shared(name)
    }

    /// The mutex guards a counter and an option that every step leaves
    /// consistent, so a poisoned one is still usable.
    fn lock(&self) -> MutexGuard<'_, Gate<Pending>> {
        self.gate.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Sends `pending`; when a response or the final flag fails to go out,
    /// the request gets an error response instead, so that the client is
    /// still answered.
    fn send_now(&self, pending: Pending) -> Result<(), TritonError> {
        let sent = match pending {
            // ResponseSend takes ownership of the response whatever it returns.
            Pending::Response(response, statistics) => {
                let sent = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSend(
                    response,
                    FINAL,
                    ptr::null_mut(),
                ));
                // The response ends when it went out; building its report is not
                // part of the time Triton records for it.
                let end_ns = crate::batch_statistics::now_ns();
                // A response that did not go out is counted through the error
                // response that replaces it, see `send_error`.
                if sent.is_ok()
                    && let Some(statistics) = statistics
                    && let Err(e) = statistics.report(end_ns, None)
                {
                    crate::log::error(&format!("failed to report response statistics: {e}"));
                }
                sent
            }
            Pending::Flags => {
                ResponseFactory::new(self.request).and_then(|factory| factory.send_flags(FINAL))
            }
            Pending::Error(code, message) => return self.send_error(code, &message),
        };
        if sent.is_ok() {
            self.release.succeeded();
        }
        if let Err(e) = &sent {
            let message = format!("failed to send the final response: {e}");
            if let Err(e) = self.send_error(ErrorCode::Internal, &message) {
                crate::log::error(&format!("failed to send error response: {e}"));
            }
        }
        sent
    }

    fn send_error(&self, code: ErrorCode, message: &str) -> Result<(), TritonError> {
        // The caller keeps ownership of the error object passed to ResponseSend,
        // it is deleted on drop.
        let error = TritonError::in_triton(code, message);
        if error.as_ptr().is_null() {
            return Err(TritonError::from_message("failed to create error object"));
        }
        // An error response is created here, so it has no response object the
        // backend could have enabled the statistics on: it enables them on the
        // request instead, see `Request::enable_statistics`, and they are
        // reported with the error, which is what puts the response into the
        // `fail` or `cancel` count of `response_stats`.
        let statistics = self.statistics.load(Ordering::Relaxed).then(|| {
            crate::response::ResponseStatistics::for_request(
                self.request,
                self.release.statistics().instance(),
            )
        });

        let response = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseNew(
            &mut _,
            self.request
        ))?;

        // ResponseSend takes ownership of the response whatever it returns.
        let sent = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSend(
            response,
            FINAL,
            error.as_ptr(),
        ));
        // The response ends when it went out; building its report is not part of
        // the time Triton records for it.
        let end_ns = crate::batch_statistics::now_ns();
        // Reported before the error object is dropped: Triton reads it while it
        // records the response.
        if let Some(statistics) = statistics
            && let Err(e) = statistics.report(end_ns, Some(error.as_ptr()))
        {
            crate::log::error(&format!("failed to report response statistics: {e}"));
        }
        sent
    }
}

impl Drop for FinalResponse {
    fn drop(&mut self) {
        // Every shared input sends what it kept when it is dropped, and holds
        // an `Arc` of this value, so nothing can be left here.
        let pending = self.lock().pending.take();
        if let Some(pending) = pending {
            crate::log::error("a final response was left unsent");
            if let Err(e) = self.send_now(pending) {
                crate::log::error(&format!("failed to send the final response: {e}"));
            }
        }
    }
}

/// When the final response of a request may be sent: once it is given and
/// no shared input is left.
pub(crate) struct Gate<T> {
    shared: usize,
    /// The inputs shared and not dropped yet, by name.
    shared_names: Vec<String>,
    given: bool,
    pending: Option<T>,
}

impl<T> Default for Gate<T> {
    fn default() -> Self {
        Self {
            shared: 0,
            shared_names: Vec::new(),
            given: false,
            pending: None,
        }
    }
}

impl<T> Gate<T> {
    /// Returns the response to send now, or keeps it.
    fn send(&mut self, response: T) -> Option<T> {
        self.given = true;
        if self.shared == 0 {
            return Some(response);
        }
        self.pending = Some(response);
        None
    }

    fn share(&mut self, name: &str) -> Result<(), &'static str> {
        if self.given {
            return Err("the request already has a final response");
        }
        self.shared = self.shared.checked_add(1).ok_or("too many shared inputs")?;
        self.shared_names.push(name.to_owned());
        Ok(())
    }

    /// Whether the input named `name` is shared and not dropped yet.
    fn is_shared(&self, name: &str) -> bool {
        self.shared_names.iter().any(|shared| shared == name)
    }

    /// Returns the response kept for the last shared input.
    fn unshare(&mut self, name: &str) -> Option<T> {
        if let Some(index) = self.shared_names.iter().position(|shared| shared == name) {
            self.shared_names.remove(index);
        }
        self.shared = self.shared.saturating_sub(1);
        if self.shared == 0 {
            self.pending.take()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Gate;

    #[test]
    fn response_without_shared_inputs_is_sent_at_once() {
        let mut gate = Gate::default();

        assert_eq!(gate.send(1), Some(1));
    }

    #[test]
    fn response_is_kept_for_the_last_shared_input() {
        let mut gate = Gate::default();
        gate.share("a").unwrap();
        gate.share("b").unwrap();

        assert_eq!(gate.send(1), None);
        assert_eq!(gate.unshare("a"), None);
        assert_eq!(gate.unshare("b"), Some(1));
    }

    #[test]
    fn shared_inputs_dropped_before_the_response_do_not_hold_it() {
        let mut gate = Gate::default();
        gate.share("a").unwrap();

        assert_eq!(gate.unshare("a"), None::<i32>);
        assert_eq!(gate.send(1), Some(1));
    }

    #[test]
    fn input_is_not_shared_after_the_response() {
        let mut gate = Gate::default();
        gate.share("a").unwrap();
        assert_eq!(gate.send(1), None);

        assert!(gate.share("a").is_err());
        assert_eq!(gate.unshare("a"), Some(1));
        assert!(gate.share("a").is_err());
    }

    #[test]
    fn a_shared_input_is_known_by_its_name_until_it_is_dropped() {
        let mut gate = Gate::<i32>::default();
        gate.share("features").unwrap();
        gate.share("features").unwrap();

        assert!(gate.is_shared("features"));
        assert!(!gate.is_shared("pixels"));

        gate.unshare("features");
        assert!(gate.is_shared("features"));

        gate.unshare("features");
        assert!(!gate.is_shared("features"));
    }
}
