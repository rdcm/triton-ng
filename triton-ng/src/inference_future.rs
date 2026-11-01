use crate::{InferenceResult, TritonError};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Waker};

/// Slot shared by the response callback, which fills it once, and
/// [`InferenceFuture`], which takes the result out of it.
pub(crate) struct Completion {
    state: Mutex<CompletionState>,
}

#[derive(Default)]
struct CompletionState {
    result: Option<InferenceResult>,
    waker: Option<Waker>,
    taken: bool,
}

impl Completion {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(CompletionState::default()),
        })
    }

    pub(crate) fn failed(message: impl Into<String>) -> Arc<Self> {
        let completion = Self::new();
        completion.complete(InferenceResult::failed(message));
        completion
    }

    fn lock(&self) -> MutexGuard<'_, CompletionState> {
        // The state stays consistent even if a holder panicked: every update is a single assignment.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Stores the result and wakes the waiting task. Only the first result is kept.
    pub(crate) fn complete(&self, result: InferenceResult) {
        let waker = {
            let mut state = self.lock();
            if state.result.is_some() || state.taken {
                return;
            }
            state.result = Some(result);
            state.waker.take()
        };
        // Woken outside the lock: an executor may poll the future right away.
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn take(state: &mut CompletionState) -> Option<Result<InferenceResult, String>> {
        if let Some(result) = state.result.take() {
            state.taken = true;
            return Some(match result.error {
                Some(error) => Err(error),
                None => Ok(result),
            });
        }
        if state.taken {
            return Some(Err("inference result was already taken".to_string()));
        }
        None
    }
}

/// Inference running in Triton, resolves to its result.
///
/// The request is already submitted when the future is created, so it runs
/// whether the future is polled or not. Dropping the future does not cancel
/// the request: Triton finishes it and the result is discarded.
///
/// The future holds no Triton pointers, it is `Send + 'static` and works
/// with any executor. Triton wakes it from its own threads.
pub struct InferenceFuture {
    completion: Arc<Completion>,
}

impl InferenceFuture {
    pub(crate) fn new(completion: Arc<Completion>) -> Self {
        Self { completion }
    }

    fn poll_result(&self, cx: &mut Context<'_>) -> Poll<Result<InferenceResult, String>> {
        let mut state = self.completion.lock();
        if let Some(result) = Completion::take(&mut state) {
            return Poll::Ready(result);
        }
        match &state.waker {
            Some(waker) if waker.will_wake(cx.waker()) => {}
            _ => state.waker = Some(cx.waker().clone()),
        }
        Poll::Pending
    }
}

impl Future for InferenceFuture {
    type Output = Result<InferenceResult, TritonError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.poll_result(cx).map_err(TritonError::from_message)
    }
}

#[cfg(test)]
mod tests {
    use super::{Completion, InferenceFuture};
    use crate::InferenceResult;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll, Wake, Waker};

    struct CountingWaker(AtomicUsize);

    impl Wake for CountingWaker {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn poll(future: &InferenceFuture, waker: &Waker) -> Poll<Result<usize, String>> {
        future
            .poll_result(&mut Context::from_waker(waker))
            .map(|r| r.map(|result| result.outputs.len()))
    }

    fn succeeded() -> InferenceResult {
        InferenceResult {
            outputs: vec![],
            error: None,
        }
    }

    #[test]
    fn completion_wakes_pending_future() {
        let counter = Arc::new(CountingWaker(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&counter));
        let completion = Completion::new();
        let future = InferenceFuture::new(Arc::clone(&completion));

        assert_eq!(poll(&future, &waker), Poll::Pending);
        completion.complete(succeeded());

        assert_eq!(counter.0.load(Ordering::SeqCst), 1);
        assert_eq!(poll(&future, &waker), Poll::Ready(Ok(0)));
    }

    #[test]
    fn completion_from_another_thread() {
        let completion = Completion::new();
        let future = InferenceFuture::new(Arc::clone(&completion));

        std::thread::spawn(move || completion.complete(succeeded()))
            .join()
            .unwrap();

        assert_eq!(poll(&future, Waker::noop()), Poll::Ready(Ok(0)));
    }

    #[test]
    fn failed_completion_resolves_to_error() {
        let future = InferenceFuture::new(Completion::failed("submit failed"));

        assert_eq!(
            poll(&future, Waker::noop()),
            Poll::Ready(Err("submit failed".to_string()))
        );
    }

    #[test]
    fn only_first_result_is_kept() {
        let completion = Completion::new();
        let future = InferenceFuture::new(Arc::clone(&completion));

        completion.complete(succeeded());
        completion.complete(InferenceResult::failed("late"));

        assert_eq!(poll(&future, Waker::noop()), Poll::Ready(Ok(0)));
    }

    #[test]
    fn polling_after_completion_is_an_error() {
        let future = InferenceFuture::new(Completion::failed("first"));

        assert!(poll(&future, Waker::noop()).is_ready());
        assert_eq!(
            poll(&future, Waker::noop()),
            Poll::Ready(Err("inference result was already taken".to_string()))
        );
    }
}
