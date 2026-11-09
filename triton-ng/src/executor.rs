use crate::backend_state::SharedState;
use crate::error::Error;
use crate::memory_manager::MemoryManager;
use crate::model::Model;
use crate::server::Server;
use std::any::Any;
use std::future::Future;
use std::pin::pin;
use std::sync::{Arc, OnceLock};
use std::task::{Poll, Wake, Waker};
use std::thread::{self, Thread};

/// What a batch gets from its backend, model and instance: the state stored
/// with [`crate::Backend::set_state`], the server for BLS calls, Triton's
/// memory manager, the state stored with [`Model::set_state`] and the state
/// of its instance stored with [`crate::ModelInstance::set_state`].
///
/// `Context` is `Send + Clone` and holds no Triton pointers, so it can be used
/// across `.await` and moved into the futures a batch joins.
///
/// # Examples
///
/// ```ignore
/// async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
///     let state = ctx.state::<State>()?;
///     let mut inference = InferenceRequest::new(ctx.server(), &state.encoder, -1)?;
///     // ... add the inputs of all `requests` as one batch ...
///     let result = ctx.server().infer_async(inference).await?;
///     // ... answer every request from its rows of `result` ...
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct Context {
    server: Server,
    memory: MemoryManager,
    backend_state: Option<SharedState>,
    state: Option<SharedState>,
    instance_state: Option<SharedState>,
    device_id: i32,
}

impl Context {
    /// The server to run BLS calls on, see [`Server::infer_async`].
    pub fn server(&self) -> &Server {
        &self.server
    }

    /// Triton's memory manager, for buffers of BLS inputs and outputs.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut tokens = ctx.memory().pinned(requests.len() * 8)?;
    /// ```
    pub fn memory(&self) -> &MemoryManager {
        &self.memory
    }

    /// The state shared by every model of the backend, stored with
    /// [`crate::Backend::set_state`].
    ///
    /// Fails when the backend has no state or it is not a `T`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let pool_size = ctx.backend_state::<BackendState>()?.pool_size;
    /// ```
    pub fn backend_state<T: Any>(&self) -> Result<&T, Error> {
        self.backend_state
            .as_deref()
            .ok_or("backend state is not initialized")?
            .downcast_ref::<T>()
            .ok_or_else(|| "backend state has unexpected type".into())
    }

    /// The state stored with [`Model::set_state`].
    ///
    /// Fails when the model has no state or it is not a `T`.
    pub fn state<T: Any>(&self) -> Result<&T, Error> {
        self.state
            .as_deref()
            .ok_or("model state is not initialized")?
            .downcast_ref::<T>()
            .ok_or_else(|| "model state has unexpected type".into())
    }

    /// The state of the instance that runs the batch, stored with
    /// [`crate::ModelInstance::set_state`]: what must not be shared between the
    /// instances of a model, like caches of GPU buffers.
    ///
    /// Fails when the instance has no state or it is not a `T`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let cache = ctx.instance_state::<Mutex<GpuCache>>()?;
    /// let state = cache.lock()?.take(ctx.memory(), ctx.device_id(), bytes)?;
    /// ```
    pub fn instance_state<T: Any>(&self) -> Result<&T, Error> {
        self.instance_state
            .as_deref()
            .ok_or("model instance state is not initialized")?
            .downcast_ref::<T>()
            .ok_or_else(|| "model instance state has unexpected type".into())
    }

    /// Device of the instance that runs the batch (GPU id for `KIND_GPU`).
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

/// The batches of one model instance and the state of the instance.
///
/// A batch runs to its end on the Triton thread that called execute, see
/// [`block_on`]: the instance takes its next batch only after that, and the
/// batches of a model run in parallel on its instances (`instance_group`).
pub(crate) struct InstanceExecutor {
    state: OnceLock<SharedState>,
    /// What the requests of the model may do, read once instead of parsing
    /// the configuration for every batch.
    policy: RequestPolicy,
}

/// What the requests of a model may do besides getting one final response,
/// from its configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RequestPolicy {
    /// Several responses, see [`crate::Request::into_sender`].
    pub(crate) decoupled: bool,
    /// Rescheduling, see [`crate::Request::reschedule`].
    pub(crate) iterative_sequence: bool,
    /// A batch dimension in every input (`max_batch_size` above 0), which
    /// gives the batch size reported to Triton's statistics.
    pub(crate) batching: bool,
}

impl InstanceExecutor {
    /// Creates the executor of an instance of `model`.
    pub(crate) fn new(model: &Model) -> Result<Self, Error> {
        Ok(Self {
            state: OnceLock::new(),
            policy: RequestPolicy {
                decoupled: model.is_decoupled()?,
                iterative_sequence: model.is_iterative_sequence()?,
                batching: model.max_batch_size()? > 0,
            },
        })
    }

    /// What the requests of the model of the instance may do.
    pub(crate) fn policy(&self) -> RequestPolicy {
        self.policy
    }

    /// Stores the state of the instance. Fails if one is already stored.
    pub(crate) fn set_state(&self, state: SharedState) -> Result<(), Error> {
        self.state
            .set(state)
            .map_err(|_| "model instance state is already set".into())
    }
}

/// Wakes the thread that waits in [`block_on`].
struct Unpark(Thread);

impl Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

/// Runs `future` to its end on the calling thread, blocking it while it waits.
///
/// BLS futures are woken by the response callback on a Triton thread of the
/// target model; nothing else runs here, so there is no reactor: timers and
/// I/O of an async runtime do not work inside a batch.
pub(crate) fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(Unpark(thread::current())));
    let mut cx = std::task::Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(output) => return output,
            // A wake that came before the park makes it return at once, a
            // spurious return only polls once more.
            Poll::Pending => thread::park(),
        }
    }
}

/// Server and memory manager of `model`, the state of its backend, of the
/// model and of the instance run by `executor`, and the device of the instance,
/// as passed to batches.
pub(crate) fn context(
    model: &Model,
    executor: &InstanceExecutor,
    device_id: i32,
) -> Result<Context, Error> {
    Ok(Context {
        server: model.get_server()?,
        memory: MemoryManager::of_model(model)?,
        backend_state: model.backend_state()?,
        state: model.shared_state()?,
        instance_state: executor.state.get().cloned(),
        device_id,
    })
}

#[cfg(test)]
mod tests {
    use super::block_on;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll};
    use std::thread;
    use tokio::sync::oneshot;

    #[test]
    fn ready_future_returns_its_output() {
        assert_eq!(block_on(async { 7 }), 7);
    }

    #[test]
    fn future_woken_from_another_thread_completes() {
        let (sender, receiver) = oneshot::channel();
        let sending = thread::spawn(move || {
            sender.send(5).unwrap();
        });
        assert_eq!(block_on(receiver), Ok(5));
        sending.join().unwrap();
    }

    /// Wakes itself before returning `Pending`: the unpark comes before the
    /// park, which must not lose it.
    struct WakesBeforePending(Arc<AtomicUsize>);

    impl Future for WakesBeforePending {
        type Output = usize;

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<usize> {
            let polls = self.0.fetch_add(1, Ordering::SeqCst) + 1;
            if polls == 3 {
                return Poll::Ready(polls);
            }
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }

    #[test]
    fn wake_before_park_is_not_lost() {
        assert_eq!(
            block_on(WakesBeforePending(Arc::new(AtomicUsize::new(0)))),
            3
        );
    }

    #[test]
    fn futures_joined_in_one_batch_wait_together() {
        let (first_sender, first) = oneshot::channel();
        let (second_sender, second) = oneshot::channel();
        let sending = thread::spawn(move || {
            second_sender.send(2).unwrap();
            first_sender.send(1).unwrap();
        });
        let (first, second) = block_on(async { (first.await, second.await) });
        assert_eq!((first, second), (Ok(1), Ok(2)));
        sending.join().unwrap();
    }

    #[test]
    fn panic_in_the_future_reaches_the_caller() {
        let caught = std::panic::catch_unwind(|| block_on(async { panic!("in batch") }));
        assert!(caught.is_err());
    }
}
