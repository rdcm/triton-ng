use crate::{Error, Model, RequestView};
use std::sync::Arc;

/// Decides which requests Triton puts into one batch, for a model whose
/// config names a batching strategy library.
///
/// Triton calls [`BatchStrategy::initialize`] once when the model loads, then
/// [`BatchStrategy::batch`] when it starts forming a batch and
/// [`BatchStrategy::include`] for each queued request in turn, oldest first,
/// until the batch is full. Answering `false` ends the batch before that
/// request: it stays in the queue and can head the next batch, the requests
/// behind it wait with it. Triton asks the strategy only for a model with
/// `dynamic_batching`, and may start a batch and abandon it again before one
/// is handed to a model instance.
///
/// A strategy that answers `false` for the request at the head of the queue
/// forms no batch at all: Triton asks again, with the same request, and the
/// queue does not move. Only a timeout takes a request out of the queue, the
/// timeout the client set or the `default_queue_policy` of the model, so a
/// model whose config sets neither stays stalled for good.
///
/// Triton r25.09 finalizes the batcher before stopping its batching thread and
/// may omit finalization of an unfinished batch. The library retires all of a
/// model's batch tokens at batcher finalization. Callbacks already running keep
/// their values alive until they return; late callbacks cannot access retired
/// values. Destructors run after the last callback and their panics are logged.
/// See the batching guide for the core's library-unload ordering limitation.
///
/// The library exports the entry points of this trait through
/// [`macro@crate::batch_strategy`]; a crate that uses the attribute is the
/// batching strategy library of the model, one crate per strategy.
///
/// # Examples
///
/// ```ignore
/// struct ByKey;
///
/// struct Batch {
///     started: bool,
///     key: Option<String>,
/// }
///
/// #[triton_ng::batch_strategy]
/// impl BatchStrategy for ByKey {
///     type Batch = Batch;
///
///     fn initialize(model: &Model) -> Result<Self, Error> {
///         Ok(Self)
///     }
///
///     fn batch(&self) -> Batch {
///         Batch { started: false, key: None }
///     }
///
///     fn include(&self, batch: &mut Batch, request: &RequestView<'_>) -> Result<bool, Error> {
///         let key = match request.parameter("batch_key")? {
///             Some(Parameter::String(key)) => Some(key),
///             _ => None,
///         };
///         if !batch.started {
///             // The request at the head of the queue joins the batch whatever
///             // its key: a batch needs its head.
///             batch.started = true;
///             batch.key = key;
///             return Ok(true);
///         }
///         Ok(key == batch.key)
///     }
/// }
/// ```
pub trait BatchStrategy: Sized + Send + Sync + 'static {
    /// What the strategy keeps about the batch it is forming.
    ///
    /// One is created when Triton starts a batch and dropped when the batch
    /// is handed to the model instance or abandoned at model unload. In-flight
    /// callbacks delay its destruction until they return. It never sees the
    /// batch execute and cannot report what the batch turned out to be.
    type Batch: Send;

    /// Reads the config of `model` and returns the strategy Triton keeps for
    /// it, until it unloads the model. Called once, on the thread that loads
    /// the model.
    ///
    /// Fails when the config of the model does not fit the strategy; Triton
    /// then fails to load the model.
    fn initialize(model: &Model) -> Result<Self, Error>;

    /// A batch about to be formed. Called on Triton's batching thread when
    /// the previous batch is complete, possibly several times before one is
    /// handed to a model instance.
    fn batch(&self) -> Self::Batch;

    /// Whether `request` joins `batch`. Called on Triton's batching thread
    /// for each queued request, oldest first.
    ///
    /// Fails when the request cannot be read. Triton logs the error and ends
    /// the batch before the request, as it does for `false`: at the head of
    /// the queue, a failing `include` keeps the model from forming a batch.
    /// Mutations made before an error or panic remain in `batch`; there is no
    /// rollback, and later callbacks see that state. Validate before mutating
    /// when the strategy needs failure to leave the batch unchanged.
    fn include(&self, batch: &mut Self::Batch, request: &RequestView<'_>) -> Result<bool, Error>;
}

/// Contains destructor panics independently, including when both the batch and
/// strategy destructors panic during the same cleanup.
pub(crate) struct Protected<T>(Option<T>);

impl<T> Protected<T> {
    pub(crate) fn new(value: T) -> Self {
        Self(Some(value))
    }
    pub(crate) fn get(&self) -> Result<&T, Error> {
        self.0
            .as_ref()
            .ok_or_else(|| "batching state was already dropped".into())
    }
    pub(crate) fn get_mut(&mut self) -> Result<&mut T, Error> {
        self.0
            .as_mut()
            .ok_or_else(|| "batching state was already dropped".into())
    }
}

impl<T> Drop for Protected<T> {
    fn drop(&mut self) {
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(self.0.take()))).is_err() {
            #[cfg(not(test))]
            crate::log::error("batching state destructor panicked");
            #[cfg(test)]
            eprintln!("batching state destructor panicked");
        }
    }
}

/// Both user values have independently contained destructors.
pub(crate) struct BatchState<S: BatchStrategy> {
    pub(crate) strategy: Arc<Protected<S>>,
    pub(crate) batch: Protected<S::Batch>,
}

impl<S: BatchStrategy> BatchState<S> {
    /// Callback errors and panics retain mutations already made by the strategy.
    pub(crate) fn include(&mut self, request: &RequestView<'_>) -> Result<bool, Error> {
        self.strategy.get()?.include(self.batch.get_mut()?, request)
    }
}

use std::any::Any;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

type Value = Arc<dyn Any + Send + Sync>;

/// Triton receives never-reused integer tokens, not dereferenceable pointers.
/// A callback clones its value under the registry lock, then runs user code
/// without that lock. Finalization retires all tokens of the model atomically;
/// in-flight callbacks retain their own Arc until they finish.
#[derive(Default)]
pub(crate) struct Registry {
    next: usize,
    entries: HashMap<usize, (usize, Value)>,
}

pub(crate) static REGISTRY: LazyLock<Mutex<Registry>> = LazyLock::new(Default::default);

impl Registry {
    pub(crate) fn insert(&mut self, owner: usize, value: Value) -> Result<usize, Error> {
        if owner != 0 && !self.entries.contains_key(&owner) {
            return Err("batcher has been finalized".into());
        }
        self.next = self
            .next
            .checked_add(1)
            .ok_or("batching token space exhausted")?;
        self.entries.insert(self.next, (owner, value));
        Ok(self.next)
    }

    pub(crate) fn get<T: Any + Send + Sync>(&self, token: usize) -> Option<Arc<T>> {
        self.entries
            .get(&token)
            .and_then(|(_, value)| Arc::clone(value).downcast().ok())
    }

    /// Returned values must be dropped outside the registry lock: user Drop
    /// implementations may reenter the library or wait for another callback.
    pub(crate) fn remove(&mut self, token: usize) -> Vec<Value> {
        if token == 0 {
            return Vec::new();
        }
        let keys: Vec<_> = self
            .entries
            .iter()
            .filter_map(|(key, (owner, _))| (*key == token || *owner == token).then_some(*key))
            .collect();
        keys.into_iter()
            .filter_map(|key| self.entries.remove(&key).map(|(_, value)| value))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn abandoned_batches_are_released_on_every_unload() -> Result<(), Error> {
        let mut actual_0 = Vec::new();
        let mut expected_0 = Vec::new();
        let mut actual_1 = Vec::new();
        let mut expected_1 = Vec::new();
        let mut actual_2 = Vec::new();
        let mut expected_2 = Vec::new();
        let mut actual_3 = Vec::new();
        let mut expected_3 = Vec::new();

        let mut registry = Registry::default();
        let drops = Arc::new(AtomicUsize::new(0));
        for cycle in 1..=5 {
            let strategy = Arc::new(Protected::new(Counted(drops.clone(), false)));
            let owner = registry.insert(0, strategy)?;
            let batch = Arc::new(Protected::new(Counted(drops.clone(), false)));
            let token = registry.insert(owner, batch)?;
            actual_0.push(registry.remove(0).is_empty());
            expected_0.push(true);
            drop(registry.remove(owner));
            actual_1.push(registry.entries.is_empty());
            expected_1.push(true);
            actual_2.push(drops.load(Ordering::SeqCst));
            expected_2.push(cycle * 2);
            actual_3.push(registry.remove(token).is_empty());
            expected_3.push(true);
        }
        let expected = (expected_0, expected_1, expected_2, expected_3);
        let actual = (actual_0, actual_1, actual_2, actual_3);
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn an_in_flight_callback_survives_finalization() -> Result<(), Error> {
        let mut registry = Registry::default();
        let drops = Arc::new(AtomicUsize::new(0));
        let owner = registry.insert(0, Arc::new(Protected::new(Counted(drops.clone(), false))))?;
        let callback = registry
            .get::<Protected<Counted>>(owner)
            .ok_or("missing strategy")?;
        let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
        let (finish_tx, finish_rx) = std::sync::mpsc::channel::<()>();
        let thread = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            finish_rx.recv().unwrap();
            drop(callback);
        });
        started_rx.recv()?;
        drop(registry.remove(owner));
        let actual_0 = drops.load(Ordering::SeqCst);
        let expected_0 = 0;
        let actual_1 = registry.get::<Protected<Counted>>(owner).is_none();
        let expected_1 = true;
        let actual_2 = registry.insert(owner, Arc::new(())).is_err();
        let expected_2 = true;

        finish_tx.send(())?;
        thread.join().map_err(|_| "callback thread panicked")?;
        let expected = (expected_0, expected_1, expected_2);
        let actual = (actual_0, actual_1, actual_2);
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn the_last_callback_releases_the_finalized_strategy() -> Result<(), Error> {
        let mut registry = Registry::default();
        let drops = Arc::new(AtomicUsize::new(0));
        let owner = registry.insert(0, Arc::new(Protected::new(Counted(drops.clone(), false))))?;
        let callback = registry
            .get::<Protected<Counted>>(owner)
            .ok_or("missing strategy")?;
        let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
        let (finish_tx, finish_rx) = std::sync::mpsc::channel::<()>();
        let thread = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            finish_rx.recv().unwrap();
            drop(callback);
        });
        started_rx.recv()?;
        drop(registry.remove(owner));
        let _ = &(registry.insert(owner, Arc::new(())).is_err());
        finish_tx.send(())?;
        thread.join().map_err(|_| "callback thread panicked")?;
        let actual = drops.load(Ordering::SeqCst);
        let expected = 1;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn a_new_strategy_has_a_distinct_owner_identity() -> Result<(), Error> {
        let mut registry = Registry::default();
        let drops = Arc::new(AtomicUsize::new(0));
        let owner = registry.insert(0, Arc::new(Protected::new(Counted(drops.clone(), false))))?;
        let callback = registry
            .get::<Protected<Counted>>(owner)
            .ok_or("missing strategy")?;
        let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
        let (finish_tx, finish_rx) = std::sync::mpsc::channel::<()>();
        let thread = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            finish_rx.recv().unwrap();
            drop(callback);
        });
        started_rx.recv()?;
        drop(registry.remove(owner));
        let _ = &(registry.insert(owner, Arc::new(())).is_err());
        finish_tx.send(())?;
        thread.join().map_err(|_| "callback thread panicked")?;
        let next = registry.insert(0, Arc::new(()))?;
        let actual = (next) != (owner);
        let expected = true;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn both_destructors_can_panic_without_aborting_cleanup() -> Result<(), Error> {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut registry = Registry::default();
        let owner = registry.insert(0, Arc::new(Protected::new(Counted(drops.clone(), true))))?;
        registry.insert(
            owner,
            Arc::new(Protected::new(Counted(drops.clone(), true))),
        )?;
        let expected = (true, 2, true);
        let actual = (
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                drop(registry.remove(owner))
            }))
            .is_ok(),
            drops.load(Ordering::SeqCst),
            registry.entries.is_empty(),
        );
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn a_failed_callback_returns_its_error() -> Result<(), Error> {
        let strategy = Arc::new(Protected::new(MutatingStrategy));
        let state = Mutex::new(BatchState {
            strategy,
            batch: Protected::new(0),
        });
        // The strategy never reads this view; no Triton calls or owned C objects.
        let request = RequestView::from_ptr(std::ptr::null_mut());
        let expected = true;
        let actual = state.lock().unwrap().include(&request).is_err();
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn a_panicking_callback_is_contained() -> Result<(), Error> {
        let strategy = Arc::new(Protected::new(MutatingStrategy));
        let state = Mutex::new(BatchState {
            strategy,
            batch: Protected::new(0),
        });
        // The strategy never reads this view; no Triton calls or owned C objects.
        let request = RequestView::from_ptr(std::ptr::null_mut());
        let _failed = state.lock().unwrap().include(&request);
        let expected = true;
        let actual = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.lock().unwrap().include(&request)
        }))
        .is_err();
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn callbacks_preserve_mutations_after_errors_and_panics() -> Result<(), Error> {
        let strategy = Arc::new(Protected::new(MutatingStrategy));
        let state = Mutex::new(BatchState {
            strategy,
            batch: Protected::new(0),
        });
        // The strategy never reads this view; no Triton calls or owned C objects.
        let request = RequestView::from_ptr(std::ptr::null_mut());
        let _ = &(state.lock().unwrap().include(&request).is_err());
        let _ = &(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.lock().unwrap().include(&request)
        }))
        .is_err());
        let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
        let expected = (2, true, 3);
        let actual = (
            *state.batch.get()?,
            state.include(&request)?,
            *state.batch.get()?,
        );
        assert_eq!(actual, expected);
        Ok(())
    }

    struct Counted(Arc<AtomicUsize>, bool);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
            if self.1 {
                std::panic::resume_unwind(Box::new("destructor failed"));
            }
        }
    }

    struct MutatingStrategy;

    impl BatchStrategy for MutatingStrategy {
        type Batch = usize;
        fn initialize(_: &Model) -> Result<Self, Error> {
            Ok(Self)
        }
        fn batch(&self) -> usize {
            0
        }
        fn include(&self, batch: &mut usize, _: &RequestView<'_>) -> Result<bool, Error> {
            *batch += 1;
            match *batch {
                1 => Err("failed after mutation".into()),
                2 => std::panic::resume_unwind(Box::new("panicked after mutation")),
                _ => Ok(*batch == 3),
            }
        }
    }
}
