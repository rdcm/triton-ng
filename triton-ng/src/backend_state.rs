use std::any::Any;
use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::sync::Arc;

/// The type erased state of a backend, a model or an instance.
pub(crate) type SharedState = Arc<dyn Any + Send + Sync>;

type BoxedState = SharedState;

/// Where the pointer of a backend state is kept.
///
/// [`crate::Backend`] implements this with `TRITONBACKEND_BackendState` and
/// `TRITONBACKEND_BackendSetState`. The unit tests implement it with a plain
/// cell, so the ownership rules of [`set`], [`get`], [`shared`] and [`clear`]
/// are tested without a Triton server.
pub(crate) trait StateStore {
    /// The error of the store itself.
    type Error;

    /// The pointer Triton holds for this backend, null when it holds none.
    fn raw_state(&self) -> Result<*mut c_void, Self::Error>;

    /// Hands Triton `state` as the pointer of the backend; a null pointer
    /// clears what Triton holds.
    fn set_raw_state(&self, state: *mut c_void) -> Result<(), Self::Error>;
}

/// What [`set`], [`get`], [`shared`] and [`clear`] can fail with.
#[derive(Debug)]
pub(crate) enum StateError<E> {
    /// The store failed: in production the Triton C API.
    Store(E),
    /// A state is stored already.
    AlreadySet,
    /// No state is stored.
    NotInitialized,
    /// The stored state is not the type the caller asked for.
    WrongType,
    /// Dropping the stored state panicked; the panic was contained here so it
    /// cannot cross the boundary of a C callback.
    DropPanicked,
}

/// Stores `state` in `store`, unless Triton already holds one.
///
/// The value is boxed once, and the box is owned by Triton from the moment the
/// store accepts it: [`clear`] drops it, [`get`] and [`shared`] borrow it.
pub(crate) fn set<S, T>(store: &S, state: T) -> Result<(), StateError<S::Error>>
where
    S: StateStore,
    T: Any + Send + Sync,
{
    if !store.raw_state().map_err(StateError::Store)?.is_null() {
        return Err(StateError::AlreadySet);
    }

    let state_ptr = Box::into_raw(Box::new(Arc::new(state) as BoxedState)) as *mut c_void;
    if let Err(e) = store.set_raw_state(state_ptr) {
        // The store did not take the pointer, so the box is still ours.
        let _ = drop_boxed(state_ptr);
        return Err(StateError::Store(e));
    }

    Ok(())
}

/// A new reference to the state of `store`, `None` when it holds none.
pub(crate) fn shared<S: StateStore>(
    store: &S,
) -> Result<Option<SharedState>, StateError<S::Error>> {
    let state_ptr = store.raw_state().map_err(StateError::Store)?;
    if state_ptr.is_null() {
        return Ok(None);
    }

    // SAFETY: `state_ptr` was created by `set` from a `Box<BoxedState>` and
    // stays alive until `clear`, which runs on backend finalize, after every
    // hook that could borrow it.
    Ok(Some(unsafe { &*(state_ptr as *const BoxedState) }.clone()))
}

/// The state of `store` as a `T`, which has to be the type it was set with.
pub(crate) fn get<S, T>(store: &S) -> Result<&T, StateError<S::Error>>
where
    S: StateStore,
    T: Any,
{
    let state_ptr = store.raw_state().map_err(StateError::Store)?;
    if state_ptr.is_null() {
        return Err(StateError::NotInitialized);
    }

    // SAFETY: as in `shared`; the returned reference borrows the store, so no
    // caller can hold it across the finalize that clears the pointer.
    let state = unsafe { &*(state_ptr as *const BoxedState) };
    state.downcast_ref::<T>().ok_or(StateError::WrongType)
}

/// Clears the pointer Triton holds and drops the state, unless none is stored.
///
/// Triton is cleared first, so a state whose destructor panics is gone from the
/// backend even though dropping it failed.
pub(crate) fn clear<S: StateStore>(store: &S) -> Result<(), StateError<S::Error>> {
    let state_ptr = store.raw_state().map_err(StateError::Store)?;
    if state_ptr.is_null() {
        return Ok(());
    }

    store
        .set_raw_state(ptr::null_mut())
        .map_err(StateError::Store)?;
    drop_boxed(state_ptr).map_err(|()| StateError::DropPanicked)
}

/// Drops the box behind `state_ptr`, containing a panic of the destructor.
fn drop_boxed(state_ptr: *mut c_void) -> Result<(), ()> {
    // SAFETY: `state_ptr` came from `Box::into_raw` of a `Box<BoxedState>` and
    // is dropped exactly once: here, or in `clear` after the store accepted it.
    catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(state_ptr as *mut BoxedState))
    }))
    .map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use super::{StateError, StateStore, clear, get, set, shared};
    use std::cell::Cell;
    use std::ffi::c_void;
    use std::ptr;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A store that keeps its pointer in a cell, like Triton keeps it in the
    /// backend, and that can be told to fail the way the C API fails.
    struct FakeStore {
        state: Cell<*mut c_void>,
        set_fails: bool,
    }

    impl FakeStore {
        fn new() -> Self {
            Self {
                state: Cell::new(ptr::null_mut()),
                set_fails: false,
            }
        }

        fn failing() -> Self {
            Self {
                state: Cell::new(ptr::null_mut()),
                set_fails: true,
            }
        }
    }

    impl StateStore for FakeStore {
        type Error = String;

        fn raw_state(&self) -> Result<*mut c_void, String> {
            Ok(self.state.get())
        }

        fn set_raw_state(&self, state: *mut c_void) -> Result<(), String> {
            if self.set_fails {
                return Err("the backend does not take a state".to_string());
            }
            self.state.set(state);
            Ok(())
        }
    }

    /// Counts its drops, so a test can tell how often a state was destroyed.
    struct Counted {
        name: &'static str,
        drops: Arc<AtomicUsize>,
    }

    impl Drop for Counted {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Panics when dropped, to test that the panic stays inside `clear`.
    struct Panicking;

    impl Drop for Panicking {
        fn drop(&mut self) {
            panic!("the destructor of the state panicked");
        }
    }

    #[test]
    fn a_state_is_stored_read_and_shared() -> Result<(), String> {
        let store = FakeStore::new();
        set(
            &store,
            Counted {
                name: "first",
                drops: Arc::new(AtomicUsize::new(0)),
            },
        )
        .map_err(|e| format!("{e:?}"))?;

        assert_eq!(
            get::<_, Counted>(&store)
                .map_err(|e| format!("{e:?}"))?
                .name,
            "first"
        );
        let shared = shared(&store)
            .map_err(|e| format!("{e:?}"))?
            .ok_or("no shared state")?;
        assert_eq!(
            shared.downcast_ref::<Counted>().map(|c| c.name),
            Some("first")
        );
        Ok(())
    }

    #[test]
    fn reading_before_setting_and_the_wrong_type_fail() -> Result<(), String> {
        let store = FakeStore::new();
        assert!(matches!(
            get::<_, Counted>(&store),
            Err(StateError::NotInitialized)
        ));
        assert!(matches!(
            get::<_, u32>(&store),
            Err(StateError::NotInitialized)
        ));
        assert!(
            shared(&store).map_err(|e| format!("{e:?}"))?.is_none(),
            "an unprepared store reports no state"
        );

        set(&store, 42u32).map_err(|e| format!("{e:?}"))?;
        assert!(matches!(
            get::<_, Counted>(&store),
            Err(StateError::WrongType)
        ));
        assert!(matches!(
            get::<_, String>(&store),
            Err(StateError::WrongType)
        ));
        assert_eq!(*get::<_, u32>(&store).map_err(|e| format!("{e:?}"))?, 42);
        Ok(())
    }

    #[test]
    fn a_second_state_is_refused_and_the_first_survives() -> Result<(), String> {
        let store = FakeStore::new();
        let drops = Arc::new(AtomicUsize::new(0));
        set(
            &store,
            Counted {
                name: "first",
                drops: Arc::clone(&drops),
            },
        )
        .map_err(|e| format!("{e:?}"))?;

        let rejected = set(
            &store,
            Counted {
                name: "second",
                drops: Arc::clone(&drops),
            },
        );
        assert!(matches!(rejected, Err(StateError::AlreadySet)));
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "the rejected state is dropped once"
        );

        let state = get::<_, Counted>(&store).map_err(|e| format!("{e:?}"))?;
        assert_eq!(state.name, "first");
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "the stored state is untouched"
        );
        Ok(())
    }

    #[test]
    fn a_store_that_refuses_the_pointer_leaves_no_leak() -> Result<(), String> {
        let store = FakeStore::failing();
        let drops = Arc::new(AtomicUsize::new(0));
        let rejected = set(
            &store,
            Counted {
                name: "rejected",
                drops: Arc::clone(&drops),
            },
        );
        assert!(matches!(rejected, Err(StateError::Store(_))));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(store.state.get(), ptr::null_mut(), "the store is unchanged");
        Ok(())
    }

    #[test]
    fn clearing_drops_once_and_nothing_else() -> Result<(), String> {
        let store = FakeStore::new();
        let drops = Arc::new(AtomicUsize::new(0));
        set(
            &store,
            Counted {
                name: "state",
                drops: Arc::clone(&drops),
            },
        )
        .map_err(|e| format!("{e:?}"))?;

        clear(&store).map_err(|e| format!("{e:?}"))?;
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(store.state.get(), ptr::null_mut());
        assert!(matches!(
            get::<_, Counted>(&store),
            Err(StateError::NotInitialized)
        ));

        clear(&store).map_err(|e| format!("{e:?}"))?;
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "clearing again drops nothing"
        );
        Ok(())
    }

    #[test]
    fn a_shared_reference_outlives_the_cleared_pointer() -> Result<(), String> {
        let store = FakeStore::new();
        let drops = Arc::new(AtomicUsize::new(0));
        set(
            &store,
            Counted {
                name: "state",
                drops: Arc::clone(&drops),
            },
        )
        .map_err(|e| format!("{e:?}"))?;

        let shared = shared(&store)
            .map_err(|e| format!("{e:?}"))?
            .ok_or("no shared state")?;
        clear(&store).map_err(|e| format!("{e:?}"))?;
        assert_eq!(drops.load(Ordering::SeqCst), 0, "the batch still holds it");
        assert_eq!(
            shared.downcast_ref::<Counted>().map(|c| c.name),
            Some("state")
        );

        drop(shared);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn a_panicking_destructor_does_not_escape_clear() -> Result<(), String> {
        let store = FakeStore::new();
        set(&store, Panicking).map_err(|e| format!("{e:?}"))?;

        let cleared = clear(&store);
        assert!(matches!(cleared, Err(StateError::DropPanicked)));
        assert_eq!(
            store.state.get(),
            ptr::null_mut(),
            "Triton is left consistent"
        );
        Ok(())
    }
}
