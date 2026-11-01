use crate::allocation::Allocation;
use crate::error::{Error, TritonError};
use crate::model::Model;
use crate::{PinnedRamBuffer, RamBuffer, UninitGpuBuffer};
use std::ffi::c_void;
use triton_ng_macros::triton_call;

pub(crate) type MemoryType = triton_ng_sys::TRITONSERVER_MemoryType;

pub(crate) const CPU: MemoryType =
    triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
pub(crate) const CPU_PINNED: MemoryType =
    triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU_PINNED;
pub(crate) const GPU: MemoryType =
    triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_GPU;

/// Triton's memory manager: buffers from the server's own pools. Batches get
/// it from [`crate::Context::memory`].
///
/// - [`MemoryManager::ram`]: plain CPU memory.
/// - [`MemoryManager::pinned`]: page-locked CPU memory from the pool sized by
///   `--pinned-memory-pool-byte-size`, copied to and from the GPU faster than
///   plain memory.
/// - [`MemoryManager::gpu`]: GPU memory from the pool sized by
///   `--cuda-memory-pool-byte-size` (64 MB per GPU by default), shared with
///   Triton itself. The pools do not grow.
///
/// Borrows the `TRITONBACKEND_MemoryManager`: Triton owns it for the whole
/// process, so there is nothing to drop. Every buffer frees its memory on
/// drop. A buffer must not outlive the server: keep buffers in the state of
/// the model or the instance, which is dropped before the server stops.
/// `Send + Sync + Clone`.
///
/// # Examples
///
/// ```ignore
/// async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
///     let mut tokens = ctx.memory().pinned(requests.len() * 8)?;
///     tokens.copy_from_slice(&encode_tokens(&requests));
///     let state = ctx.memory().gpu(ctx.device_id(), requests.len() * 320 * 4)?;
///     // ... pass `tokens` as an input and `state` as an output of a BLS call ...
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct MemoryManager {
    pub(crate) ptr: *mut triton_ng_sys::TRITONBACKEND_MemoryManager,
    /// Triton calls go through these, so unit tests can use heap memory
    /// without linking libtritonserver.
    pub(crate) allocate: Allocate,
    pub(crate) free: Free,
}

type ManagerPtr = *mut triton_ng_sys::TRITONBACKEND_MemoryManager;
type Allocate = fn(ManagerPtr, usize, MemoryType, i64) -> Result<*mut c_void, Error>;
type Free = fn(ManagerPtr, *mut c_void, MemoryType, i64);

fn triton_allocate(
    manager: ManagerPtr,
    len: usize,
    memory_type: MemoryType,
    memory_type_id: i64,
) -> Result<*mut c_void, Error> {
    Ok(triton_call!(
        triton_ng_sys::TRITONBACKEND_MemoryManagerAllocate(
            manager,
            &mut _,
            memory_type,
            memory_type_id,
            len as u64,
        )
    )?)
}

fn triton_free(
    manager: ManagerPtr,
    ptr: *mut c_void,
    memory_type: MemoryType,
    memory_type_id: i64,
) {
    if let Err(e) = triton_call!(triton_ng_sys::TRITONBACKEND_MemoryManagerFree(
        manager,
        ptr,
        memory_type,
        memory_type_id,
    )) {
        crate::log::error(&format!("failed to free a memory manager buffer: {e}"));
    }
}

// SAFETY: the pointer is the memory manager of the core, a single object for
// the whole process (TRITONBACKEND_BackendMemoryManager returns the same
// static for every backend). Allocate and Free are thread-safe: the GPU pool
// (cnmem) and the pinned pool lock a mutex, CPU memory is malloc/free.
unsafe impl Send for MemoryManager {}
unsafe impl Sync for MemoryManager {}

impl MemoryManager {
    pub(crate) fn of_model(model: &Model) -> Result<Self, TritonError> {
        let backend = triton_call!(triton_ng_sys::TRITONBACKEND_ModelBackend(
            model.as_ptr(),
            &mut _
        ))?;
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_BackendMemoryManager(
            backend,
            &mut _
        ))?;
        if ptr.is_null() {
            return Err(TritonError::from_message(
                "TRITONBACKEND_BackendMemoryManager returned null",
            ));
        }
        Ok(Self {
            ptr,
            allocate: triton_allocate,
            free: triton_free,
        })
    }

    /// A manager that takes every kind of memory from the heap, for unit tests.
    #[cfg(test)]
    pub(crate) fn heap() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            allocate: |_, len, _, _| Ok(unsafe { libc::calloc(len, 1) }),
            free: |_, ptr, _, _| unsafe { libc::free(ptr) },
        }
    }

    /// Allocates `len` bytes of CPU memory, filled with zeros.
    ///
    /// Fails when the memory cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut image = ctx.memory().ram(28 * 28 * 4)?;
    /// image.copy_from_slice(request.get_input("Input3")?.as_bytes()?);
    /// ```
    pub fn ram(&self, len: usize) -> Result<RamBuffer, Error> {
        Ok(RamBuffer(Allocation::zeroed(self, len, CPU, 0)?))
    }

    /// Allocates `len` bytes of pinned CPU memory, filled with zeros.
    ///
    /// Fails when the pinned pool has no free block of `len` bytes or is
    /// disabled (`--pinned-memory-pool-byte-size=0`), and on servers built
    /// without GPU support: there is no fallback to plain memory.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut tokens = ctx.memory().pinned(batch * 8)?;
    /// tokens.copy_from_slice(&token_bytes);
    /// ```
    pub fn pinned(&self, len: usize) -> Result<PinnedRamBuffer, Error> {
        Ok(PinnedRamBuffer(Allocation::zeroed(
            self, len, CPU_PINNED, 0,
        )?))
    }

    /// Allocates `len` bytes on GPU `device_id`. The memory is not
    /// initialized and cannot be written from the CPU: the buffer becomes a
    /// [`GpuBuffer`] once a model writes its output into it.
    ///
    /// Fails when the GPU pool has no free block of `len` bytes (Triton
    /// reports it as an INTERNAL error), when `device_id` is negative, and on
    /// servers built without GPU support.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let state = ctx.memory().gpu(ctx.device_id(), batch * 320 * 4)?;
    /// ```
    pub fn gpu(&self, device_id: i32, len: usize) -> Result<UninitGpuBuffer, Error> {
        if device_id < 0 {
            return Err(format!("GPU device id must not be negative, got {device_id}").into());
        }
        Ok(UninitGpuBuffer(Allocation::new(
            self,
            len,
            GPU,
            i64::from(device_id),
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::MemoryManager;

    #[test]
    fn cpu_buffers_are_zeroed_and_writable() {
        let memory = MemoryManager::heap();
        let mut ram = memory.ram(16).unwrap();
        let mut pinned = memory.pinned(8).unwrap();

        assert_eq!(&ram[..], &[0; 16]);
        assert_eq!(&pinned[..], &[0; 8]);
        ram[3] = 7;
        pinned.copy_from_slice(&[1; 8]);
        assert_eq!(ram[3], 7);
        assert_eq!(&pinned[..], &[1; 8]);
    }

    #[test]
    fn negative_gpu_device_is_rejected() {
        assert!(MemoryManager::heap().gpu(-1, 4).is_err());
    }

    #[test]
    fn empty_buffers_do_not_reach_triton() {
        let memory = MemoryManager {
            ptr: std::ptr::null_mut(),
            allocate: |_, _, _, _| panic!("allocated an empty buffer"),
            free: |_, _, _, _| panic!("freed an empty buffer"),
        };

        assert!(memory.ram(0).unwrap().is_empty());
        assert!(memory.pinned(0).unwrap().is_empty());
        assert!(memory.gpu(0, 0).unwrap().is_empty());
    }
}
