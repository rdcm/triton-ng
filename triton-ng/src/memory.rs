use crate::error::TritonError;
use crate::model::Model;
use std::ffi::c_void;
use std::ops::{Deref, DerefMut};
use std::ptr;
use triton_ng_macros::triton_call;

type MemoryType = triton_ng_sys::TRITONSERVER_MemoryType;

const CPU: MemoryType = triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
const CPU_PINNED: MemoryType =
    triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU_PINNED;
const GPU: MemoryType = triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_GPU;

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
/// Every buffer frees its memory on drop. A buffer must not outlive the
/// server: keep buffers in the state of the model or the instance, which is
/// dropped before the server stops. `MemoryManager` is `Send + Sync + Clone`.
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
    ptr: *mut triton_ng_sys::TRITONBACKEND_MemoryManager,
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
        Ok(Self { ptr })
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
    pub fn ram(&self, len: usize) -> Result<RamBuffer, TritonError> {
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
    pub fn pinned(&self, len: usize) -> Result<PinnedRamBuffer, TritonError> {
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
    pub fn gpu(&self, device_id: i32, len: usize) -> Result<UninitGpuBuffer, TritonError> {
        if device_id < 0 {
            return Err(TritonError::from_message(format!(
                "GPU device id must not be negative, got {device_id}"
            )));
        }
        Ok(UninitGpuBuffer(Allocation::new(
            self,
            len,
            GPU,
            i64::from(device_id),
        )?))
    }
}

/// Memory from [`MemoryManager`], freed on drop. Zero-length allocations do
/// not reach Triton and hold a null pointer.
struct Allocation {
    manager: MemoryManager,
    ptr: *mut c_void,
    len: usize,
    memory_type: MemoryType,
    memory_type_id: i64,
}

// SAFETY: the allocation is plain memory owned by this value, and the memory
// manager is `Send + Sync`.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Allocation {
    fn new(
        manager: &MemoryManager,
        len: usize,
        memory_type: MemoryType,
        memory_type_id: i64,
    ) -> Result<Self, TritonError> {
        let ptr = if len == 0 {
            ptr::null_mut()
        } else {
            let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_MemoryManagerAllocate(
                manager.ptr,
                &mut _,
                memory_type,
                memory_type_id,
                len as u64,
            ))?;
            if ptr.is_null() {
                return Err(TritonError::from_message(
                    "TRITONBACKEND_MemoryManagerAllocate returned null",
                ));
            }
            ptr
        };
        Ok(Self {
            manager: manager.clone(),
            ptr,
            len,
            memory_type,
            memory_type_id,
        })
    }

    fn zeroed(
        manager: &MemoryManager,
        len: usize,
        memory_type: MemoryType,
        memory_type_id: i64,
    ) -> Result<Self, TritonError> {
        let allocation = Self::new(manager, len, memory_type, memory_type_id)?;
        if len > 0 {
            unsafe { ptr::write_bytes(allocation.ptr.cast::<u8>(), 0, len) };
        }
        Ok(allocation)
    }

    /// The bytes of a CPU-accessible allocation.
    fn bytes(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: CPU or pinned memory of `len` bytes, initialized by
        // `zeroed`, and borrowed from its only owner.
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len) }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: as in `bytes`, borrowed mutably from its only owner.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.cast::<u8>(), self.len) }
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if let Err(e) = triton_call!(triton_ng_sys::TRITONBACKEND_MemoryManagerFree(
            self.manager.ptr,
            self.ptr,
            self.memory_type,
            self.memory_type_id,
        )) {
            crate::log::error(&format!("failed to free a memory manager buffer: {e}"));
        }
    }
}

/// CPU memory from [`MemoryManager::ram`], readable and writable as `[u8]`.
pub struct RamBuffer(Allocation);

/// Pinned CPU memory from [`MemoryManager::pinned`], readable and writable
/// as `[u8]`.
pub struct PinnedRamBuffer(Allocation);

/// GPU memory a model has written its output into. It cannot be read on the
/// CPU: pass it as an input of the next BLS call, or turn it back into an
/// [`UninitGpuBuffer`] to receive another output.
pub struct GpuBuffer(Allocation);

/// GPU memory from [`MemoryManager::gpu`] whose content is not defined yet.
/// It can only receive the output of a BLS call.
pub struct UninitGpuBuffer(Allocation);

impl Deref for RamBuffer {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.0.bytes()
    }
}

impl DerefMut for RamBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.0.bytes_mut()
    }
}

impl Deref for PinnedRamBuffer {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.0.bytes()
    }
}

impl DerefMut for PinnedRamBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.0.bytes_mut()
    }
}

impl GpuBuffer {
    /// Size in bytes.
    pub fn len(&self) -> usize {
        self.0.len
    }

    /// Whether the buffer has no bytes.
    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }

    /// The GPU the memory is on.
    pub fn device_id(&self) -> i32 {
        self.0.memory_type_id as i32
    }

    /// Gives up the content, so the memory can receive another output.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let reused: UninitGpuBuffer = cached.into_uninit();
    /// ```
    pub fn into_uninit(self) -> UninitGpuBuffer {
        UninitGpuBuffer(self.0)
    }
}

impl UninitGpuBuffer {
    /// Size in bytes.
    pub fn len(&self) -> usize {
        self.0.len
    }

    /// Whether the buffer has no bytes.
    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }

    /// The GPU the memory is on.
    pub fn device_id(&self) -> i32 {
        self.0.memory_type_id as i32
    }
}

#[cfg(test)]
mod tests {
    use super::{GPU, MemoryManager};
    use std::ptr;

    fn manager() -> MemoryManager {
        MemoryManager {
            ptr: ptr::null_mut(),
        }
    }

    #[test]
    fn empty_buffers_do_not_reach_triton() {
        let manager = manager();
        let mut ram = manager.ram(0).unwrap();
        assert!(ram.is_empty());
        assert!(ram.iter_mut().next().is_none());

        let pinned = manager.pinned(0).unwrap();
        assert!(pinned.is_empty());

        let gpu = manager.gpu(1, 0).unwrap();
        assert!(gpu.is_empty());
        assert_eq!(gpu.device_id(), 1);
        assert_eq!(gpu.0.memory_type, GPU);
    }

    #[test]
    fn buffers_can_move_to_tasks() {
        fn assert_shared<T: Send + Sync + 'static>() {}
        assert_shared::<MemoryManager>();
        assert_shared::<super::RamBuffer>();
        assert_shared::<super::PinnedRamBuffer>();
        assert_shared::<super::GpuBuffer>();
        assert_shared::<super::UninitGpuBuffer>();
    }
}
