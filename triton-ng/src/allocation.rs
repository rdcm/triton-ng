use crate::memory_manager::MemoryManager;
use crate::{Error, MemoryType, RawBuffer};
use std::ffi::c_void;
use std::ptr;

/// Memory from [`MemoryManager`].
///
/// Owns the memory and frees it on drop. Zero-length allocations do not
/// reach Triton and hold a null pointer. `Send + Sync`.
pub(crate) struct Allocation {
    pub(crate) manager: MemoryManager,
    pub(crate) ptr: *mut c_void,
    pub(crate) len: usize,
    pub(crate) memory_type: MemoryType,
    pub(crate) memory_type_id: i64,
}

// SAFETY: the allocation is plain memory owned by this value, and the memory
// manager is `Send + Sync`.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Allocation {
    pub(crate) fn new(
        manager: &MemoryManager,
        len: usize,
        memory_type: MemoryType,
        memory_type_id: i64,
    ) -> Result<Self, Error> {
        let ptr = if len == 0 {
            ptr::null_mut()
        } else {
            let ptr = (manager.allocate)(manager.ptr, len, memory_type, memory_type_id)?;
            if ptr.is_null() {
                return Err("TRITONBACKEND_MemoryManagerAllocate returned null".into());
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

    pub(crate) fn zeroed(
        manager: &MemoryManager,
        len: usize,
        memory_type: MemoryType,
        memory_type_id: i64,
    ) -> Result<Self, Error> {
        let allocation = Self::new(manager, len, memory_type, memory_type_id)?;
        if len > 0 {
            unsafe { ptr::write_bytes(allocation.ptr.cast::<u8>(), 0, len) };
        }
        Ok(allocation)
    }

    /// The bytes of a CPU-accessible allocation.
    pub(crate) fn bytes(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: CPU or pinned memory of `len` bytes, initialized by
        // `zeroed`, and borrowed from its only owner.
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len) }
    }

    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
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
        (self.manager.free)(
            self.manager.ptr,
            self.ptr,
            self.memory_type,
            self.memory_type_id,
        );
    }
}

impl Allocation {
    pub(crate) fn raw(&self) -> RawBuffer {
        RawBuffer {
            ptr: self.ptr,
            len: self.len,
            memory_type: self.memory_type,
            memory_type_id: self.memory_type_id,
        }
    }
}
