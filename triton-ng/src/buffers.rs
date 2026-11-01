use crate::allocation::Allocation;
use std::ops::{Deref, DerefMut};

/// CPU memory from [`MemoryManager::ram`], readable and writable as `[u8]`.
pub struct RamBuffer(pub(crate) Allocation);

/// Pinned CPU memory from [`MemoryManager::pinned`], readable and writable
/// as `[u8]`.
pub struct PinnedRamBuffer(pub(crate) Allocation);

/// GPU memory a model has written its output into. It cannot be read on the
/// CPU: pass it as an input of the next BLS call, or turn it back into an
/// [`UninitGpuBuffer`] to receive another output.
pub struct GpuBuffer(pub(crate) Allocation);

/// GPU memory from [`MemoryManager::gpu`] whose content is not defined yet.
/// It can only receive the output of a BLS call.
pub struct UninitGpuBuffer(pub(crate) Allocation);

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
