//! Descriptions of the memory a model or a model instance occupies.

use crate::error::TritonError;
use crate::memory_manager::{CPU, CPU_PINNED, GPU};
use triton_ng_macros::triton_call;

/// Memory of one memory type and id, handed to
/// [`crate::Model::report_memory_usage`] or
/// [`crate::ModelInstance::report_memory_usage`].
///
/// Owns the `TRITONSERVER_BufferAttributes` and deletes it on drop. Triton
/// copies what it needs while the report call runs, so the value can be
/// dropped afterwards. Neither `Send` nor `Sync`.
///
/// # Examples
///
/// ```ignore
/// let usage = [BufferAttributes::gpu(0, 64 << 20)?];
/// instance.report_memory_usage(&usage)?;
/// ```
pub struct BufferAttributes {
    ptr: *mut triton_ng_sys::TRITONSERVER_BufferAttributes,
}

impl BufferAttributes {
    /// Memory in the host RAM of the machine.
    ///
    /// # Errors
    ///
    /// Fails when `byte_size` does not fit the platform's `usize`, or when
    /// Triton cannot create the attributes.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let weights = BufferAttributes::cpu(4096)?;
    /// ```
    pub fn cpu(byte_size: u64) -> Result<Self, TritonError> {
        Self::new(CPU, 0, byte_size)
    }

    /// Page-locked host memory, the kind [`crate::PinnedRamBuffer`] holds.
    ///
    /// # Errors
    ///
    /// The same as [`BufferAttributes::cpu`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let staging = BufferAttributes::cpu_pinned(1 << 20)?;
    /// ```
    pub fn cpu_pinned(byte_size: u64) -> Result<Self, TritonError> {
        Self::new(CPU_PINNED, 0, byte_size)
    }

    /// Memory on the CUDA device `device_id`, counting from zero.
    ///
    /// # Errors
    ///
    /// The same as [`BufferAttributes::cpu`]; the device is not checked, Triton
    /// reports an id it has no device for as it is.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let weights = BufferAttributes::gpu(1, 256 << 20)?;
    /// ```
    pub fn gpu(device_id: i64, byte_size: u64) -> Result<Self, TritonError> {
        Self::new(GPU, device_id, byte_size)
    }

    fn new(
        memory_type: crate::memory_manager::MemoryType,
        memory_type_id: i64,
        byte_size: u64,
    ) -> Result<Self, TritonError> {
        let attributes = Self {
            ptr: triton_call!(triton_ng_sys::TRITONSERVER_BufferAttributesNew(&mut _))?,
        };
        // Triton reads every field of the entry, and the C object starts with
        // none of them set, so all three are set here.
        triton_call!(triton_ng_sys::TRITONSERVER_BufferAttributesSetMemoryType(
            attributes.ptr,
            memory_type,
        ))?;
        triton_call!(triton_ng_sys::TRITONSERVER_BufferAttributesSetMemoryTypeId(
            attributes.ptr,
            memory_type_id,
        ))?;
        let byte_size = usize::try_from(byte_size)
            .map_err(|_| TritonError::from_message("memory usage byte size is too large"))?;
        triton_call!(triton_ng_sys::TRITONSERVER_BufferAttributesSetByteSize(
            attributes.ptr,
            byte_size,
        ))?;
        Ok(attributes)
    }

    /// The pointers of a report and the count Triton takes. The pointers stay
    /// valid while the returned vector is alive.
    pub(crate) fn report(
        usage: &[BufferAttributes],
    ) -> Result<(Vec<*mut triton_ng_sys::TRITONSERVER_BufferAttributes>, u32), TritonError> {
        let pointers: Vec<_> = usage.iter().map(|attributes| attributes.ptr).collect();
        let count = u32::try_from(pointers.len())
            .map_err(|_| TritonError::from_message("too many memory usage entries"))?;
        Ok((pointers, count))
    }
}

impl Drop for BufferAttributes {
    fn drop(&mut self) {
        if let Err(error) =
            triton_call!(triton_ng_sys::TRITONSERVER_BufferAttributesDelete(self.ptr))
        {
            crate::log::error(&format!("failed to delete the buffer attributes: {error}"));
        }
    }
}
