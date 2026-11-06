//! Descriptions of the memory a model or a model instance occupies.

use crate::MemoryType;
use crate::error::TritonError;
use triton_ng_macros::triton_call;

/// Memory of one kind, handed to [`crate::Model::report_memory_usage`] or
/// [`crate::ModelInstance::report_memory_usage`].
///
/// Owns the `TRITONSERVER_BufferAttributes` and deletes it on drop. Triton
/// copies what it needs while the report call runs, so the value can be dropped
/// afterwards. Neither `Send` nor `Sync`.
///
/// Entries of the same memory type are summed by the report call, since Triton
/// keeps one entry per memory type and id and would drop the others.
///
/// # Examples
///
/// ```ignore
/// let usage = [BufferAttributes::new(MemoryType::Gpu { device_id: 0 }, 64 << 20)?];
/// instance.report_memory_usage(&usage)?;
/// ```
pub struct BufferAttributes {
    ptr: *mut triton_ng_sys::TRITONSERVER_BufferAttributes,
    memory_type: MemoryType,
    byte_size: u64,
}

impl BufferAttributes {
    /// Memory of `byte_size` bytes of `memory_type`.
    ///
    /// # Errors
    ///
    /// Fails when `byte_size` does not fit the platform's `usize`, or when
    /// Triton cannot create the attributes.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let weights = BufferAttributes::new(MemoryType::Cpu, 4096)?;
    /// ```
    pub fn new(memory_type: MemoryType, byte_size: u64) -> Result<Self, TritonError> {
        let size = usize::try_from(byte_size)
            .map_err(|_| TritonError::from_message("memory usage byte size is too large"))?;
        let (kind, memory_type_id) = memory_type.to_sys();
        // The attributes are owned from here on, so a failing setter deletes them.
        let attributes = Self {
            ptr: triton_call!(triton_ng_sys::TRITONSERVER_BufferAttributesNew(&mut _))?,
            memory_type,
            byte_size,
        };
        // Triton reads every field of the entry, and the C object starts with
        // none of them set, so all three are set here.
        triton_call!(triton_ng_sys::TRITONSERVER_BufferAttributesSetMemoryType(
            attributes.ptr,
            kind
        ))?;
        triton_call!(triton_ng_sys::TRITONSERVER_BufferAttributesSetMemoryTypeId(
            attributes.ptr,
            memory_type_id
        ))?;
        triton_call!(triton_ng_sys::TRITONSERVER_BufferAttributesSetByteSize(
            attributes.ptr,
            size
        ))?;
        Ok(attributes)
    }

    /// Groups `usage` by memory type and builds the entries Triton takes,
    /// together with their count. The entries own the attributes, so they must
    /// outlive the call that hands their pointers to Triton.
    pub(crate) fn report(
        usage: &[BufferAttributes],
    ) -> Result<(Vec<BufferAttributes>, u32), TritonError> {
        let entries: Vec<_> = usage
            .iter()
            .map(|usage| (usage.memory_type, usage.byte_size))
            .collect();
        // Triton keeps the last entry of a memory type and id, so entries of the
        // same one have to be summed before they are reported.
        let totals = totals(&entries).ok_or_else(|| {
            TritonError::from_message("the memory usage of one memory type and id overflows u64")
        })?;
        let grouped = totals
            .into_iter()
            .map(|(memory_type, byte_size)| Self::new(memory_type, byte_size))
            .collect::<Result<Vec<_>, _>>()?;
        let count = u32::try_from(grouped.len())
            .map_err(|_| TritonError::from_message("too many memory usage entries"))?;
        Ok((grouped, count))
    }

    /// The `TRITONSERVER_BufferAttributes`, valid while `self` lives.
    pub(crate) fn as_ptr(&self) -> *mut triton_ng_sys::TRITONSERVER_BufferAttributes {
        self.ptr
    }
}

/// Sums the entries of each memory type, keeping the order in which they first
/// appear. `None` when one of the sums does not fit `u64`.
fn totals(entries: &[(MemoryType, u64)]) -> Option<Vec<(MemoryType, u64)>> {
    let mut totals: Vec<(MemoryType, u64)> = Vec::with_capacity(entries.len());
    for &(memory_type, byte_size) in entries {
        match totals.iter_mut().find(|(kind, _)| *kind == memory_type) {
            Some((_, total)) => *total = total.checked_add(byte_size)?,
            None => totals.push((memory_type, byte_size)),
        }
    }
    Some(totals)
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

#[cfg(test)]
mod tests {
    use super::totals;
    use crate::MemoryType;

    #[test]
    fn one_entry_per_memory_type_stays_as_it_is() {
        assert_eq!(
            totals(&[(MemoryType::Cpu, 4096), (MemoryType::CpuPinned, 8192)]),
            Some(vec![(MemoryType::Cpu, 4096), (MemoryType::CpuPinned, 8192)])
        );
    }

    #[test]
    fn entries_of_one_memory_type_are_summed() {
        assert_eq!(
            totals(&[
                (MemoryType::Cpu, 111),
                (MemoryType::CpuPinned, 7),
                (MemoryType::Cpu, 222),
            ]),
            Some(vec![(MemoryType::Cpu, 333), (MemoryType::CpuPinned, 7)])
        );
    }

    #[test]
    fn the_same_memory_type_on_another_device_stays_separate() {
        let first = MemoryType::Gpu { device_id: 0 };
        let second = MemoryType::Gpu { device_id: 1 };
        assert_eq!(
            totals(&[(first, 1), (second, 2), (first, 3)]),
            Some(vec![(first, 4), (second, 2)])
        );
    }

    #[test]
    fn a_sum_that_does_not_fit_u64_is_refused() {
        assert_eq!(
            totals(&[(MemoryType::Cpu, u64::MAX), (MemoryType::Cpu, 1)]),
            None
        );
    }
}
