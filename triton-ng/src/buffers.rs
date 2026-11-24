use crate::Error;
use crate::allocation::Allocation;
use std::ops::{Deref, DerefMut, Range};
use std::sync::Arc;

/// CPU memory from [`crate::MemoryManager::ram`], readable and writable as `[u8]`.
pub struct RamBuffer(pub(crate) Allocation);

/// Pinned CPU memory from [`crate::MemoryManager::pinned`], readable and writable
/// as `[u8]`.
pub struct PinnedRamBuffer(pub(crate) Allocation);

/// GPU memory a model has written its output into. It cannot be read on the
/// CPU: pass it as an input of the next BLS call, or turn it back into an
/// [`UninitGpuBuffer`] to receive another output.
pub struct GpuBuffer(pub(crate) Allocation);

/// GPU memory from [`crate::MemoryManager::gpu`] whose content is not defined yet.
/// It can only receive the output of a BLS call.
pub struct UninitGpuBuffer(pub(crate) Allocation);

/// A byte range of a GPU buffer, usable as a BLS input without copying.
///
/// Keeps the whole allocation alive until Triton releases the input. Created
/// by [`GpuBuffer::slice`]; it cannot mutate or receive outputs into the buffer.
pub struct GpuBufferSlice {
    pub(crate) buffer: Arc<GpuBuffer>,
    pub(crate) range: Range<usize>,
}

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
    /// Borrows a byte range as an owned BLS input, retaining this allocation.
    ///
    /// Fails when the range is reversed or exceeds the allocation. Offsets
    /// are bytes; callers must align ranges to their tensor element size.
    /// An outstanding slice keeps the buffer shared until request release.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let state = Arc::new(result.take_gpu("state_next")?);
    /// inference.append_input_buffer("state", state.slice(0..row_bytes)?)?;
    /// ```
    pub fn slice(self: &Arc<Self>, range: Range<usize>) -> Result<GpuBufferSlice, Error> {
        if range.start > range.end || range.end > self.len() {
            return Err(
                format!("GPU slice {range:?} exceeds buffer of {} bytes", self.len()).into(),
            );
        }
        Ok(GpuBufferSlice {
            buffer: Arc::clone(self),
            range,
        })
    }

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
    use super::{GpuBuffer, GpuBufferSlice};
    use crate::{Error, GPU, MemoryManager, Sealed};
    use std::ops::Range;
    use std::sync::Arc;

    #[test]
    fn gpu_slice_reports_the_requested_range_and_device() -> Result<(), Error> {
        let sut = Sut::new()?;
        let actual = sut.describe_slice(4..12)?;
        let expected = Placement {
            pointer_matches: true,
            len: 8,
            is_gpu: true,
            device_id: 3,
        };
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn gpu_slice_rejects_reversed_and_out_of_bounds_ranges() -> Result<(), Error> {
        let sut = Sut::new()?;
        let actual = sut.invalid_ranges_are_rejected();
        let expected = (true, true, true, 1);
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn gpu_slice_allows_an_empty_range_at_the_end() -> Result<(), Error> {
        let sut = Sut::new()?;
        let actual = sut.describe_slice(16..16)?;
        let expected = Placement {
            pointer_matches: true,
            len: 0,
            is_gpu: true,
            device_id: 3,
        };
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn gpu_slice_retains_the_allocation_after_the_source_is_dropped() -> Result<(), Error> {
        let sut = Sut::new()?;
        let actual = sut.observe_allocation_lifetime()?;
        let expected = (true, true);
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn gpu_slice_prevents_buffer_reuse_until_it_is_dropped() -> Result<(), Error> {
        let sut = Sut::new()?;
        let actual = sut.observe_buffer_reuse()?;
        let expected = (true, true);
        assert_eq!(actual, expected);
        Ok(())
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Placement {
        pointer_matches: bool,
        len: usize,
        is_gpu: bool,
        device_id: i64,
    }

    struct Sut {
        buffer: Arc<GpuBuffer>,
    }

    impl Sut {
        fn new() -> Result<Self, Error> {
            let memory = MemoryManager::heap();
            Ok(Self {
                buffer: Arc::new(GpuBuffer(memory.gpu(3, 16)?.0)),
            })
        }

        fn describe_slice(&self, range: Range<usize>) -> Result<Placement, Error> {
            let expected_ptr = self
                .buffer
                .0
                .ptr
                .cast::<u8>()
                .wrapping_add(range.start)
                .cast();
            let slice = self.buffer.slice(range)?;
            let raw = slice
                .buffers()
                .into_iter()
                .next()
                .ok_or("missing GPU slice buffer")?;
            Ok(Placement {
                pointer_matches: raw.ptr == expected_ptr,
                len: raw.len,
                is_gpu: raw.memory_type == GPU,
                device_id: raw.memory_type_id,
            })
        }

        fn invalid_ranges_are_rejected(&self) -> (bool, bool, bool, usize) {
            (
                self.buffer.slice(Range { start: 9, end: 8 }).is_err(),
                self.buffer.slice(0..17).is_err(),
                self.buffer.slice(17..17).is_err(),
                Arc::strong_count(&self.buffer),
            )
        }

        fn observe_allocation_lifetime(self) -> Result<(bool, bool), Error> {
            let weak = Arc::downgrade(&self.buffer);
            let slice: GpuBufferSlice = self.buffer.slice(4..12)?;
            drop(self.buffer);
            let retained = weak.strong_count() == 1;
            drop(slice);
            Ok((retained, weak.upgrade().is_none()))
        }

        fn observe_buffer_reuse(self) -> Result<(bool, bool), Error> {
            let slice = self.buffer.slice(4..12)?;
            match Arc::try_unwrap(self.buffer) {
                Ok(_) => Ok((false, false)),
                Err(buffer) => {
                    drop(slice);
                    Ok((true, Arc::try_unwrap(buffer).is_ok()))
                }
            }
        }
    }
}
