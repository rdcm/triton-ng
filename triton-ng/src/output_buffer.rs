use crate::{GpuBuffer, PinnedRamBuffer, RamBuffer, RawBuffer, UninitGpuBuffer};

/// Memory a BLS output is written into, see
/// [`crate::InferenceRequest::add_requested_output_into`]. Its length must be
/// the exact size of the output.
pub enum OutputBuffer {
    Ram(RamBuffer),
    Pinned(PinnedRamBuffer),
    Gpu(UninitGpuBuffer),
}

impl From<RamBuffer> for OutputBuffer {
    fn from(buffer: RamBuffer) -> Self {
        Self::Ram(buffer)
    }
}

impl From<PinnedRamBuffer> for OutputBuffer {
    fn from(buffer: PinnedRamBuffer) -> Self {
        Self::Pinned(buffer)
    }
}

impl From<UninitGpuBuffer> for OutputBuffer {
    fn from(buffer: UninitGpuBuffer) -> Self {
        Self::Gpu(buffer)
    }
}

/// An [`OutputBuffer`] a model has written its output into.
pub(crate) enum WrittenBuffer {
    Ram(RamBuffer),
    Pinned(PinnedRamBuffer),
    Gpu(GpuBuffer),
}

impl OutputBuffer {
    pub(crate) fn raw(&self) -> RawBuffer {
        match self {
            Self::Ram(buffer) => buffer.0.raw(),
            Self::Pinned(buffer) => buffer.0.raw(),
            Self::Gpu(buffer) => buffer.0.raw(),
        }
    }

    pub(crate) fn into_written(self) -> WrittenBuffer {
        match self {
            Self::Ram(buffer) => WrittenBuffer::Ram(buffer),
            Self::Pinned(buffer) => WrittenBuffer::Pinned(buffer),
            Self::Gpu(buffer) => WrittenBuffer::Gpu(GpuBuffer(buffer.0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{OutputBuffer, WrittenBuffer};
    use crate::{GPU, MemoryManager};

    #[test]
    fn gpu_memory_is_reused_as_output() {
        let memory = MemoryManager::heap();
        let WrittenBuffer::Gpu(gpu) = OutputBuffer::from(memory.gpu(0, 32).unwrap()).into_written()
        else {
            unreachable!()
        };
        let ptr = gpu.0.raw().ptr;

        let reused = OutputBuffer::from(gpu.into_uninit());

        assert_eq!(reused.raw().ptr, ptr);
        assert_eq!(reused.raw().memory_type, GPU);
    }
}
