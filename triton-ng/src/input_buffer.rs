use crate::{CPU, GpuBuffer, GpuBufferSlice, PinnedRamBuffer, RamBuffer, RawBuffer, Sealed};
use std::sync::Arc;

/// Data of a BLS input: `Vec<u8>`, [`RamBuffer`], [`PinnedRamBuffer`],
/// [`GpuBuffer`], [`GpuBufferSlice`], an input of the request being answered
/// ([`crate::SharedInput`]), or an `Arc` of any of them to keep using the
/// buffer, for example in a cache, while Triton reads it.
///
/// The request owns the buffer until Triton releases the request, which can
/// be after the response has arrived: an `Arc` passed as an input is not
/// unique again before that.
///
/// # Examples
///
/// ```ignore
/// let state: Arc<GpuBuffer> = Arc::new(result.take_gpu("ho")?);
/// inference.add_input_with_buffer("hi", DataType::Fp32, &[1, batch, 320], Arc::clone(&state))?;
/// ```
pub trait InputBuffer: Sealed + Send + 'static {}

impl Sealed for Vec<u8> {
    fn buffers(&self) -> Vec<RawBuffer> {
        vec![RawBuffer {
            ptr: self.as_ptr().cast_mut().cast(),
            len: self.len(),
            memory_type: CPU,
            memory_type_id: 0,
        }]
    }
}

impl InputBuffer for Vec<u8> {}

impl Sealed for RamBuffer {
    fn buffers(&self) -> Vec<RawBuffer> {
        vec![self.0.raw()]
    }
}

impl InputBuffer for RamBuffer {}

impl Sealed for PinnedRamBuffer {
    fn buffers(&self) -> Vec<RawBuffer> {
        vec![self.0.raw()]
    }
}

impl InputBuffer for PinnedRamBuffer {}

impl Sealed for GpuBuffer {
    fn buffers(&self) -> Vec<RawBuffer> {
        vec![self.0.raw()]
    }
}

impl InputBuffer for GpuBuffer {}

impl Sealed for GpuBufferSlice {
    fn buffers(&self) -> Vec<RawBuffer> {
        let mut raw = self.buffer.0.raw();
        raw.ptr = raw.ptr.cast::<u8>().wrapping_add(self.range.start).cast();
        raw.len = self.range.end - self.range.start;
        vec![raw]
    }
}

impl InputBuffer for GpuBufferSlice {}

impl<T: InputBuffer + Sync> Sealed for Arc<T> {
    fn buffers(&self) -> Vec<RawBuffer> {
        self.as_ref().buffers()
    }
}

impl<T: InputBuffer + Sync> InputBuffer for Arc<T> {}

#[cfg(test)]
mod tests {
    use crate::{CPU, CPU_PINNED, GPU, GpuBuffer, MemoryManager, RawBuffer, Sealed};
    use std::sync::Arc;

    #[test]
    fn input_buffers_report_their_memory_placement() {
        let memory = MemoryManager::heap();
        let vec = vec![0u8; 4];
        let ram = memory.ram(4).unwrap();
        let pinned = memory.pinned(4).unwrap();
        let gpu = Arc::new(GpuBuffer(memory.gpu(1, 4).unwrap().0));

        let placement = |buffers: Vec<RawBuffer>| {
            buffers
                .iter()
                .map(|raw| (raw.memory_type, raw.memory_type_id, raw.len))
                .collect::<Vec<_>>()
        };
        let expected = (
            vec![(CPU, 0, 4)],
            vec![(CPU, 0, 4)],
            vec![(CPU_PINNED, 0, 4)],
            vec![(GPU, 1, 4)],
        );
        let actual = (
            placement(vec.buffers()),
            placement(ram.buffers()),
            placement(pinned.buffers()),
            placement(gpu.buffers()),
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn shared_gpu_inputs_preserve_their_buffer_addresses() {
        let memory = MemoryManager::heap();
        let gpu = Arc::new(GpuBuffer(memory.gpu(1, 4).unwrap().0));

        let ptrs = |buffers: Vec<RawBuffer>| buffers.iter().map(|raw| raw.ptr).collect::<Vec<_>>();
        let actual = ptrs(Arc::clone(&gpu).buffers());
        let expected = ptrs(gpu.as_ref().buffers());
        assert_eq!(actual, expected);
    }
}
