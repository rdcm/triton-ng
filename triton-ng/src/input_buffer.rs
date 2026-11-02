use crate::{CPU, GpuBuffer, PinnedRamBuffer, RamBuffer, RawBuffer, Sealed};
use std::sync::Arc;

/// Data of a BLS input: `Vec<u8>`, [`RamBuffer`], [`PinnedRamBuffer`],
/// [`GpuBuffer`], an input of the request being answered
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
    fn inputs_tell_triton_where_their_memory_is() {
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
        assert_eq!(placement(vec.buffers()), [(CPU, 0, 4)]);
        assert_eq!(placement(ram.buffers()), [(CPU, 0, 4)]);
        assert_eq!(placement(pinned.buffers()), [(CPU_PINNED, 0, 4)]);
        assert_eq!(placement(gpu.buffers()), [(GPU, 1, 4)]);
        let ptrs = |buffers: Vec<RawBuffer>| buffers.iter().map(|raw| raw.ptr).collect::<Vec<_>>();
        assert_eq!(
            ptrs(Arc::clone(&gpu).buffers()),
            ptrs(gpu.as_ref().buffers())
        );
    }
}
