use crate::error::TritonError;
use crate::final_response::FinalResponse;
use crate::{CPU, CPU_PINNED, DataType, GPU, InputBuffer, InputProperties, RawBuffer, Sealed};
use std::sync::Arc;

/// An input tensor of a [`crate::Request`] passed on as a BLS input without a
/// copy, made by [`crate::Input::share`]. Pass it, or an `Arc` of it to use
/// it in several BLS requests, to
/// [`crate::InferenceRequest::add_input_with_buffer`].
///
/// Points into the memory the request keeps the tensor in, CPU or GPU, and
/// cannot read it: it is only a BLS input. While it is alive, the request is
/// not released and its final response is held back, see
/// [`crate::Input::share`].
///
/// Holds the right to answer and release the request, not the
/// `TRITONBACKEND_Input`: dropping the last shared input sends the final
/// response kept for it, then lets the request be released. `Send + Sync`.
///
/// # Examples
///
/// ```ignore
/// let features = Arc::new(request.get_input("features")?.share()?);
/// for model in ["decoder", "classifier"] {
///     let mut inference = InferenceRequest::new(ctx.server(), model, -1)?;
///     let shape = features.shape().to_vec();
///     inference.add_input_with_buffer("features", features.datatype(), &shape, Arc::clone(&features))?;
///     pending.push(ctx.server().infer_async(inference));
/// }
/// ```
pub struct SharedInput {
    name: String,
    datatype: DataType,
    shape: Vec<i64>,
    byte_size: usize,
    buffers: Vec<RawBuffer>,
    final_response: Arc<FinalResponse>,
}

// SAFETY: the buffers are read-only memory of the request, which stays alive
// and unchanged until `final_response` lets it go; they are never read or
// written here, only handed to Triton.
unsafe impl Send for SharedInput {}
unsafe impl Sync for SharedInput {}

impl SharedInput {
    pub(crate) fn new(
        properties: InputProperties,
        buffers: Vec<RawBuffer>,
        final_response: Arc<FinalResponse>,
    ) -> Result<Self, TritonError> {
        let (buffers, byte_size) = checked_buffers(&properties.name, properties.byte_size, buffers)
            .map_err(TritonError::from_message)?;
        final_response.share()?;
        Ok(Self {
            name: properties.name,
            datatype: properties.datatype,
            shape: properties.shape,
            byte_size,
            buffers,
            final_response,
        })
    }

    /// Name of the input in the request.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Datatype of the tensor.
    pub fn datatype(&self) -> DataType {
        self.datatype
    }

    /// Shape of the tensor, with the batch dimension of a batching model.
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Size of the tensor in bytes.
    pub fn byte_size(&self) -> usize {
        self.byte_size
    }
}

impl Sealed for SharedInput {
    fn buffers(&self) -> Vec<RawBuffer> {
        self.buffers.clone()
    }
}

impl InputBuffer for SharedInput {}

impl Drop for SharedInput {
    fn drop(&mut self) {
        self.final_response.unshare();
    }
}

/// The non-empty buffers of input `name` of `byte_size` bytes and that size,
/// or why Triton cannot read them as a BLS input.
fn checked_buffers(
    name: &str,
    byte_size: u64,
    buffers: Vec<RawBuffer>,
) -> Result<(Vec<RawBuffer>, usize), String> {
    let mut total: u128 = 0;
    let mut kept = Vec::with_capacity(buffers.len());
    for (index, buffer) in buffers.into_iter().enumerate() {
        if buffer.len == 0 {
            continue;
        }
        if buffer.ptr.is_null() {
            return Err(format!("input '{name}' buffer {index} is null"));
        }
        if ![CPU, CPU_PINNED, GPU].contains(&buffer.memory_type) {
            return Err(format!(
                "input '{name}' buffer {index} is in unknown memory type {}",
                buffer.memory_type
            ));
        }
        total += buffer.len as u128;
        kept.push(buffer);
    }

    if total != u128::from(byte_size) {
        return Err(format!(
            "input '{name}' has {total} bytes in its buffers, expected {byte_size}"
        ));
    }
    let byte_size = usize::try_from(byte_size)
        .map_err(|_| format!("input '{name}' of {byte_size} bytes does not fit in memory"))?;
    Ok((kept, byte_size))
}

#[cfg(test)]
mod tests {
    use super::checked_buffers;
    use crate::{CPU, GPU, RawBuffer};

    fn buffer(data: &[u8], memory_type: u32) -> RawBuffer {
        RawBuffer {
            ptr: data.as_ptr().cast_mut().cast(),
            len: data.len(),
            memory_type,
            memory_type_id: 0,
        }
    }

    #[test]
    fn every_non_empty_buffer_is_kept_in_order() {
        let (first, second) = ([1u8, 2], [3u8; 4]);
        let empty = RawBuffer {
            ptr: std::ptr::null_mut(),
            ..buffer(&[], CPU)
        };

        let (kept, byte_size) = checked_buffers(
            "x",
            6,
            vec![buffer(&first, GPU), empty, buffer(&second, CPU)],
        )
        .unwrap();

        assert_eq!(byte_size, 6);
        let kept: Vec<_> = kept.iter().map(|b| (b.ptr, b.memory_type)).collect();
        assert_eq!(
            kept,
            [
                (first.as_ptr().cast_mut().cast(), GPU),
                (second.as_ptr().cast_mut().cast(), CPU)
            ]
        );
    }

    #[test]
    fn empty_tensor_has_no_buffers() {
        let (kept, byte_size) = checked_buffers("x", 0, vec![]).unwrap();

        assert!(kept.is_empty());
        assert_eq!(byte_size, 0);
    }

    #[test]
    fn buffers_of_another_size_null_or_unknown_memory_are_refused() {
        let data = [0u8; 4];
        let null = RawBuffer {
            ptr: std::ptr::null_mut(),
            ..buffer(&data, CPU)
        };

        let errors = [
            checked_buffers("x", 8, vec![buffer(&data, CPU)]),
            checked_buffers("x", 4, vec![null]),
            checked_buffers("x", 4, vec![buffer(&data, 42)]),
        ]
        .map(|result| result.err().unwrap());

        assert!(errors[0].contains("has 4 bytes in its buffers, expected 8"));
        assert!(errors[1].contains("is null"));
        assert!(errors[2].contains("unknown memory type 42"));
    }
}
