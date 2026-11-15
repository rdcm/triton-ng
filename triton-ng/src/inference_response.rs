use crate::RawBuffer;
use crate::TritonError;
use crate::inference_result::{OutputData, OutputTensor, ResponseParameters};
use crate::response_allocator::{OutputMemory, OutputTag};
use crate::types::{DataType, Parameter};
use crate::utils::cstr_to_string;
use std::ffi::{c_char, c_void};
use std::ptr;
use std::slice;
use triton_ng_macros::triton_call;

/// A response to a BLS request, read in the response callback.
///
/// Owns the `TRITONSERVER_InferenceResponse` and deletes it on drop, which
/// releases its output buffers. `Send`, not `Sync`: reading the outputs
/// takes the buffers out of their allocator tags.
pub(crate) struct InferenceResponse {
    ptr: *mut triton_ng_sys::TRITONSERVER_InferenceResponse,
}

// SAFETY: Triton delivers the response on one of its threads and lets it be
// deleted from any thread; output buffers are released through
// `response_allocator`, whose callbacks are thread-safe.
unsafe impl Send for InferenceResponse {}

impl InferenceResponse {
    pub(crate) fn from_ptr(
        ptr: *mut triton_ng_sys::TRITONSERVER_InferenceResponse,
    ) -> Result<Self, TritonError> {
        if ptr.is_null() {
            return Err(TritonError::from_message("null inference response"));
        }
        Ok(Self { ptr })
    }

    pub fn error(&self) -> Option<TritonError> {
        let error_ptr = unsafe { triton_ng_sys::TRITONSERVER_InferenceResponseError(self.ptr) };

        if error_ptr.is_null() {
            None
        } else {
            Some(unsafe { TritonError::new(error_ptr) })
        }
    }

    pub fn outputs(&self) -> Result<Vec<OutputTensor>, TritonError> {
        let mut output_count: u32 = 0;
        triton_call!(triton_ng_sys::TRITONSERVER_InferenceResponseOutputCount(
            self.ptr,
            &mut output_count,
        ))?;

        (0..output_count).map(|i| self.get_output(i)).collect()
    }

    /// The parameters of the response.
    pub fn parameters(&self) -> Result<ResponseParameters, TritonError> {
        let mut count: u32 = 0;
        triton_call!(triton_ng_sys::TRITONSERVER_InferenceResponseParameterCount(
            self.ptr, &mut count,
        ))?;

        (0..count)
            .map(|index| {
                let mut name: *const c_char = ptr::null();
                let mut kind: triton_ng_sys::TRITONSERVER_ParameterType = 0;
                let mut value: *const c_void = ptr::null();
                triton_call!(triton_ng_sys::TRITONSERVER_InferenceResponseParameter(
                    self.ptr, index, &mut name, &mut kind, &mut value,
                ))?;
                if name.is_null() {
                    return Err(TritonError::from_message(format!(
                        "response parameter {index} has no name"
                    )));
                }
                // SAFETY: Triton returns `value` matching `kind`; it and the
                // name are owned by the response, which outlives the call,
                // and both are copied here.
                let name = unsafe { cstr_to_string(name) };
                let value = unsafe { Parameter::from_sys(kind, value) };
                Ok((name, value))
            })
            .collect()
    }

    fn get_output(&self, index: u32) -> Result<OutputTensor, TritonError> {
        let mut name_ptr: *const c_char = ptr::null();
        let mut datatype: triton_ng_sys::TRITONSERVER_DataType = 0;
        let mut shape_ptr: *const i64 = ptr::null();
        let mut dim_count: u64 = 0;
        let mut base: *const c_void = ptr::null();
        let mut byte_size: usize = 0;
        let mut memory_type: triton_ng_sys::TRITONSERVER_MemoryType = 0;
        let mut memory_type_id: i64 = 0;
        let mut userp: *mut c_void = ptr::null_mut();

        triton_call!(triton_ng_sys::TRITONSERVER_InferenceResponseOutput(
            self.ptr,
            index,
            &mut name_ptr,
            &mut datatype,
            &mut shape_ptr,
            &mut dim_count,
            &mut base,
            &mut byte_size,
            &mut memory_type,
            &mut memory_type_id,
            &mut userp,
        ))?;

        let name = if name_ptr.is_null() {
            format!("output_{index}")
        } else {
            unsafe { cstr_to_string(name_ptr) }
        };

        let shape = if shape_ptr.is_null() || dim_count == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(shape_ptr, dim_count as usize).to_vec() }
        };

        if userp.is_null() {
            return Err(TritonError::from_message(format!(
                "output '{name}' is not in memory from the allocator"
            )));
        }
        // SAFETY: a non-null userp is the tag `alloc_fn` made for the output.
        // Triton frees it only in the release callback, when this response
        // is deleted, so nothing else touches it now.
        let tag = unsafe { &mut *(userp as *mut OutputTag) };
        let placed = RawBuffer {
            ptr: base.cast_mut(),
            len: byte_size,
            memory_type,
            memory_type_id,
        };
        let data = written_output(tag, &name, &placed).map_err(TritonError::from_message)?;

        Ok(OutputTensor {
            name,
            data,
            shape,
            datatype: DataType::from_sys(datatype),
        })
    }
}

/// Takes the memory out of the tag of output `name`, checking that Triton
/// placed the output (`placed`) exactly in it.
fn written_output(
    tag: &mut OutputTag,
    name: &str,
    placed: &RawBuffer,
) -> Result<OutputData, String> {
    let memory = tag
        .take()
        .ok_or_else(|| format!("memory of output '{name}' is already taken"))?;
    let raw = memory.raw();
    if raw.ptr != placed.ptr
        || raw.len != placed.len
        || raw.memory_type != placed.memory_type
        || raw.memory_type_id != placed.memory_type_id
    {
        return Err(format!("output '{name}' is not where its memory is"));
    }
    Ok(match memory {
        OutputMemory::Host(data) => OutputData::Host(data),
        OutputMemory::Buffer(buffer) => OutputData::Buffer(buffer.into_written()),
    })
}

impl Drop for InferenceResponse {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                triton_ng_sys::TRITONSERVER_InferenceResponseDelete(self.ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::written_output;
    use crate::inference_result::OutputData;
    use crate::response_allocator::{OutputMemory, OutputTag};
    use crate::{CPU, CPU_PINNED, GPU, MemoryManager, OutputBuffer, RawBuffer, WrittenBuffer};

    #[test]
    fn output_in_its_buffer_is_handed_over() {
        let (mut tag, placed) = tagged_gpu();

        let written = written_output(&mut tag, "ho", &placed).unwrap();

        let actual_0 =
            matches!(written, OutputData::Buffer(WrittenBuffer::Gpu(buffer)) if buffer.len() == 40);
        let expected_0 = true;
        let actual_1 = tag.is_none();
        let expected_1 = true;

        let expected = (expected_0, expected_1);
        let actual = (actual_0, actual_1);
        assert_eq!(actual, expected);
    }

    #[test]
    fn host_output_is_handed_over_without_a_copy() {
        let (mut tag, placed) = tagged_host(40);

        let OutputData::Host(data) = written_output(&mut tag, "dec", &placed).unwrap() else {
            panic!("host output is not in host memory");
        };

        let expected = (placed.ptr.cast_const().cast(), vec![7; 40], CPU, true);
        let actual = (data.as_ptr(), data, placed.memory_type, tag.is_none());
        assert_eq!(actual, expected);
    }

    #[test]
    fn empty_host_output_has_no_memory() {
        let (mut tag, placed) = tagged_host(0);

        let expected = (true, true);
        let actual = (
            placed.ptr.is_null(),
            matches!(written_output(&mut tag, "dec", &placed), Ok(OutputData::Host(data)) if data.is_empty()),
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn output_placed_elsewhere_is_refused() {
        let mut actual_0 = Vec::new();
        let mut expected_0 = Vec::new();
        let mut actual_1 = Vec::new();
        let mut expected_1 = Vec::new();
        let mut actual_2 = Vec::new();
        let mut expected_2 = Vec::new();

        let elsewhere: [fn(RawBuffer) -> RawBuffer; 4] = [
            |placed| RawBuffer { len: 4, ..placed },
            |placed| RawBuffer {
                memory_type: CPU_PINNED,
                ..placed
            },
            |placed| RawBuffer {
                memory_type_id: 1,
                ..placed
            },
            |placed| RawBuffer {
                ptr: std::ptr::null_mut(),
                ..placed
            },
        ];

        for move_output in elsewhere {
            let (mut tag, placed) = tagged_gpu();
            actual_0.push(placed.memory_type);
            expected_0.push(GPU);
            actual_1.push(written_output(&mut tag, "ho", &move_output(placed)).is_err());
            expected_1.push(true);

            let (mut tag, placed) = tagged_host(40);
            actual_2.push(written_output(&mut tag, "dec", &move_output(placed)).is_err());
            expected_2.push(true);
        }
        let expected = (expected_0, expected_1, expected_2);
        let actual = (actual_0, actual_1, actual_2);
        assert_eq!(actual, expected);
    }

    #[test]
    fn taken_memory_is_not_handed_over_again() {
        let (mut tag, placed) = tagged_gpu();
        written_output(&mut tag, "ho", &placed).unwrap();

        let actual = written_output(&mut tag, "ho", &placed).is_err();
        let expected = true;
        assert_eq!(actual, expected);
    }

    fn tagged_gpu() -> (OutputTag, RawBuffer) {
        let buffer = OutputBuffer::from(MemoryManager::heap().gpu(0, 40).unwrap());
        let raw = buffer.raw();
        (Some(OutputMemory::Buffer(buffer)), raw)
    }

    fn tagged_host(len: usize) -> (OutputTag, RawBuffer) {
        let memory = OutputMemory::Host(vec![7; len]);
        let raw = memory.raw();
        (Some(memory), raw)
    }
}
