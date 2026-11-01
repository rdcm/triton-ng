use crate::TritonError;
use crate::memory::{RawBuffer, WrittenBuffer};
use crate::response_allocator::OutputTag;
use crate::server::{OutputData, OutputTensor};
use crate::types::DataType;
use crate::utils::cstr_to_string;
use std::ffi::{c_char, c_void};
use std::ptr;
use std::slice;
use triton_ng_macros::triton_call;

pub struct InferenceResponse {
    ptr: *mut triton_ng_sys::TRITONSERVER_InferenceResponse,
}

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

        let data = if userp.is_null() {
            if memory_type != crate::memory::CPU {
                return Err(TritonError::from_message(format!(
                    "output '{name}' is not in CPU memory without a buffer"
                )));
            }
            OutputData::Host(if base.is_null() || byte_size == 0 {
                Vec::new()
            } else {
                unsafe { slice::from_raw_parts(base as *const u8, byte_size).to_vec() }
            })
        } else {
            // SAFETY: a non-null userp is the tag `alloc_fn` made for an
            // output buffer. Triton frees it only in the release callback,
            // when this response is deleted, so nothing else touches it now.
            let tag = unsafe { &mut *(userp as *mut OutputTag) };
            let placed = RawBuffer {
                ptr: base.cast_mut(),
                len: byte_size,
                memory_type,
                memory_type_id,
            };
            OutputData::Buffer(
                written_output(tag, &name, &placed).map_err(TritonError::from_message)?,
            )
        };

        Ok(OutputTensor {
            name,
            data,
            shape,
            datatype: DataType::from_sys(datatype),
        })
    }
}

/// Takes the buffer out of the tag of output `name`, checking that Triton
/// placed the output (`placed`) exactly in it.
fn written_output(
    tag: &mut OutputTag,
    name: &str,
    placed: &RawBuffer,
) -> Result<WrittenBuffer, String> {
    let buffer = tag
        .take()
        .ok_or_else(|| format!("buffer of output '{name}' is already taken"))?;
    let raw = buffer.raw();
    if raw.ptr != placed.ptr
        || raw.len != placed.len
        || raw.memory_type != placed.memory_type
        || raw.memory_type_id != placed.memory_type_id
    {
        return Err(format!("output '{name}' is not where its buffer is"));
    }
    Ok(buffer.into_written())
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
    use crate::memory::{CPU_PINNED, GPU, MemoryManager, OutputBuffer, RawBuffer, WrittenBuffer};

    fn tagged_gpu() -> (Option<OutputBuffer>, RawBuffer) {
        let buffer = OutputBuffer::from(MemoryManager::heap().gpu(0, 40).unwrap());
        let raw = buffer.raw();
        (Some(buffer), raw)
    }

    #[test]
    fn output_in_its_buffer_is_handed_over() {
        let (mut tag, placed) = tagged_gpu();

        let written = written_output(&mut tag, "ho", &placed).unwrap();

        assert!(matches!(written, WrittenBuffer::Gpu(buffer) if buffer.len() == 40));
        assert!(tag.is_none());
    }

    #[test]
    fn output_placed_elsewhere_is_refused() {
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
            assert_eq!(placed.memory_type, GPU);
            assert!(written_output(&mut tag, "ho", &move_output(placed)).is_err());
        }
    }

    #[test]
    fn taken_buffer_is_not_handed_over_again() {
        let (mut tag, placed) = tagged_gpu();
        written_output(&mut tag, "ho", &placed).unwrap();

        assert!(written_output(&mut tag, "ho", &placed).is_err());
    }
}
