use crate::utils::{cstr_to_string, decode_le, decode_string};
use crate::{DataType, Error, Request};
use libc::c_void;
use std::marker::PhantomData;
use std::ptr;
use std::slice;
use triton_ng_macros::triton_call;

/// Input tensor of a [`Request`].
///
/// Borrows the `TRITONBACKEND_Input` from the request: Triton frees it
/// together with the request, so there is nothing to drop, and the lifetime
/// keeps it from outliving the request. Neither `Send` nor `Sync`, like the
/// borrow of the request it holds.
pub struct Input<'a> {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Input,
    request: PhantomData<&'a Request>,
}

impl Input<'_> {
    pub(crate) fn from_ptr(ptr: *mut triton_ng_sys::TRITONBACKEND_Input) -> Self {
        Self {
            ptr,
            request: PhantomData,
        }
    }

    /// Copies the tensor into a contiguous host buffer.
    ///
    /// Triton may split a tensor into several buffers (`buffer_count`),
    /// all of them are concatenated in order.
    fn buffer(&self) -> Result<Vec<u8>, Error> {
        let properties = self.properties()?;
        let mut data = Vec::with_capacity(properties.byte_size as usize);

        for index in 0..properties.buffer_count {
            let mut buffer: *const c_void = ptr::null();
            let mut memory_type: triton_ng_sys::TRITONSERVER_MemoryType =
                triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
            let mut memory_type_id = 0;
            let mut buffer_byte_size = 0;
            triton_call!(triton_ng_sys::TRITONBACKEND_InputBuffer(
                self.ptr,
                index,
                &mut buffer,
                &mut buffer_byte_size,
                &mut memory_type,
                &mut memory_type_id,
            ))?;

            if buffer_byte_size == 0 {
                continue;
            }

            if buffer.is_null() {
                return Err(format!("input '{}' buffer {index} is null", properties.name).into());
            }

            if memory_type != triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU
                && memory_type
                    != triton_ng_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU_PINNED
            {
                return Err(format!(
                    "input '{}' buffer {index} is not in host memory, only host memory is supported",
                    properties.name
                )
                .into());
            }

            let mem: &[u8] =
                unsafe { slice::from_raw_parts(buffer as *const u8, buffer_byte_size as usize) };
            data.extend_from_slice(mem);
        }

        if data.len() as u64 != properties.byte_size {
            return Err(format!(
                "input '{}' has {} bytes in its buffers, expected {}",
                properties.name,
                data.len(),
                properties.byte_size
            )
            .into());
        }

        Ok(data)
    }

    /// Raw little-endian tensor bytes.
    pub fn as_bytes(&self) -> Result<Vec<u8>, Error> {
        self.buffer()
    }

    /// First element of a BYTES tensor decoded as UTF-8 (lossy).
    pub fn as_string(&self) -> Result<String, Error> {
        let buffer = self.buffer()?;
        let strings = decode_string(&buffer)?;
        strings
            .into_iter()
            .next()
            .ok_or_else(|| "empty BYTES tensor".into())
    }

    /// All elements of a BYTES tensor decoded as UTF-8 (lossy).
    pub fn as_strings(&self) -> Result<Vec<String>, Error> {
        Ok(decode_string(&self.buffer()?)?)
    }

    pub fn as_u64(&self) -> Result<u64, Error> {
        let buffer = self.buffer()?;

        let bytes: [u8; 8] = buffer
            .as_slice()
            .try_into()
            .map_err(|_| format!("expected 8 bytes for UINT64 scalar, got {}", buffer.len()))?;

        Ok(u64::from_le_bytes(bytes))
    }

    pub fn as_fp32_vec(&self) -> Result<Vec<f32>, Error> {
        decode_le(&self.buffer()?, f32::from_le_bytes)
    }

    pub fn as_i32_vec(&self) -> Result<Vec<i32>, Error> {
        decode_le(&self.buffer()?, i32::from_le_bytes)
    }

    pub fn as_i64_vec(&self) -> Result<Vec<i64>, Error> {
        decode_le(&self.buffer()?, i64::from_le_bytes)
    }

    pub fn properties(&self) -> Result<InputProperties, Error> {
        let mut name = std::ptr::null();
        let mut datatype = 0u32;
        let mut shape_ptr: *const i64 = std::ptr::null();
        let mut dims_count = 0u32;
        let mut byte_size = 0u64;
        let mut buffer_count = 0u32;

        triton_call!(triton_ng_sys::TRITONBACKEND_InputProperties(
            self.ptr,
            &mut name,
            &mut datatype,
            &mut shape_ptr,
            &mut dims_count,
            &mut byte_size,
            &mut buffer_count,
        ))?;

        let name = unsafe { cstr_to_string(name) };

        let shape = if shape_ptr.is_null() || dims_count == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(shape_ptr, dims_count as usize).to_vec() }
        };

        Ok(InputProperties {
            name,
            datatype: DataType::from_sys(datatype),
            shape,
            dims_count,
            byte_size,
            buffer_count,
        })
    }
}

#[derive(Debug)]
pub struct InputProperties {
    pub name: String,
    pub datatype: DataType,
    pub shape: Vec<i64>,
    pub dims_count: u32,
    pub byte_size: u64,
    pub buffer_count: u32,
}
