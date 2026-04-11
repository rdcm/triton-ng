use crate::error::{Error, TritonError};
use crate::types::DataType;
use crate::utils::{cstr_to_string, cstring_from_str, decode_string};
use crate::{ensure_ptr, ffi_call};
use libc::{c_char, c_void};
use std::slice;

pub struct Request {
    ptr: *mut triton_sys::TRITONBACKEND_Request,
}

impl Request {
    pub fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_Request) -> Self {
        Self { ptr }
    }

    pub fn as_ptr(&self) -> *mut triton_sys::TRITONBACKEND_Request {
        self.ptr
    }

    pub fn input_names(&self) -> Result<Vec<String>, TritonError> {
        let mut count = 0u32;
        ffi_call!(triton_sys::TRITONBACKEND_RequestInputCount(
            self.ptr, &mut count
        ))?;
        (0..count)
            .map(|i| {
                let mut name: *const c_char = std::ptr::null();
                ffi_call!(triton_sys::TRITONBACKEND_RequestInputName(
                    self.ptr, i, &mut name
                ))?;
                Ok(unsafe { cstr_to_string(name) })
            })
            .collect()
    }

    pub fn output_names(&self) -> Result<Vec<String>, TritonError> {
        let mut count = 0u32;
        ffi_call!(triton_sys::TRITONBACKEND_RequestOutputCount(
            self.ptr, &mut count
        ))?;
        (0..count)
            .map(|i| {
                let mut name: *const c_char = std::ptr::null();
                ffi_call!(triton_sys::TRITONBACKEND_RequestOutputName(
                    self.ptr, i, &mut name
                ))?;
                Ok(unsafe { cstr_to_string(name) })
            })
            .collect()
    }

    pub fn get_input(&self, name: &str) -> Result<Input, TritonError> {
        let name = cstring_from_str(name)?;

        let mut input: *mut triton_sys::TRITONBACKEND_Input = std::ptr::null_mut();
        ffi_call!(triton_sys::TRITONBACKEND_RequestInput(
            self.ptr,
            name.as_ptr(),
            &mut input
        ))?;
        ensure_ptr!(input)?;

        Ok(Input::from_ptr(input))
    }
}

pub struct Input {
    ptr: *mut triton_sys::TRITONBACKEND_Input,
}

impl Input {
    pub fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_Input) -> Self {
        Self { ptr }
    }

    fn buffer(&self) -> Result<Vec<u8>, Error> {
        let mut buffer: *const c_void = std::ptr::null_mut();
        let index = 0;
        let mut memory_type: triton_sys::TRITONSERVER_MemoryType = 0;
        let mut memory_type_id = 0;
        let mut buffer_byte_size = 0;
        ffi_call!(triton_sys::TRITONBACKEND_InputBuffer(
            self.ptr,
            index,
            &mut buffer,
            &mut buffer_byte_size,
            &mut memory_type,
            &mut memory_type_id,
        ))?;

        let mem: &[u8] =
            unsafe { slice::from_raw_parts(buffer as *mut u8, buffer_byte_size as usize) };
        Ok(mem.to_vec())
    }

    pub fn as_string(&self) -> Result<String, Error> {
        let buffer = self.buffer()?;
        let strings = decode_string(&buffer)?;
        strings
            .into_iter()
            .next()
            .ok_or_else(|| "empty BYTES tensor".into())
    }

    pub fn as_u64(&self) -> Result<u64, Error> {
        let buffer = self.buffer()?;

        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&buffer);

        Ok(u64::from_le_bytes(bytes))
    }

    pub fn as_fp32_vec(&self) -> Result<Vec<f32>, Error> {
        let buffer = self.buffer()?;

        let count = buffer.len() / std::mem::size_of::<f32>();
        let mut result = Vec::with_capacity(count);

        for i in 0..count {
            let offset = i * 4;
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&buffer[offset..offset + 4]);
            result.push(f32::from_le_bytes(bytes));
        }

        Ok(result)
    }

    pub fn properties(&self) -> Result<InputProperties, Error> {
        let mut name = std::ptr::null();
        let mut datatype = 0u32;
        let mut shape_ptr: *const i64 = std::ptr::null();
        let mut dims_count = 0u32;
        let mut byte_size = 0u64;
        let mut buffer_count = 0u32;

        ffi_call!(triton_sys::TRITONBACKEND_InputProperties(
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
