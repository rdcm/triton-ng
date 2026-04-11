use crate::error::TritonError;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::ffi::{CString, c_char};
use std::io::{Cursor, Read};

pub fn decode_string(data: &[u8]) -> Result<Vec<String>, std::io::Error> {
    let mut strings = vec![];
    let mut cursor = Cursor::new(data);

    while cursor.position() < data.len() as u64 {
        let len = cursor.read_u32::<LittleEndian>()? as usize;
        let mut string_bytes = vec![0u8; len];
        cursor.read_exact(&mut string_bytes)?;
        strings.push(String::from_utf8_lossy(&string_bytes).to_string());
    }

    Ok(strings)
}

pub fn encode_string(value: &str) -> Result<Vec<u8>, std::io::Error> {
    let mut bytes = vec![];
    bytes.write_u32::<LittleEndian>(value.len() as u32)?;
    bytes.extend_from_slice(value.as_bytes());
    Ok(bytes)
}

#[allow(clippy::missing_safety_doc)]
pub unsafe fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }
}

pub fn cstring_from_str(s: &str) -> Result<CString, TritonError> {
    CString::new(s).map_err(|e| TritonError::from_message(e.to_string()))
}
