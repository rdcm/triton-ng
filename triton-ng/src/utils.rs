use crate::error::{Error, TritonError};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::ffi::{CString, c_char};
use std::io::Cursor;

/// Splits serialized BYTES tensor content (`u32` little-endian length + bytes
/// per element) into elements. A length running past the end is an error.
fn split_strings(data: &[u8]) -> impl Iterator<Item = Result<&[u8], std::io::Error>> {
    let mut cursor = Cursor::new(data);
    std::iter::from_fn(move || {
        let position = cursor.position() as usize;
        if position >= data.len() {
            return None;
        }
        Some(cursor.read_u32::<LittleEndian>().and_then(|len| {
            let start = position + 4;
            let end = start
                .checked_add(len as usize)
                .filter(|&end| end <= data.len())
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!("string of {len} bytes at offset {position} runs past the end"),
                    )
                })?;
            cursor.set_position(end as u64);
            Ok(&data[start..end])
        }))
    })
}

pub fn decode_string(data: &[u8]) -> Result<Vec<String>, std::io::Error> {
    split_strings(data)
        .map(|bytes| bytes.map(|b| String::from_utf8_lossy(b).into_owned()))
        .collect()
}

pub(crate) fn count_strings(data: &[u8]) -> Result<usize, std::io::Error> {
    split_strings(data).try_fold(0, |count, bytes| bytes.map(|_| count + 1))
}

pub fn encode_string(value: &str) -> Result<Vec<u8>, std::io::Error> {
    let len = u32::try_from(value.len()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "string of {} bytes does not fit a BYTES element",
                value.len()
            ),
        )
    })?;
    let mut bytes = vec![];
    bytes.write_u32::<LittleEndian>(len)?;
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

/// Decodes a buffer of little-endian fixed-size elements.
pub(crate) fn decode_le<T, const N: usize>(
    buffer: &[u8],
    from_le: fn([u8; N]) -> T,
) -> Result<Vec<T>, Error> {
    if !buffer.len().is_multiple_of(N) {
        return Err(format!(
            "tensor byte size {} is not a multiple of element size {N}",
            buffer.len()
        )
        .into());
    }

    Ok(buffer
        .as_chunks::<N>()
        .0
        .iter()
        .map(|&chunk| from_le(chunk))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{count_strings, decode_string, encode_string};

    fn encoded(values: &[&str]) -> Result<Vec<u8>, std::io::Error> {
        let mut data = Vec::new();
        for value in values {
            data.extend(encode_string(value)?);
        }
        Ok(data)
    }

    #[test]
    fn strings_round_trip() -> Result<(), std::io::Error> {
        let data = encoded(&["", "привет", "abc"])?;
        assert_eq!(decode_string(&data)?, vec!["", "привет", "abc"]);
        assert_eq!(count_strings(&data)?, 3);
        Ok(())
    }

    #[test]
    fn empty_content_has_no_strings() -> Result<(), std::io::Error> {
        assert!(decode_string(&[])?.is_empty());
        assert_eq!(count_strings(&[])?, 0);
        Ok(())
    }

    #[test]
    fn length_past_the_end_is_rejected() {
        let mut data = u32::MAX.to_le_bytes().to_vec();
        data.extend(b"abc");
        assert!(decode_string(&data).is_err());
        assert!(count_strings(&data).is_err());
    }

    #[test]
    fn truncated_length_is_rejected() {
        assert!(decode_string(&[1, 0]).is_err());
        assert!(count_strings(&[1, 0]).is_err());
    }
}
