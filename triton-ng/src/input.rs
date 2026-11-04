use crate::utils::{cstr_to_string, decode_le, decode_string};
use crate::{CPU, CPU_PINNED, DataType, Error, RawBuffer, Request, SharedInput};
use libc::c_void;
use std::borrow::Cow;
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
    request: &'a Request,
}

impl<'a> Input<'a> {
    pub(crate) fn from_ptr(
        ptr: *mut triton_ng_sys::TRITONBACKEND_Input,
        request: &'a Request,
    ) -> Self {
        Self { ptr, request }
    }

    /// The bytes of the tensor, borrowed from the request when Triton keeps
    /// them in one CPU buffer (the usual case), copied into one when the
    /// tensor is split over several buffers.
    ///
    /// Fails when a buffer of the tensor is in GPU memory, which the CPU
    /// cannot read, or the buffers do not add up to the size of the tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let image = request.get_input("Input3")?.bytes()?;
    /// let mut pinned = ctx.memory().pinned(image.len())?;
    /// pinned.copy_from_slice(&image);
    /// ```
    pub fn bytes(&self) -> Result<Cow<'a, [u8]>, Error> {
        let properties = self.properties()?;
        let buffers = self.buffers(&properties)?;

        // SAFETY: Triton keeps the input buffers of a request unchanged until
        // the request is released, and `'a` keeps the request alive.
        Ok(unsafe { contiguous(&properties.name, properties.byte_size, &buffers) }?)
    }

    /// The tensor as a BLS input that Triton reads where the request keeps
    /// it, in CPU or GPU memory, without copying it, see [`SharedInput`].
    ///
    /// The request is answered and released only after every shared input
    /// is dropped: the data of a client request lives until its final
    /// response is sent, so a final response given before (by
    /// [`crate::Response::send`], `respond_*`, [`crate::ResponseSender::finish`]
    /// or the drop of the request) is held back and sent by the last shared
    /// input dropped. A BLS request owns its inputs until Triton releases it,
    /// which can be after its result has arrived.
    ///
    /// Fails when the buffers do not add up to the size of the tensor, a
    /// non-empty buffer is null, or the request already has a final response.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let features = request.get_input("features")?.share()?;
    /// let mut inference = InferenceRequest::new(ctx.server(), "decoder", -1)?;
    /// let (datatype, shape) = (features.datatype(), features.shape().to_vec());
    /// inference.add_input_with_buffer("features", datatype, &shape, features)?;
    /// let result = ctx.server().infer_async(inference).await?;
    /// ```
    pub fn share(&self) -> Result<SharedInput, Error> {
        let properties = self.properties()?;
        let buffers = self.buffers(&properties)?;
        Ok(SharedInput::new(
            properties,
            buffers,
            self.request.final_response(),
        )?)
    }

    /// The buffers of the tensor as Triton reports them.
    fn buffers(&self, properties: &InputProperties) -> Result<Vec<RawBuffer>, Error> {
        let mut buffers = Vec::with_capacity(properties.buffer_count as usize);
        for index in 0..properties.buffer_count {
            let mut buffer: *const c_void = ptr::null();
            let mut memory_type: triton_ng_sys::TRITONSERVER_MemoryType = CPU;
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
            let len = usize::try_from(buffer_byte_size).map_err(|_| {
                format!(
                    "input '{}' buffer {index} of {buffer_byte_size} bytes does not fit in memory",
                    properties.name
                )
            })?;
            buffers.push(RawBuffer {
                ptr: buffer.cast_mut(),
                len,
                memory_type,
                memory_type_id,
            });
        }
        Ok(buffers)
    }

    /// The bytes of the tensor in a `Vec`, see [`Input::bytes`].
    pub fn as_bytes(&self) -> Result<Vec<u8>, Error> {
        Ok(self.bytes()?.into_owned())
    }

    /// First element of a BYTES tensor decoded as UTF-8 (lossy).
    pub fn as_string(&self) -> Result<String, Error> {
        let strings = decode_string(&self.bytes()?)?;
        strings
            .into_iter()
            .next()
            .ok_or_else(|| "empty BYTES tensor".into())
    }

    /// All elements of a BYTES tensor decoded as UTF-8 (lossy).
    pub fn as_strings(&self) -> Result<Vec<String>, Error> {
        Ok(decode_string(&self.bytes()?)?)
    }

    /// The one element of a UINT64 input.
    ///
    /// Fails when the input does not hold exactly 8 bytes.
    pub fn as_u64(&self) -> Result<u64, Error> {
        let buffer = self.bytes()?;

        let bytes: [u8; 8] = buffer
            .as_ref()
            .try_into()
            .map_err(|_| format!("expected 8 bytes for UINT64 scalar, got {}", buffer.len()))?;

        Ok(u64::from_le_bytes(bytes))
    }

    /// The elements of an FP32 input.
    ///
    /// Fails when the bytes do not divide into elements or the input is in GPU memory.
    pub fn as_fp32_vec(&self) -> Result<Vec<f32>, Error> {
        decode_le(&self.bytes()?, f32::from_le_bytes)
    }

    /// The elements of an INT32 input.
    ///
    /// Fails when the bytes do not divide into elements or the input is in GPU memory.
    pub fn as_i32_vec(&self) -> Result<Vec<i32>, Error> {
        decode_le(&self.bytes()?, i32::from_le_bytes)
    }

    /// The elements of an INT64 input.
    ///
    /// Fails when the bytes do not divide into elements or the input is in GPU memory.
    pub fn as_i64_vec(&self) -> Result<Vec<i64>, Error> {
        decode_le(&self.bytes()?, i64::from_le_bytes)
    }

    /// Name, datatype, shape and size of the input, see [`InputProperties`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let shape = request.get_input("AUDIO")?.properties()?.shape;
    /// ```
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
/// What Triton knows about an input, from [`Input::properties`].
pub struct InputProperties {
    /// Name of the input.
    pub name: String,
    /// Element type of the input.
    pub datatype: DataType,
    /// Shape of the input, batch dimension included.
    pub shape: Vec<i64>,
    /// Number of dimensions, the length of `shape`.
    pub dims_count: u32,
    /// Total size of the data, in bytes.
    pub byte_size: u64,
    /// Number of buffers the data is split into.
    pub buffer_count: u32,
}

/// The bytes of input `name` of `byte_size` bytes kept in `buffers`, borrowed
/// when one buffer holds all of them. Empty buffers are skipped.
///
/// # Safety
///
/// Every non-empty, non-null buffer must be readable for `'a`.
unsafe fn contiguous<'a>(
    name: &str,
    byte_size: u64,
    buffers: &[RawBuffer],
) -> Result<Cow<'a, [u8]>, String> {
    let mut parts: Vec<&'a [u8]> = Vec::with_capacity(buffers.len());
    for (index, buffer) in buffers.iter().enumerate() {
        if buffer.len == 0 {
            continue;
        }
        if buffer.ptr.is_null() {
            return Err(format!("input '{name}' buffer {index} is null"));
        }
        if buffer.memory_type != CPU && buffer.memory_type != CPU_PINNED {
            return Err(format!(
                "input '{name}' buffer {index} is not in host memory, only host memory is supported"
            ));
        }
        // SAFETY: the caller guarantees the buffer is readable for `'a`.
        parts.push(unsafe { slice::from_raw_parts(buffer.ptr.cast::<u8>(), buffer.len) });
    }

    let total: u128 = parts.iter().map(|part| part.len() as u128).sum();
    if total != u128::from(byte_size) {
        return Err(format!(
            "input '{name}' has {total} bytes in its buffers, expected {byte_size}"
        ));
    }

    Ok(match parts.as_slice() {
        [] => Cow::Borrowed(&[]),
        [part] => Cow::Borrowed(part),
        _ => Cow::Owned(parts.concat()),
    })
}

#[cfg(test)]
mod tests {
    use super::contiguous;
    use crate::{CPU, CPU_PINNED, GPU, RawBuffer};
    use std::borrow::Cow;

    fn cpu(data: &[u8]) -> RawBuffer {
        RawBuffer {
            ptr: data.as_ptr().cast_mut().cast(),
            len: data.len(),
            memory_type: CPU,
            memory_type_id: 0,
        }
    }

    fn empty() -> RawBuffer {
        RawBuffer {
            ptr: std::ptr::null_mut(),
            len: 0,
            memory_type: GPU,
            memory_type_id: 0,
        }
    }

    #[test]
    fn one_buffer_is_borrowed() {
        let data = [1u8, 2, 3, 4];

        let bytes = unsafe { contiguous("x", 4, &[empty(), cpu(&data), empty()]) }.unwrap();

        assert!(matches!(&bytes, Cow::Borrowed(borrowed) if borrowed.as_ptr() == data.as_ptr()));
        assert_eq!(&bytes[..], &data);
    }

    #[test]
    fn pinned_buffer_is_borrowed() {
        let data = [5u8; 8];
        let pinned = RawBuffer {
            memory_type: CPU_PINNED,
            ..cpu(&data)
        };

        let bytes = unsafe { contiguous("x", 8, &[pinned]) }.unwrap();

        assert!(matches!(bytes, Cow::Borrowed(_)));
    }

    #[test]
    fn split_tensor_is_copied_in_order() {
        let (first, second) = ([1u8, 2], [3u8, 4, 5]);

        let bytes = unsafe { contiguous("x", 5, &[cpu(&first), cpu(&second)]) }.unwrap();

        assert!(matches!(&bytes, Cow::Owned(owned) if owned == &[1, 2, 3, 4, 5]));
    }

    #[test]
    fn empty_tensor_has_no_bytes() {
        let bytes = unsafe { contiguous("x", 0, &[]) }.unwrap();

        assert!(bytes.is_empty());
    }

    #[test]
    fn buffers_of_another_size_are_refused() {
        let data = [0u8; 4];

        let error = unsafe { contiguous("x", 8, &[cpu(&data)]) }.unwrap_err();

        assert!(
            error.contains("has 4 bytes in its buffers, expected 8"),
            "{error}"
        );
    }

    #[test]
    fn gpu_and_null_buffers_are_refused() {
        let data = [0u8; 4];
        let gpu = RawBuffer {
            memory_type: GPU,
            ..cpu(&data)
        };
        let null = RawBuffer {
            ptr: std::ptr::null_mut(),
            ..cpu(&data)
        };

        assert!(
            unsafe { contiguous("x", 4, &[gpu]) }
                .unwrap_err()
                .contains("host memory")
        );
        assert!(
            unsafe { contiguous("x", 4, &[null]) }
                .unwrap_err()
                .contains("null")
        );
    }
}
