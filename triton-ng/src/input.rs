use crate::utils::{cstr_to_string, cstring_from_str, decode_le, decode_string};
use crate::{CPU, CPU_PINNED, DataType, Error, RawBuffer, Request, SharedInput};
use libc::c_void;
use std::borrow::Cow;
use std::ffi::CStr;
use std::ptr;
use std::slice;
use triton_ng_macros::triton_call;

/// Input tensor of a [`Request`].
///
/// Borrows the `TRITONBACKEND_Input` from the request: Triton frees it
/// together with the request, so there is nothing to drop, and the lifetime
/// keeps it from outliving the request. Neither `Send` nor `Sync`, like the
/// borrow of the request it holds.
///
/// `P` is what the input is borrowed from: a [`Request`] when a backend reads
/// its own requests, a [`RequestView`](crate::RequestView) when a
/// [`BatchStrategy`](crate::BatchStrategy) decides on a batch Triton has not
/// formed yet. Only an input of a [`Request`] can be
/// [shared](Input::share) into a BLS request.
pub struct Input<'a, P = Request> {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Input,
    owner: &'a P,
}

impl<'a, P> Input<'a, P> {
    pub(crate) fn from_ptr(ptr: *mut triton_ng_sys::TRITONBACKEND_Input, owner: &'a P) -> Self {
        Self { ptr, owner }
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
    /// The result owns its name and shape. Fails on a Triton error or malformed
    /// metadata, including a shape inconsistent with the fixed-size datatype.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let shape = request.get_input("AUDIO")?.properties()?.shape;
    /// ```
    pub fn properties(&self) -> Result<InputProperties, Error> {
        self.read_properties(None)
    }

    /// Metadata for the input data Triton selects for `host_policy_name`.
    ///
    /// Copies the name and shape out of the request; the result owns its data.
    /// r25.09 falls back to the ordinary input when no data was supplied for
    /// that policy, including an unknown name. This does not allocate a NUMA
    /// copy or guarantee that the selected buffers are pinned host memory.
    ///
    /// Fails for a name containing NUL, a Triton error, or invalid metadata
    /// (null name/shape, unknown datatype, negative dimensions or a byte size
    /// inconsistent with a fixed-size datatype and shape).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let state = ctx.instance_state::<Buffers>()?;
    /// let policy = state.host_policy.as_ref().ok_or("missing host policy")?.name();
    /// let properties = request.get_input("AUDIO")?.properties_for_host_policy(policy)?;
    /// ```
    pub fn properties_for_host_policy(
        &self,
        host_policy_name: &str,
    ) -> Result<InputProperties, Error> {
        let policy = cstring_from_str(host_policy_name)?;
        self.read_properties(Some(&policy))
    }

    /// Copies one buffer selected for `host_policy_name` into owned host bytes.
    ///
    /// `index` addresses a buffer, not a tensor element. CPU and pinned CPU
    /// buffers are copied; GPU and unknown memory types are rejected. Triton
    /// falls back to the ordinary input if this policy has no data. The result
    /// holds no Triton pointer and can outlive the request.
    ///
    /// Fails for invalid metadata or policy names, an out-of-range index,
    /// Triton errors, a null nonempty buffer, invalid sizes or non-host memory.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let first = request.get_input("AUDIO")?.buffer_for_host_policy("pinned", 0)?;
    /// ```
    pub fn buffer_for_host_policy(
        &self,
        host_policy_name: &str,
        index: u32,
    ) -> Result<Vec<u8>, Error> {
        let policy = cstring_from_str(host_policy_name)?;
        let properties = self.read_properties(Some(&policy))?;
        let buffer = self.policy_buffer(&policy, &properties, index)?;
        // SAFETY: Input::DataBufferForHostPolicy returns request-owned memory
        // (r25.09 infer_request.cc). `self` keeps the request alive during the
        // copy; no pointer or borrowed slice escapes this call.
        Ok(unsafe { copy_host_buffers(&properties.name, buffer.len as u64, &[buffer]) }?)
    }

    /// Copies all buffers selected for `host_policy_name`, in order, into
    /// owned host bytes. The result can outlive the request.
    ///
    /// CPU and pinned CPU buffers are supported. GPU memory is not copied to
    /// the host. A policy without its own data uses the ordinary input; the
    /// policy name alone does not guarantee pinned or NUMA-local allocation.
    ///
    /// Fails for invalid metadata or names, Triton errors, non-host buffers,
    /// null nonempty buffers, invalid sizes or buffers whose total size does
    /// not match the selected input's metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let state = ctx.instance_state::<Buffers>()?;
    /// let policy = state.host_policy.as_ref().ok_or("missing host policy")?.name();
    /// let bytes = request.get_input("AUDIO")?.as_bytes_for_host_policy(policy)?;
    /// ```
    pub fn as_bytes_for_host_policy(&self, host_policy_name: &str) -> Result<Vec<u8>, Error> {
        let policy = cstring_from_str(host_policy_name)?;
        let properties = self.read_properties(Some(&policy))?;
        let buffers = (0..properties.buffer_count)
            .map(|index| self.policy_buffer(&policy, &properties, index))
            .collect::<Result<Vec<_>, _>>()?;
        // SAFETY: same request-owned buffers as buffer_for_host_policy; copy
        // them while the input is borrowed, before the request can be released.
        Ok(unsafe { copy_host_buffers(&properties.name, properties.byte_size, &buffers) }?)
    }

    /// Reads a policy buffer without exposing its pointer to backend code.
    fn policy_buffer(
        &self,
        policy: &CStr,
        properties: &InputProperties,
        index: u32,
    ) -> Result<RawBuffer, Error> {
        if index >= properties.buffer_count {
            return Err(format!(
                "input '{}' has {} buffers, index {index} is out of range",
                properties.name, properties.buffer_count
            )
            .into());
        }
        let mut buffer = ptr::null();
        let mut byte_size = 0;
        let mut memory_type = CPU;
        let mut memory_type_id = 0;
        triton_call!(triton_ng_sys::TRITONBACKEND_InputBufferForHostPolicy(
            self.ptr,
            policy.as_ptr(),
            index,
            &mut buffer,
            &mut byte_size,
            &mut memory_type,
            &mut memory_type_id,
        ))?;
        if byte_size > properties.byte_size {
            return Err(format!(
                "input '{}' buffer {index} exceeds the input byte size",
                properties.name
            )
            .into());
        }
        let len = usize::try_from(byte_size).map_err(|_| "input buffer size does not fit usize")?;
        Ok(RawBuffer {
            ptr: buffer.cast_mut(),
            len,
            memory_type,
            memory_type_id,
        })
    }

    /// Both C calls return pointers borrowed from the input, never allocations
    /// owned by this wrapper (backend_model.cc, InputPropertiesForHostPolicy).
    fn read_properties(&self, policy: Option<&CStr>) -> Result<InputProperties, Error> {
        let mut name = ptr::null();
        let mut datatype = 0u32;
        let mut shape_ptr: *const i64 = ptr::null();
        let mut dims_count = 0u32;
        let mut byte_size = 0u64;
        let mut buffer_count = 0u32;
        if let Some(policy) = policy {
            triton_call!(triton_ng_sys::TRITONBACKEND_InputPropertiesForHostPolicy(
                self.ptr,
                policy.as_ptr(),
                &mut name,
                &mut datatype,
                &mut shape_ptr,
                &mut dims_count,
                &mut byte_size,
                &mut buffer_count,
            ))?;
        } else {
            triton_call!(triton_ng_sys::TRITONBACKEND_InputProperties(
                self.ptr,
                &mut name,
                &mut datatype,
                &mut shape_ptr,
                &mut dims_count,
                &mut byte_size,
                &mut buffer_count,
            ))?;
        }
        if name.is_null() {
            return Err("Triton returned a null input name".into());
        }
        let rank = usize::try_from(dims_count).map_err(|_| "input rank does not fit usize")?;
        if rank > (isize::MAX as usize) / size_of::<i64>() {
            return Err("input shape exceeds Rust slice limits".into());
        }
        if rank != 0 && (shape_ptr.is_null() || !shape_ptr.is_aligned()) {
            return Err("Triton returned a null or unaligned input shape".into());
        }
        // SAFETY: the C API borrows a name and `dims_count` dimensions from the
        // request. Copy both while self keeps it alive, after checking null and
        // slice layout. A rank-zero shape needs no pointer dereference.
        let name = unsafe { cstr_to_string(name) };
        let shape = if rank == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(shape_ptr, rank).to_vec() }
        };
        let datatype = DataType::from_sys(datatype);
        check_metadata(datatype, &shape, byte_size)?;
        Ok(InputProperties {
            name,
            datatype,
            shape,
            dims_count,
            byte_size,
            buffer_count,
        })
    }
}

impl<'a> Input<'a, Request> {
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
            self.owner.final_response(),
        )?)
    }
}

#[derive(Debug)]
/// Owned metadata from [`Input::properties`] or
/// [`Input::properties_for_host_policy`].
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

/// Validates fixed-size tensor metadata without reaching any Triton symbols.
fn check_metadata(datatype: DataType, shape: &[i64], byte_size: u64) -> Result<(), String> {
    if datatype == DataType::Invalid {
        return Err("Triton returned an unknown input datatype".into());
    }
    let mut elements = 1u64;
    for &dim in shape {
        let dim = u64::try_from(dim).map_err(|_| "input shape contains a negative dimension")?;
        elements = elements
            .checked_mul(dim)
            .ok_or("input shape overflows u64")?;
    }
    if let Some(size) = datatype.element_size() {
        let expected = elements
            .checked_mul(size as u64)
            .ok_or("input byte size overflows u64")?;
        if expected != byte_size {
            return Err(format!(
                "input shape and datatype require {expected} bytes, got {byte_size}"
            ));
        }
    }
    Ok(())
}

/// Copies request-owned host buffers, including the single-buffer case.
///
/// # Safety
/// Every nonempty nonnull host buffer must be readable for this call.
unsafe fn copy_host_buffers(
    name: &str,
    byte_size: u64,
    buffers: &[RawBuffer],
) -> Result<Vec<u8>, String> {
    Ok(unsafe { contiguous(name, byte_size, buffers) }?.into_owned())
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
    let total: u128 = buffers.iter().map(|buffer| buffer.len as u128).sum();
    if total != u128::from(byte_size) {
        return Err(format!(
            "input '{name}' has {total} bytes in its buffers, expected {byte_size}"
        ));
    }
    if total > isize::MAX as u128 {
        return Err(format!("input '{name}' exceeds Rust slice limits"));
    }
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

    Ok(match parts.as_slice() {
        [] => Cow::Borrowed(&[]),
        [part] => Cow::Borrowed(part),
        _ => Cow::Owned(parts.concat()),
    })
}

#[cfg(test)]
mod tests {
    use super::{check_metadata, contiguous, copy_host_buffers};
    use crate::DataType;
    use crate::{CPU, CPU_PINNED, GPU, RawBuffer};
    use std::borrow::Cow;

    #[test]
    fn one_buffer_is_borrowed() {
        let data = [1u8, 2, 3, 4];

        let bytes = unsafe { contiguous("x", 4, &[empty(), cpu(&data), empty()]) }.unwrap();

        let actual_0 =
            matches!(&bytes, Cow::Borrowed(borrowed) if borrowed.as_ptr() == data.as_ptr());
        let expected_0 = true;
        let actual_1 = &bytes[..];
        let expected_1 = &data[..];

        let expected = (expected_0, expected_1);
        let actual = (actual_0, actual_1);
        assert_eq!(actual, expected);
    }

    #[test]
    fn pinned_buffer_is_borrowed() {
        let data = [5u8; 8];
        let pinned = RawBuffer {
            memory_type: CPU_PINNED,
            ..cpu(&data)
        };

        let bytes = unsafe { contiguous("x", 8, &[pinned]) }.unwrap();

        let actual = matches!(bytes, Cow::Borrowed(_));
        let expected = true;
        assert_eq!(actual, expected);
    }

    #[test]
    fn split_tensor_is_copied_in_order() {
        let (first, second) = ([1u8, 2], [3u8, 4, 5]);

        let bytes = unsafe { contiguous("x", 5, &[cpu(&first), cpu(&second)]) }.unwrap();

        let actual = matches!(&bytes, Cow::Owned(owned) if owned == &[1, 2, 3, 4, 5]);
        let expected = true;
        assert_eq!(actual, expected);
    }

    #[test]
    fn empty_tensor_has_no_bytes() {
        let bytes = unsafe { contiguous("x", 0, &[]) }.unwrap();

        let actual = bytes.is_empty();
        let expected = true;
        assert_eq!(actual, expected);
    }

    #[test]
    fn buffers_of_another_size_are_refused() {
        let data = [0u8; 4];

        let error = unsafe { contiguous("x", 8, &[cpu(&data)]) }.unwrap_err();

        let actual = error.contains("has 4 bytes in its buffers, expected 8");
        let expected = true;
        assert_eq!(actual, expected);
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

        let unknown = RawBuffer {
            memory_type: 42,
            ..cpu(&data)
        };
        let expected = (true, true, true);
        let actual = (
            unsafe { copy_host_buffers("x", 4, &[unknown]) }
                .unwrap_err()
                .contains("host memory"),
            unsafe { contiguous("x", 4, &[gpu]) }
                .unwrap_err()
                .contains("host memory"),
            unsafe { contiguous("x", 4, &[null]) }
                .unwrap_err()
                .contains("null"),
        );
        assert_eq!(actual, expected);
    }
    #[test]
    fn policy_copies_are_owned_for_single_and_split_host_buffers() {
        let mut first = [1, 2];
        let second = [3, 4];
        let single = unsafe { copy_host_buffers("x", 2, &[cpu(&first)]) }.unwrap();
        let split = unsafe {
            copy_host_buffers(
                "x",
                4,
                &[
                    cpu(&first),
                    RawBuffer {
                        memory_type: CPU_PINNED,
                        ..cpu(&second)
                    },
                ],
            )
        }
        .unwrap();
        first.fill(9);
        let expected = (vec![1, 2], vec![1, 2, 3, 4], true);
        let actual = (
            single,
            split,
            unsafe { copy_host_buffers("x", 0, &[]) }
                .unwrap()
                .is_empty(),
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn policy_input_metadata_rejects_invalid_shapes_and_sizes() {
        let expected = (true, true, true, true, true, true, true, true, true);
        let actual = (
            check_metadata(DataType::Fp32, &[1, 2], 8).is_ok(),
            check_metadata(DataType::Fp32, &[], 4).is_ok(),
            check_metadata(DataType::Fp32, &[0], 0).is_ok(),
            check_metadata(DataType::Bytes, &[2], 19).is_ok(),
            check_metadata(DataType::Invalid, &[1], 4).is_err(),
            check_metadata(DataType::Fp32, &[-1], 4).is_err(),
            check_metadata(DataType::Fp32, &[1], 8).is_err(),
            check_metadata(DataType::Fp64, &[i64::MAX, i64::MAX], 0).is_err(),
            check_metadata(DataType::Fp64, &[i64::MAX], 0).is_err(),
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn policy_input_buffers_reject_sizes_beyond_the_slice_limit() {
        let oversized = RawBuffer {
            len: (isize::MAX as usize) + 1,
            ..empty()
        };
        let actual = unsafe { copy_host_buffers("x", oversized.len as u64, &[oversized]) }
            .unwrap_err()
            .contains("slice limits");
        let expected = true;
        assert_eq!(actual, expected);
    }

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
}
