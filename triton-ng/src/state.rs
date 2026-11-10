//! State a backend keeps between the requests of a sequence.

use crate::memory_manager::CPU;
use crate::types::MemoryType;
use crate::utils::cstring_from_str;
use crate::{DataType, Request, TritonError};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;
use triton_ng_macros::triton_call;

/// The state `name` of a request of a model with `sequence_batching` and a
/// `state` section in its configuration.
///
/// Borrows the `TRITONBACKEND_State` from the request it belongs to: Triton
/// owns it inside the sequence states of the request and frees it with the
/// request, there is no call that frees it. The borrow is what keeps the state
/// from outliving the request: answering, rescheduling or dropping the request
/// consumes it, so a state that is still held cannot be used after that, and
/// the compiler says so.
///
/// [`State::update`] consumes the state, because Triton documents the state as
/// valid only until the update: the data written into it belongs to the
/// sequence afterwards, and the next request of the same sequence reads it as
/// its input state, as the input its configuration pairs with the output.
///
/// Neither `Send` nor `Sync`, like the borrow of the request it holds.
///
/// # Examples
///
/// ```ignore
/// let mut state = State::new(&request, "counter", DataType::Fp32, &[1])?;
/// let previous = request.get_input("counter")?.as_fp32_vec()?;
/// state.write_bytes(&(previous[0] + 1.0).to_ne_bytes())?;
/// state.update()?;
/// ```
pub struct State<'a> {
    ptr: *mut triton_ng_sys::TRITONBACKEND_State,
    /// The size Triton resizes the state buffer to, from `datatype` and `shape`.
    byte_size: u64,
    /// Holds the request the state lives in, for as long as the state is used.
    _request: &'a Request,
}

impl<'a> State<'a> {
    /// The state `name` of `request`, a tensor of `datatype` and `shape`
    /// without the batch dimension, like the `state` entry of the model
    /// configuration.
    ///
    /// `name` is the `output_name` of that entry: the input state of the
    /// sequence reaches the backend as an input under its `input_name`, which
    /// the configuration pairs with it.
    ///
    /// Fails when the model has no sequence batching or no state of that name,
    /// Triton has nothing to create the state on, `name` has a NUL byte, the
    /// shape has a negative dimension, or `datatype` is BYTES, whose size
    /// Triton cannot give.
    ///
    /// The datatype and the shape are the ones Triton uses for the state from
    /// here on, so they have to be the ones of the model configuration: a
    /// mismatch is not detected, Triton would hand the wrong number of bytes to
    /// the next request of the sequence. The configuration also has to declare
    /// an `initial_state`: without one Triton allocates the state memory and
    /// leaves it uninitialized.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut state = State::new(&request, "counter", DataType::Fp32, &[1])?;
    /// state.write_bytes(&0.0_f32.to_ne_bytes())?;
    /// state.update()?;
    /// ```
    pub fn new(
        request: &'a Request,
        name: &str,
        datatype: DataType,
        shape: &[i64],
    ) -> Result<Self, TritonError> {
        let name_cstr = cstring_from_str(name)?;
        let byte_size = state_byte_size(datatype, shape)
            .map_err(|e| TritonError::from_message(format!("state '{name}' {e}")))?;
        let dims_count = u32::try_from(shape.len()).map_err(|_| {
            TritonError::from_message(format!(
                "state '{name}' has more dimensions than Triton takes"
            ))
        })?;

        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_StateNew(
            &mut _,
            request.as_ptr(),
            name_cstr.as_ptr(),
            datatype.to_sys(),
            shape.as_ptr(),
            dims_count,
        ))?;

        Ok(Self {
            ptr,
            byte_size,
            _request: request,
        })
    }

    /// The buffer Triton keeps for the state, asked for in `memory_type`, see
    /// [`StateBuffer`].
    ///
    /// `memory_type` is what the state is asked for, not what it gets: Triton
    /// reuses the buffer it has when its size matches, and allocates a new one
    /// otherwise, falling back from GPU memory to pinned host memory to plain
    /// host memory. The memory type the state is really in travels with the
    /// buffer.
    ///
    /// Fails when Triton cannot resize the buffer, returns no buffer at all, or
    /// names a memory type this version does not know.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut state = State::new(&request, "counter", DataType::Fp32, &[1])?;
    /// let values = state.buffer(MemoryType::Cpu)?.as_mut_slice().ok_or("the state is on the GPU")?;
    /// values[..4].copy_from_slice(&7.0_f32.to_ne_bytes());
    /// state.update()?;
    /// ```
    pub fn buffer(&mut self, memory_type: MemoryType) -> Result<StateBuffer<'_>, TritonError> {
        let (mut kind, mut memory_type_id) = memory_type.to_sys();
        let mut buffer: *mut c_void = ptr::null_mut();
        triton_call!(triton_ng_sys::TRITONBACKEND_StateBuffer(
            self.ptr,
            &mut buffer,
            self.byte_size,
            &mut kind,
            &mut memory_type_id,
        ))?;
        if buffer.is_null() {
            return Err(TritonError::from_message(
                "Triton returned a null state buffer",
            ));
        }
        let memory_type =
            MemoryType::from_sys(kind, memory_type_id).map_err(TritonError::from_message)?;
        let len = usize::try_from(self.byte_size)
            .map_err(|_| TritonError::from_message("the state is larger than usize bytes"))?;
        Ok(StateBuffer {
            ptr: buffer,
            len,
            memory_type,
            _state: PhantomData,
        })
    }

    /// Copies `bytes` into the state buffer: exactly as many bytes as the state
    /// holds, which is the size of its datatype over its shape.
    ///
    /// The buffer is asked for in CPU memory. Triton allocates host memory
    /// through its pinned pool, so the buffer can come back pinned, which is
    /// host memory as well and is copied into the same way; a state in GPU
    /// memory is refused, since a copy from the host cannot fill it. Use
    /// [`State::buffer`] for those.
    ///
    /// Fails when `bytes` is not the size of the state, when Triton cannot
    /// resize the buffer, or when the buffer Triton returns is not host memory.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut state = State::new(&request, "counter", DataType::Fp32, &[1])?;
    /// state.write_bytes(&7.0_f32.to_ne_bytes())?;
    /// state.update()?;
    /// ```
    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), TritonError> {
        let len = u64::try_from(bytes.len())
            .map_err(|_| TritonError::from_message("the state data is larger than u64 bytes"))?;
        if len != self.byte_size {
            return Err(TritonError::from_message(format!(
                "the state holds {} bytes and got {len}",
                self.byte_size
            )));
        }

        let mut buffer = self.buffer(MemoryType::Cpu)?;
        let data = buffer.as_mut_slice().ok_or_else(|| {
            TritonError::from_message(
                "the state is held in GPU memory, which write_bytes cannot copy into; fill it through State::buffer",
            )
        })?;
        data.copy_from_slice(bytes);
        Ok(())
    }

    /// Where Triton keeps the buffer of the state and how large it is, read
    /// back from the state itself.
    ///
    /// Triton allocates the buffer of a state on demand, so the state has to
    /// have one: an output state the configuration does not share with the
    /// input gets its buffer from [`State::buffer`] or [`State::write_bytes`],
    /// and reading the attributes before either of them fails.
    ///
    /// The `TRITONSERVER_BufferAttributes` belong to the state and are not
    /// freed here: the values are copied out of them.
    ///
    /// Fails when Triton cannot return the attributes, has none for the state
    /// yet, or reports a memory type this version does not know.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut state = State::new(&request, "counter", DataType::Fp32, &[1])?;
    /// state.write_bytes(&0.0_f32.to_ne_bytes())?;
    /// let attributes = state.buffer_attributes()?;
    /// assert_eq!(attributes.byte_size, 4);
    /// ```
    pub fn buffer_attributes(&self) -> Result<StateBufferAttributes, TritonError> {
        let mut raw: *mut triton_ng_sys::TRITONSERVER_BufferAttributes = ptr::null_mut();
        triton_call!(triton_ng_sys::TRITONBACKEND_StateBufferAttributes(
            self.ptr, &mut raw,
        ))?;
        if raw.is_null() {
            return Err(TritonError::from_message(
                "Triton returned no buffer attributes for the state",
            ));
        }

        let mut kind = CPU;
        let mut memory_type_id = 0;
        let mut byte_size = 0_usize;
        triton_call!(triton_ng_sys::TRITONSERVER_BufferAttributesMemoryType(
            raw, &mut kind,
        ))?;
        triton_call!(triton_ng_sys::TRITONSERVER_BufferAttributesMemoryTypeId(
            raw,
            &mut memory_type_id,
        ))?;
        triton_call!(triton_ng_sys::TRITONSERVER_BufferAttributesByteSize(
            raw,
            &mut byte_size,
        ))?;

        let byte_size = u64::try_from(byte_size)
            .map_err(|_| TritonError::from_message("the state buffer is larger than u64 bytes"))?;
        Ok(StateBufferAttributes {
            memory_type: MemoryType::from_sys(kind, memory_type_id)
                .map_err(TritonError::from_message)?,
            byte_size,
        })
    }

    /// Hands the state to the sequence: `TRITONBACKEND_StateUpdate`, after
    /// which the state is no longer valid and the next request of the sequence
    /// reads the data written here as its input state.
    ///
    /// Consumes the state for that reason. Fails when Triton refuses the
    /// update, which leaves the sequence on its previous state.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut state = State::new(&request, "counter", DataType::Fp32, &[1])?;
    /// state.write_bytes(&3.0_f32.to_ne_bytes())?;
    /// state.update()?;
    /// ```
    pub fn update(self) -> Result<(), TritonError> {
        triton_call!(triton_ng_sys::TRITONBACKEND_StateUpdate(self.ptr))
    }
}

/// The memory Triton holds for a [`State`]: where it is, its address and its
/// size.
///
/// Borrowed from the state, which owns the memory: nothing here frees it, and
/// Triton declares it invalid after [`State::update`], so the borrow of the
/// state keeps the buffer from outliving the state it was asked from. Neither
/// `Send` nor `Sync`.
pub struct StateBuffer<'a> {
    ptr: *mut c_void,
    len: usize,
    memory_type: MemoryType,
    /// Ties the buffer to the borrow of the state that returned it.
    _state: PhantomData<&'a mut ()>,
}

impl StateBuffer<'_> {
    /// The number of bytes the state holds.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the state holds no data at all.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The memory Triton keeps the state in: plain or pinned CPU memory, or the
    /// memory of a CUDA device.
    pub fn memory_type(&self) -> MemoryType {
        self.memory_type
    }

    /// The address of the data: a host address for CPU and pinned memory, a
    /// device address for GPU memory, to be filled with the CUDA API of the
    /// backend.
    ///
    /// The address belongs to Triton and is valid until [`State::update`].
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// The data as a slice of the host, `None` when Triton keeps the state in
    /// GPU memory, which a slice of the host cannot stand for.
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        if matches!(self.memory_type, MemoryType::Gpu { .. }) {
            return None;
        }
        // SAFETY: Triton returned `len` bytes of host memory owned by the
        // state, and `&mut self` makes this the only slice into it.
        Some(unsafe { std::slice::from_raw_parts_mut(self.ptr.cast::<u8>(), self.len) })
    }
}

/// Where Triton keeps the buffer of a [`State`], see
/// [`State::buffer_attributes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateBufferAttributes {
    /// The memory the state is kept in.
    pub memory_type: MemoryType,
    /// The number of bytes the buffer holds.
    pub byte_size: u64,
}

/// The number of bytes a state of `datatype` and `shape` holds, without the
/// batch dimension. `Err` says what is wrong with the two.
fn state_byte_size(datatype: DataType, shape: &[i64]) -> Result<u64, String> {
    let element_size = datatype
        .element_size()
        .ok_or_else(|| format!("of {datatype:?} has no size Triton can copy"))?;
    let mut count: u64 = 1;
    for dim in shape {
        let dim = u64::try_from(*dim)
            .map_err(|_| format!("has the negative dimension {dim} in its shape"))?;
        count = count
            .checked_mul(dim)
            .ok_or_else(|| "has a shape larger than u64 elements".to_string())?;
    }
    count
        .checked_mul(element_size as u64)
        .ok_or_else(|| "holds more bytes than u64 takes".to_string())
}

#[cfg(test)]
mod tests {
    use super::state_byte_size;
    use crate::DataType;

    #[test]
    fn a_state_holds_its_datatype_over_its_shape() {
        assert_eq!(state_byte_size(DataType::Fp32, &[1]), Ok(4));
        assert_eq!(state_byte_size(DataType::Fp32, &[3, 64]), Ok(768));
        assert_eq!(state_byte_size(DataType::Int64, &[2, 2]), Ok(32));
        assert_eq!(state_byte_size(DataType::Fp32, &[]), Ok(4));
    }

    #[test]
    fn a_state_of_a_type_without_a_size_is_refused() {
        assert!(state_byte_size(DataType::Bytes, &[1]).is_err());
        assert!(state_byte_size(DataType::Invalid, &[1]).is_err());
    }

    #[test]
    fn a_state_of_a_negative_dimension_is_refused() {
        assert!(state_byte_size(DataType::Fp32, &[1, -1]).is_err());
    }

    #[test]
    fn a_state_too_large_for_u64_is_refused() {
        assert!(state_byte_size(DataType::Int64, &[i64::MAX, 2]).is_err());
        assert!(state_byte_size(DataType::Fp64, &[i64::MAX]).is_err());
    }
}
