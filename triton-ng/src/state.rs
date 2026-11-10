//! State a backend keeps between the requests of a sequence.

use crate::memory_manager::CPU;
use crate::types::MemoryType;
use crate::utils::cstring_from_str;
use crate::{DataType, Request, TritonError};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;
use triton_ng_macros::triton_call;

/// A state the model configuration declares, from
/// [`crate::Model::sequence_states`].
///
/// Holds the `state` entry of the `sequence_batching` configuration: the name a
/// [`State`] is created under, its datatype, and its dimensions without the
/// batch dimension, -1 for a dimension the configuration leaves dynamic.
///
/// A state is only created from one of these, which is what makes the datatype
/// and the fixed dimensions of the configured state the ones Triton keeps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateConfig {
    name: String,
    input_name: String,
    datatype: DataType,
    dims: Vec<i64>,
}

impl StateConfig {
    /// Reads one entry of `sequence_batching.state`.
    pub(crate) fn from_json(entry: &serde_json::Value) -> Result<Self, TritonError> {
        let string = |key: &str| {
            entry[key]
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| TritonError::from_message(format!("state entry has no '{key}'")))
        };
        let name = string("output_name")?;
        let input_name = string("input_name")?;
        let declared = string("data_type")?;
        let datatype = DataType::from_config_name(&declared).ok_or_else(|| {
            TritonError::from_message(format!(
                "state '{name}' has the datatype {declared}, which this version does not know"
            ))
        })?;
        let dims = entry["dims"]
            .as_array()
            .map(|dims| dims.iter().filter_map(serde_json::Value::as_i64).collect())
            .unwrap_or_default();
        Ok(Self {
            name,
            input_name,
            datatype,
            dims,
        })
    }

    /// The `output_name` of the entry: the name the state is created under, and
    /// the data the next request of the sequence reads as its input state.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The `input_name` of the entry, under which the input state of the
    /// sequence reaches the backend as an input of the request.
    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    /// The datatype of the state.
    pub fn datatype(&self) -> DataType {
        self.datatype
    }

    /// The dimensions of the state without the batch dimension, -1 for a
    /// dimension the configuration leaves dynamic.
    pub fn dims(&self) -> &[i64] {
        &self.dims
    }

    /// The number of bytes a state of `shape` holds, a message saying why
    /// `shape` does not fit the configuration.
    fn check_shape(&self, shape: &[i64]) -> Result<u64, String> {
        if shape.len() != self.dims.len() {
            return Err(format!(
                "has {} dimensions where the configuration declares {}",
                shape.len(),
                self.dims.len()
            ));
        }
        for (dim, declared) in shape.iter().zip(&self.dims) {
            if *dim <= 0 {
                return Err(format!("has the dimension {dim}, which is not positive"));
            }
            if *declared >= 0 && *declared != *dim {
                return Err(format!(
                    "has the dimension {dim} where the configuration declares {declared}"
                ));
            }
        }
        state_byte_size(self.datatype, shape)
    }
}

/// The state `config` of a request of a model with `sequence_batching`, filled
/// with the shape the request needs.
///
/// Borrows the request mutably for as long as the state is alive: Triton keeps
/// one state per name for the whole sequence, so a second state of the same
/// configuration would be the same memory, and two slices of it would be
/// aliasing pointers. The mutable borrow is what makes that a compile error,
/// and it also keeps the inputs of the request, shared or not, out of reach
/// while the state is written.
///
/// [`State::update`] consumes the state, because Triton documents the state as
/// valid only until the update: the data written into it belongs to the
/// sequence afterwards, and the next request of the same sequence reads it as
/// its input state, as the input its configuration pairs with it.
///
/// Neither `Send` nor `Sync`, like the borrow of the request it holds.
///
/// # Examples
///
/// ```ignore
/// let mut request = request;
/// let previous = request.get_input("counter")?.as_fp32_vec()?;
/// let mut state = State::new(&mut request, &config, &[1])?;
/// state.write_bytes(&(previous[0] + 1.0).to_ne_bytes())?;
/// state.update()?;
/// ```
pub struct State<'a> {
    ptr: *mut triton_ng_sys::TRITONBACKEND_State,
    /// The size Triton resizes the state buffer to, from the datatype of the
    /// configuration over `shape`.
    byte_size: u64,
    /// Whether the buffer was written, see [`State::update`].
    written: bool,
    /// Holds the request the state lives in, for as long as the state is used.
    _request: PhantomData<&'a mut Request>,
}

impl<'a> State<'a> {
    /// The state `config` of `request`, whose data has `shape` without the
    /// batch dimension.
    ///
    /// `shape` has to fit the configuration: the same number of dimensions, the
    /// dimensions it declares fixed, and a positive value for every dimension
    /// it declares dynamic as -1. Triton changes the metadata of the state to
    /// what it is given and does not compare it with the configuration, so a
    /// shape that does not fit would hand the wrong state to the next request
    /// of the sequence; it is refused here, before Triton touches the state.
    ///
    /// Fails when the model has no sequence batching or no state of that name,
    /// Triton has nothing to create the state on, or `shape` does not fit the
    /// configuration. The configuration also has to declare an `initial_state`:
    /// without one Triton allocates the memory of the state and leaves it
    /// uninitialized.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut state = State::new(&mut request, &counter, &[1])?;
    /// state.write_bytes(&0.0_f32.to_ne_bytes())?;
    /// state.update()?;
    /// ```
    pub fn new(
        request: &'a mut Request,
        config: &StateConfig,
        shape: &[i64],
    ) -> Result<Self, TritonError> {
        let byte_size = config
            .check_shape(shape)
            .map_err(|e| TritonError::from_message(format!("state '{}' {e}", config.name)))?;
        let name = cstring_from_str(&config.name)?;
        let dims_count = u32::try_from(shape.len()).map_err(|_| {
            TritonError::from_message(format!(
                "state '{}' has more dimensions than Triton takes",
                config.name
            ))
        })?;

        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_StateNew(
            &mut _,
            request.as_ptr(),
            name.as_ptr(),
            config.datatype.to_sys(),
            shape.as_ptr(),
            dims_count,
        ))?;

        Ok(Self {
            ptr,
            byte_size,
            written: false,
            _request: PhantomData,
        })
    }

    /// The number of bytes the state holds, the size of its datatype over
    /// `shape`.
    pub fn byte_size(&self) -> u64 {
        self.byte_size
    }

    /// Copies `bytes` into the state buffer: exactly as many bytes as the state
    /// holds.
    ///
    /// The buffer is asked for in CPU memory. Triton allocates host memory
    /// through its pinned pool, so the buffer can come back pinned, which is
    /// host memory as well and is copied into the same way; a state Triton
    /// keeps in GPU memory is refused, see [`State::device_buffer`].
    ///
    /// Fails when `bytes` is not the size of the state, when Triton cannot
    /// resize the buffer, or when the buffer Triton returns is not host memory.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut state = State::new(&mut request, &counter, &[1])?;
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

        let (buffer, memory_type) = self.raw_buffer(MemoryType::Cpu)?;
        if matches!(memory_type, MemoryType::Gpu { .. }) {
            return Err(TritonError::from_message(
                "the state is held in GPU memory, which write_bytes cannot copy into; fill it through State::device_buffer",
            ));
        }
        // SAFETY: Triton returned `byte_size` bytes of host memory owned by the
        // state, `bytes` holds exactly that many, and `&mut self` makes this the
        // only write into the buffer.
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast::<u8>(), bytes.len()) };
        self.written = true;
        Ok(())
    }

    /// The address Triton keeps a state in GPU memory at, with its device and
    /// its size, for the CUDA code of the backend to fill.
    ///
    /// The state counts as written once this returns, and the backend has to
    /// finish the writes on the device, for instance by synchronizing its
    /// stream, before it calls [`State::update`]: Triton swaps the buffer of the
    /// state then, and reads the device memory afterwards.
    ///
    /// Fails when Triton cannot allocate the buffer or hands it out in host
    /// memory, which [`State::write_bytes`] fills.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut state = State::new(&mut request, &counter, &[1])?;
    /// let buffer = state.device_buffer(MemoryType::Gpu { id: 0 })?;
    /// copy_to_device(buffer.ptr, &next, stream)?;
    /// stream.synchronize()?;
    /// state.update()?;
    /// ```
    pub fn device_buffer(&mut self, memory_type: MemoryType) -> Result<DeviceBuffer, TritonError> {
        if !matches!(memory_type, MemoryType::Gpu { .. }) {
            return Err(TritonError::from_message(
                "a device buffer is asked for in GPU memory",
            ));
        }
        let (buffer, got) = self.raw_buffer(memory_type)?;
        let MemoryType::Gpu { device_id } = got else {
            return Err(TritonError::from_message(
                "Triton handed the state out in host memory, which State::write_bytes fills",
            ));
        };
        self.written = true;
        Ok(DeviceBuffer {
            ptr: buffer,
            byte_size: self.byte_size,
            device_id: i64::from(device_id),
        })
    }

    /// Where Triton keeps the buffer of the state and how large it is, read
    /// back from the state itself.
    ///
    /// Triton allocates the buffer of a state on demand, so the state has to
    /// have one: an output state the configuration does not share with the
    /// input gets its buffer from [`State::write_bytes`] or
    /// [`State::device_buffer`], and reading the attributes before either of
    /// them fails.
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
    /// let mut state = State::new(&mut request, &counter, &[1])?;
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
    /// Consumes the state for that reason. Fails when the state was never
    /// written, since Triton would hand the buffer Triton allocated, which holds
    /// whatever the allocator left in it, to the next request of the sequence;
    /// and when Triton refuses the update, which leaves the sequence on its
    /// previous state.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut state = State::new(&mut request, &counter, &[1])?;
    /// state.write_bytes(&3.0_f32.to_ne_bytes())?;
    /// state.update()?;
    /// ```
    pub fn update(self) -> Result<(), TritonError> {
        if !self.written {
            return Err(TritonError::from_message(
                "the state was never written, so Triton would hand its uninitialized buffer to the next request",
            ));
        }
        triton_call!(triton_ng_sys::TRITONBACKEND_StateUpdate(self.ptr))
    }

    /// The buffer Triton keeps for the state, asked for in `memory_type`, with
    /// the memory type it really is in.
    fn raw_buffer(
        &mut self,
        memory_type: MemoryType,
    ) -> Result<(*mut c_void, MemoryType), TritonError> {
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
        Ok((buffer, memory_type))
    }
}

/// The device memory Triton keeps a [`State`] in, see
/// [`State::device_buffer`].
///
/// The address belongs to Triton and is valid until [`State::update`].
#[derive(Debug)]
pub struct DeviceBuffer {
    /// The device address of the state data.
    pub ptr: *mut c_void,
    /// The number of bytes Triton allocated for the state.
    pub byte_size: u64,
    /// The id of the device the address belongs to.
    pub device_id: i64,
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
        .ok_or_else(|| format!("is of {datatype:?}, which has no size Triton can copy"))?;
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
    use super::{StateConfig, state_byte_size};
    use crate::DataType;

    fn counter() -> StateConfig {
        StateConfig {
            name: "counter_out".to_string(),
            input_name: "counter_in".to_string(),
            datatype: DataType::Fp32,
            dims: vec![1],
        }
    }

    #[test]
    fn a_shape_that_fits_the_configuration_holds_the_data() {
        assert_eq!(counter().check_shape(&[1]), Ok(4));

        let dynamic = StateConfig {
            dims: vec![-1, 3],
            ..counter()
        };
        assert_eq!(dynamic.check_shape(&[7, 3]), Ok(84));
    }

    #[test]
    fn a_shape_the_configuration_does_not_declare_is_refused() {
        assert!(counter().check_shape(&[2]).is_err());
        assert!(counter().check_shape(&[1, 1]).is_err());
        assert!(counter().check_shape(&[0]).is_err());
        assert!(counter().check_shape(&[-1]).is_err());
    }

    #[test]
    fn a_state_of_a_type_without_a_size_is_refused() {
        let bytes = StateConfig {
            datatype: DataType::Bytes,
            ..counter()
        };
        assert!(bytes.check_shape(&[1]).is_err());
        assert!(state_byte_size(DataType::Invalid, &[1]).is_err());
    }

    #[test]
    fn a_state_too_large_for_u64_is_refused() {
        assert!(state_byte_size(DataType::Int64, &[i64::MAX, 2]).is_err());
        assert!(state_byte_size(DataType::Fp64, &[i64::MAX]).is_err());
    }
}
