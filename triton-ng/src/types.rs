use crate::memory_manager::SysMemoryType;

/// Tensor data type — mirrors `TRITONSERVER_DataType` without exposing the C enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// Not a datatype; Triton uses it for an unset type.
    Invalid,
    /// Booleans.
    Bool,
    /// 8-bit unsigned integers.
    Uint8,
    /// 16-bit unsigned integers.
    Uint16,
    /// 32-bit unsigned integers.
    Uint32,
    /// 64-bit unsigned integers.
    Uint64,
    /// 8-bit signed integers.
    Int8,
    /// 16-bit signed integers.
    Int16,
    /// 32-bit signed integers.
    Int32,
    /// 64-bit signed integers.
    Int64,
    /// 16-bit floats, as raw bytes.
    Fp16,
    /// 32-bit floats.
    Fp32,
    /// 64-bit floats.
    Fp64,
    /// Byte strings of variable length, each with a 4-byte length prefix.
    Bytes,
    /// 16-bit brain floats, as raw bytes.
    Bf16,
}

impl DataType {
    /// Size of one element in bytes, `None` for BYTES and INVALID.
    pub fn element_size(self) -> Option<usize> {
        match self {
            Self::Bool | Self::Uint8 | Self::Int8 => Some(1),
            Self::Uint16 | Self::Int16 | Self::Fp16 | Self::Bf16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Fp32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Fp64 => Some(8),
            Self::Bytes | Self::Invalid => None,
        }
    }

    /// Name of the type in a model config (`TYPE_FP32`), `None` for INVALID.
    pub(crate) fn config_name(self) -> Option<&'static str> {
        Some(match self {
            Self::Invalid => return None,
            Self::Bool => "TYPE_BOOL",
            Self::Uint8 => "TYPE_UINT8",
            Self::Uint16 => "TYPE_UINT16",
            Self::Uint32 => "TYPE_UINT32",
            Self::Uint64 => "TYPE_UINT64",
            Self::Int8 => "TYPE_INT8",
            Self::Int16 => "TYPE_INT16",
            Self::Int32 => "TYPE_INT32",
            Self::Int64 => "TYPE_INT64",
            Self::Fp16 => "TYPE_FP16",
            Self::Fp32 => "TYPE_FP32",
            Self::Fp64 => "TYPE_FP64",
            Self::Bytes => "TYPE_STRING",
            Self::Bf16 => "TYPE_BF16",
        })
    }

    pub(crate) fn to_sys(self) -> triton_ng_sys::TRITONSERVER_DataType {
        use triton_ng_sys::*;
        match self {
            Self::Invalid => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INVALID,
            Self::Bool => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BOOL,
            Self::Uint8 => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT8,
            Self::Uint16 => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT16,
            Self::Uint32 => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT32,
            Self::Uint64 => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT64,
            Self::Int8 => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT8,
            Self::Int16 => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT16,
            Self::Int32 => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT32,
            Self::Int64 => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT64,
            Self::Fp16 => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP16,
            Self::Fp32 => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32,
            Self::Fp64 => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP64,
            Self::Bytes => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BYTES,
            Self::Bf16 => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BF16,
        }
    }

    #[allow(non_upper_case_globals)]
    pub(crate) fn from_sys(v: triton_ng_sys::TRITONSERVER_DataType) -> Self {
        use triton_ng_sys::*;
        match v {
            TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BOOL => Self::Bool,
            TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT8 => Self::Uint8,
            TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT16 => Self::Uint16,
            TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT32 => Self::Uint32,
            TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT64 => Self::Uint64,
            TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT8 => Self::Int8,
            TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT16 => Self::Int16,
            TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT32 => Self::Int32,
            TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT64 => Self::Int64,
            TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP16 => Self::Fp16,
            TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32 => Self::Fp32,
            TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP64 => Self::Fp64,
            TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BYTES => Self::Bytes,
            TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BF16 => Self::Bf16,
            _ => Self::Invalid,
        }
    }
}

/// Kind of a model instance group, mirrors `TRITONSERVER_InstanceGroupKind`:
/// the kind of an instance ([`crate::ModelInstance::kind`]) or of a group the
/// backend prefers ([`crate::BackendAttributes::preferred_instance_group`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceGroupKind {
    /// Triton picks GPU when there is one, CPU otherwise.
    Auto,
    /// An instance on the CPU.
    Cpu,
    /// An instance on a GPU.
    Gpu,
    /// The model decides where it runs.
    Model,
}

impl InstanceGroupKind {
    pub(crate) fn to_sys(self) -> triton_ng_sys::TRITONSERVER_InstanceGroupKind {
        use triton_ng_sys::*;
        match self {
            Self::Auto => TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_AUTO,
            Self::Cpu => TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_CPU,
            Self::Gpu => TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_GPU,
            Self::Model => TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_MODEL,
        }
    }

    /// Fails on a kind this version of triton-ng does not know.
    #[allow(non_upper_case_globals)]
    pub(crate) fn from_sys(
        v: triton_ng_sys::TRITONSERVER_InstanceGroupKind,
    ) -> Result<Self, String> {
        use triton_ng_sys::*;
        match v {
            TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_AUTO => {
                Ok(Self::Auto)
            }
            TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_CPU => Ok(Self::Cpu),
            TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_GPU => Ok(Self::Gpu),
            TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_MODEL => {
                Ok(Self::Model)
            }
            other => Err(format!("unknown instance group kind {other}")),
        }
    }
}

/// Value of a request or response parameter, see [`crate::Request::parameter`]
/// and [`crate::Response::set_parameter`].
///
/// # Examples
///
/// ```ignore
/// if request.parameter("verbose")? == Some(Parameter::Bool(true)) {
///     // ...
/// }
/// response.set_parameter("tokens", 42)?;
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Parameter {
    /// A boolean.
    Bool(bool),
    /// An integer.
    Int(i64),
    /// A double-precision float.
    Double(f64),
    /// A string.
    String(String),
}

impl Parameter {
    /// Reads a parameter value returned by Triton.
    ///
    /// # Safety
    ///
    /// `value` must point to the value of a parameter of type `kind`: a
    /// NUL-terminated string, an `i64`, a `bool` or an `f64`.
    #[allow(non_upper_case_globals)]
    pub(crate) unsafe fn from_sys(
        kind: triton_ng_sys::TRITONSERVER_ParameterType,
        value: *const std::ffi::c_void,
    ) -> Result<Self, String> {
        use triton_ng_sys::*;
        if value.is_null() {
            return Err("parameter value is null".to_string());
        }
        unsafe {
            match kind {
                TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_BOOL => {
                    Ok(Self::Bool(*(value as *const bool)))
                }
                TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_INT => {
                    Ok(Self::Int(*(value as *const i64)))
                }
                TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_DOUBLE => {
                    Ok(Self::Double(*(value as *const f64)))
                }
                TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_STRING => Ok(Self::String(
                    crate::utils::cstr_to_string(value as *const libc::c_char),
                )),
                other => Err(format!("unsupported parameter type {other}")),
            }
        }
    }
}

impl From<bool> for Parameter {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<i64> for Parameter {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<f64> for Parameter {
    fn from(value: f64) -> Self {
        Self::Double(value)
    }
}

impl From<&str> for Parameter {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<String> for Parameter {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

/// The correlation ID of a request: the sequence it belongs to for stateful
/// models. Clients set it as `sequence_id` of the gRPC `ModelInferRequest`,
/// a number or a string. A request without one has none: Triton has no
/// zero number or empty string ID.
///
/// # Examples
///
/// ```ignore
/// match request.correlation_id()? {
///     Some(CorrelationId::Int(id)) => sessions.get(&id),
///     Some(CorrelationId::String(id)) => named_sessions.get(&id),
///     None => None,
/// };
/// inference.set_correlation_id(CorrelationId::String("call-42".to_string()))?;
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CorrelationId {
    /// A numeric ID, never 0.
    Int(u64),
    /// A string ID, never empty and at most 128 bytes.
    String(String),
}

impl From<u64> for CorrelationId {
    fn from(id: u64) -> Self {
        Self::Int(id)
    }
}

impl From<&str> for CorrelationId {
    fn from(id: &str) -> Self {
        Self::String(id.to_string())
    }
}

impl From<String> for CorrelationId {
    fn from(id: String) -> Self {
        Self::String(id)
    }
}

/// Flags of a request in a sequence: whether it starts or ends it.
/// Clients set them as `sequence_start` and `sequence_end` of the gRPC
/// `ModelInferRequest`.
///
/// # Examples
///
/// ```ignore
/// let flags = request.flags()?;
/// if flags.sequence_start {
///     sessions.open(id);
/// }
/// inference.set_flags(RequestFlags { sequence_start: true, sequence_end: false })?;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RequestFlags {
    /// The request is the first of its sequence.
    pub sequence_start: bool,
    /// The request is the last of its sequence.
    pub sequence_end: bool,
}

const SEQUENCE_START: u32 =
    triton_ng_sys::tritonserver_requestflag_enum_TRITONSERVER_REQUEST_FLAG_SEQUENCE_START;
const SEQUENCE_END: u32 =
    triton_ng_sys::tritonserver_requestflag_enum_TRITONSERVER_REQUEST_FLAG_SEQUENCE_END;

impl RequestFlags {
    /// Reads the flags Triton returns. Fails on bits unknown to this
    /// version, which could change what the request means.
    pub(crate) fn from_sys(flags: u32) -> Result<Self, String> {
        let unknown = flags & !(SEQUENCE_START | SEQUENCE_END);
        if unknown != 0 {
            return Err(format!("request has unknown flags {unknown:#x}"));
        }
        Ok(Self {
            sequence_start: flags & SEQUENCE_START != 0,
            sequence_end: flags & SEQUENCE_END != 0,
        })
    }

    pub(crate) fn to_sys(self) -> u32 {
        let start = if self.sequence_start {
            SEQUENCE_START
        } else {
            0
        };
        let end = if self.sequence_end { SEQUENCE_END } else { 0 };
        start | end
    }
}

/// The kind of memory a buffer or a memory usage report is about
/// (`TRITONSERVER_MemoryType` without the C enum).
///
/// # Examples
///
/// ```ignore
/// let staging = ctx.memory().pinned(1 << 20)?;
/// instance.report_memory_usage(&[
///     BufferAttributes::new(MemoryType::CpuPinned, staging.len() as u64)?,
///     BufferAttributes::new(MemoryType::Gpu { device_id: 0 }, 8 << 20)?,
/// ])?;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryType {
    /// Ordinary host memory.
    Cpu,
    /// Page-locked host memory, the kind [`crate::PinnedRamBuffer`] holds.
    CpuPinned,
    /// Memory of the CUDA device `device_id`, counting from zero.
    Gpu {
        /// The CUDA device, as [`crate::ModelInstance::device_id`] returns it.
        device_id: i32,
    },
}

impl MemoryType {
    /// The C memory type and the memory type id Triton takes for it.
    pub(crate) fn to_sys(self) -> (SysMemoryType, i64) {
        use crate::memory_manager::{CPU, CPU_PINNED, GPU};
        match self {
            Self::Cpu => (CPU, 0),
            Self::CpuPinned => (CPU_PINNED, 0),
            Self::Gpu { device_id } => (GPU, i64::from(device_id)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{InstanceGroupKind, Parameter, RequestFlags};
    use std::ffi::{CString, c_void};
    use triton_ng_sys::*;

    #[test]
    fn instance_group_kinds_round_trip_and_unknown_ones_fail() {
        for kind in [
            InstanceGroupKind::Auto,
            InstanceGroupKind::Cpu,
            InstanceGroupKind::Gpu,
            InstanceGroupKind::Model,
        ] {
            assert_eq!(InstanceGroupKind::from_sys(kind.to_sys()), Ok(kind));
        }
        assert!(InstanceGroupKind::from_sys(42).is_err());
    }

    #[test]
    fn request_flags_round_trip() {
        for sequence_start in [false, true] {
            for sequence_end in [false, true] {
                let flags = RequestFlags {
                    sequence_start,
                    sequence_end,
                };
                assert_eq!(RequestFlags::from_sys(flags.to_sys()), Ok(flags));
            }
        }
        assert_eq!(RequestFlags::default().to_sys(), 0);
    }

    #[test]
    fn unknown_request_flags_are_refused() {
        assert!(RequestFlags::from_sys(4).is_err());
        assert!(RequestFlags::from_sys(u32::MAX).is_err());
    }

    fn read<T>(kind: TRITONSERVER_ParameterType, value: &T) -> Result<Parameter, String> {
        unsafe { Parameter::from_sys(kind, value as *const T as *const c_void) }
    }

    #[test]
    fn reads_every_supported_type() {
        assert_eq!(
            read(
                TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_BOOL,
                &true
            ),
            Ok(Parameter::Bool(true))
        );
        assert_eq!(
            read(
                TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_INT,
                &-7i64
            ),
            Ok(Parameter::Int(-7))
        );
        assert_eq!(
            read(
                TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_DOUBLE,
                &0.5f64
            ),
            Ok(Parameter::Double(0.5))
        );

        let text = CString::new("hello").unwrap();
        let string = unsafe {
            Parameter::from_sys(
                TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_STRING,
                text.as_ptr() as *const c_void,
            )
        };
        assert_eq!(string, Ok(Parameter::String("hello".to_string())));
    }

    #[test]
    fn rejects_bytes_and_null() {
        assert!(
            read(
                TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_BYTES,
                &0u8
            )
            .is_err()
        );
        let null = unsafe {
            Parameter::from_sys(
                TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_BOOL,
                std::ptr::null(),
            )
        };
        assert!(null.is_err());
    }
}
