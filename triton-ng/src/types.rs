/// Tensor data type — mirrors `TRITONSERVER_DataType` without exposing the C enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Invalid,
    Bool,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Int8,
    Int16,
    Int32,
    Int64,
    Fp16,
    Fp32,
    Fp64,
    Bytes,
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

/// Model instance group kind — mirrors `TRITONSERVER_InstanceGroupKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceKind {
    Auto,
    Cpu,
    Gpu,
    Model,
}

impl InstanceKind {
    #[allow(non_upper_case_globals)]
    pub(crate) fn from_sys(v: triton_ng_sys::TRITONSERVER_InstanceGroupKind) -> Self {
        use triton_ng_sys::*;
        match v {
            TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_CPU => Self::Cpu,
            TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_GPU => Self::Gpu,
            TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_MODEL => Self::Model,
            _ => Self::Auto,
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
    Bool(bool),
    Int(i64),
    Double(f64),
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

#[cfg(test)]
mod tests {
    use super::Parameter;
    use std::ffi::{CString, c_void};
    use triton_ng_sys::*;

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
