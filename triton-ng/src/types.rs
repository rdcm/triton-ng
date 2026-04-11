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
    pub(crate) fn to_sys(self) -> triton_sys::TRITONSERVER_DataType {
        use triton_sys::*;
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
    pub(crate) fn from_sys(v: triton_sys::TRITONSERVER_DataType) -> Self {
        use triton_sys::*;
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
    pub(crate) fn from_sys(v: triton_sys::TRITONSERVER_InstanceGroupKind) -> Self {
        use triton_sys::*;
        match v {
            TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_CPU => Self::Cpu,
            TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_GPU => Self::Gpu,
            TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_MODEL => Self::Model,
            _ => Self::Auto,
        }
    }
}
