use std::fmt;

use crate::error::{Error, Result};

/// Tensor element type as defined by the Triton inference protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Datatype {
    Bool,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Int8,
    Int16,
    Int32,
    Int64,
    /// Half-precision float (16-bit), transmitted as raw little-endian bytes.
    Fp16,
    Fp32,
    Fp64,
    /// Brain float (16-bit), transmitted as raw little-endian bytes.
    Bf16,
    /// Variable-length byte string.
    Bytes,
}

impl Datatype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::Uint8 => "UINT8",
            Self::Uint16 => "UINT16",
            Self::Uint32 => "UINT32",
            Self::Uint64 => "UINT64",
            Self::Int8 => "INT8",
            Self::Int16 => "INT16",
            Self::Int32 => "INT32",
            Self::Int64 => "INT64",
            Self::Fp16 => "FP16",
            Self::Fp32 => "FP32",
            Self::Fp64 => "FP64",
            Self::Bf16 => "BF16",
            Self::Bytes => "BYTES",
        }
    }
}

impl TryFrom<String> for Datatype {
    type Error = Error;

    fn try_from(s: String) -> Result<Self> {
        Self::try_from(s.as_str())
    }
}

impl TryFrom<&str> for Datatype {
    type Error = Error;

    fn try_from(s: &str) -> Result<Self> {
        match s {
            "BOOL" => Ok(Self::Bool),
            "UINT8" => Ok(Self::Uint8),
            "UINT16" => Ok(Self::Uint16),
            "UINT32" => Ok(Self::Uint32),
            "UINT64" => Ok(Self::Uint64),
            "INT8" => Ok(Self::Int8),
            "INT16" => Ok(Self::Int16),
            "INT32" => Ok(Self::Int32),
            "INT64" => Ok(Self::Int64),
            "FP16" => Ok(Self::Fp16),
            "FP32" => Ok(Self::Fp32),
            "FP64" => Ok(Self::Fp64),
            "BF16" => Ok(Self::Bf16),
            "BYTES" => Ok(Self::Bytes),
            other => Err(Error::Decode(format!("unknown datatype: {other}"))),
        }
    }
}

impl fmt::Display for Datatype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
