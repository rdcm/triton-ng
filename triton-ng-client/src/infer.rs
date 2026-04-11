use std::time::Duration;

use crate::generated::inference::ModelInferResponse;

use crate::datatype::Datatype;
use crate::error::{Error, Result};

// ── Input ─────────────────────────────────────────────────────────────────────

/// Typed tensor data for an inference input.
#[derive(Debug, Clone, PartialEq)]
pub enum InputData {
    Bool(Vec<bool>),
    Uint8(Vec<u8>),
    Uint16(Vec<u16>),
    Uint32(Vec<u32>),
    Uint64(Vec<u64>),
    Int8(Vec<i8>),
    Int16(Vec<i16>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    /// FP16 elements as raw little-endian bytes (2 bytes per element).
    Fp16(Vec<u8>),
    Fp32(Vec<f32>),
    Fp64(Vec<f64>),
    /// BF16 elements as raw little-endian bytes (2 bytes per element).
    Bf16(Vec<u8>),
    /// Variable-length byte strings (each element is an independent byte sequence).
    Bytes(Vec<Vec<u8>>),
}

impl InputData {
    /// Returns the [`Datatype`] corresponding to this variant.
    pub fn datatype(&self) -> Datatype {
        match self {
            Self::Bool(_) => Datatype::Bool,
            Self::Uint8(_) => Datatype::Uint8,
            Self::Uint16(_) => Datatype::Uint16,
            Self::Uint32(_) => Datatype::Uint32,
            Self::Uint64(_) => Datatype::Uint64,
            Self::Int8(_) => Datatype::Int8,
            Self::Int16(_) => Datatype::Int16,
            Self::Int32(_) => Datatype::Int32,
            Self::Int64(_) => Datatype::Int64,
            Self::Fp16(_) => Datatype::Fp16,
            Self::Fp32(_) => Datatype::Fp32,
            Self::Fp64(_) => Datatype::Fp64,
            Self::Bf16(_) => Datatype::Bf16,
            Self::Bytes(_) => Datatype::Bytes,
        }
    }

    pub(crate) fn to_raw_bytes(&self) -> Vec<u8> {
        match self {
            Self::Bool(v) => v.iter().map(|&b| b as u8).collect(),
            Self::Uint8(v) => v.clone(),
            Self::Uint16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::Uint32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::Uint64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::Int8(v) => v.iter().map(|&x| x as u8).collect(),
            Self::Int16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::Int32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::Int64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::Fp16(v) | Self::Bf16(v) => v.clone(),
            Self::Fp32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::Fp64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::Bytes(v) => {
                let mut out = Vec::new();
                for elem in v {
                    out.extend_from_slice(&(elem.len() as u32).to_le_bytes());
                    out.extend_from_slice(elem);
                }
                out
            }
        }
    }
}

/// An input tensor for an inference request.
#[derive(Debug, Clone)]
pub struct InferInput {
    /// Tensor name as declared in the model configuration.
    pub name: String,
    /// Actual tensor shape. All dimensions must be positive — replace any `-1` values
    /// from [`crate::TensorMetadata::shape`] with the real sizes before creating this.
    pub shape: Vec<i64>,
    pub data: InputData,
}

impl InferInput {
    pub fn bool(name: impl Into<String>, shape: Vec<i64>, data: Vec<bool>) -> Self {
        Self {
            name: name.into(),
            shape,
            data: InputData::Bool(data),
        }
    }

    pub fn uint8(name: impl Into<String>, shape: Vec<i64>, data: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            shape,
            data: InputData::Uint8(data),
        }
    }

    pub fn uint16(name: impl Into<String>, shape: Vec<i64>, data: Vec<u16>) -> Self {
        Self {
            name: name.into(),
            shape,
            data: InputData::Uint16(data),
        }
    }

    pub fn uint32(name: impl Into<String>, shape: Vec<i64>, data: Vec<u32>) -> Self {
        Self {
            name: name.into(),
            shape,
            data: InputData::Uint32(data),
        }
    }

    pub fn uint64(name: impl Into<String>, shape: Vec<i64>, data: Vec<u64>) -> Self {
        Self {
            name: name.into(),
            shape,
            data: InputData::Uint64(data),
        }
    }

    pub fn int8(name: impl Into<String>, shape: Vec<i64>, data: Vec<i8>) -> Self {
        Self {
            name: name.into(),
            shape,
            data: InputData::Int8(data),
        }
    }

    pub fn int16(name: impl Into<String>, shape: Vec<i64>, data: Vec<i16>) -> Self {
        Self {
            name: name.into(),
            shape,
            data: InputData::Int16(data),
        }
    }

    pub fn int32(name: impl Into<String>, shape: Vec<i64>, data: Vec<i32>) -> Self {
        Self {
            name: name.into(),
            shape,
            data: InputData::Int32(data),
        }
    }

    pub fn int64(name: impl Into<String>, shape: Vec<i64>, data: Vec<i64>) -> Self {
        Self {
            name: name.into(),
            shape,
            data: InputData::Int64(data),
        }
    }

    /// FP16 input as raw little-endian bytes (2 bytes per element).
    pub fn fp16(name: impl Into<String>, shape: Vec<i64>, data: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            shape,
            data: InputData::Fp16(data),
        }
    }

    pub fn fp32(name: impl Into<String>, shape: Vec<i64>, data: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            shape,
            data: InputData::Fp32(data),
        }
    }

    pub fn fp64(name: impl Into<String>, shape: Vec<i64>, data: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            shape,
            data: InputData::Fp64(data),
        }
    }

    /// BF16 input as raw little-endian bytes (2 bytes per element).
    pub fn bf16(name: impl Into<String>, shape: Vec<i64>, data: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            shape,
            data: InputData::Bf16(data),
        }
    }

    pub fn bytes(name: impl Into<String>, shape: Vec<i64>, data: Vec<Vec<u8>>) -> Self {
        Self {
            name: name.into(),
            shape,
            data: InputData::Bytes(data),
        }
    }
}

// ── Output ────────────────────────────────────────────────────────────────────

/// Decoded tensor data from an inference response.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputData {
    Bool(Vec<bool>),
    Uint8(Vec<u8>),
    Uint16(Vec<u16>),
    Uint32(Vec<u32>),
    Uint64(Vec<u64>),
    Int8(Vec<i8>),
    Int16(Vec<i16>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    /// FP16 elements as raw little-endian bytes (2 bytes per element).
    Fp16(Vec<u8>),
    Fp32(Vec<f32>),
    Fp64(Vec<f64>),
    /// BF16 elements as raw little-endian bytes (2 bytes per element).
    Bf16(Vec<u8>),
    /// Variable-length byte strings.
    Bytes(Vec<Vec<u8>>),
}

/// A decoded output tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct InferOutput {
    pub name: String,
    pub datatype: Datatype,
    /// Shape of the tensor as reported by Triton.
    pub shape: Vec<i64>,
    pub data: OutputData,
}

/// Response from an inference request.
#[derive(Debug, Clone, PartialEq)]
pub struct InferResponse {
    /// Request ID echoed back from Triton. Empty if no ID was set on the request.
    pub id: String,
    pub outputs: Vec<InferOutput>,
}

impl InferResponse {
    pub(crate) fn from_proto(response: ModelInferResponse) -> Result<Self> {
        let outputs = response
            .outputs
            .iter()
            .enumerate()
            .map(|(i, meta)| {
                let raw = response
                    .raw_output_contents
                    .get(i)
                    .ok_or_else(|| Error::Decode(format!("missing raw data for output {i}")))?;

                let datatype = Datatype::try_from(meta.datatype.as_str())?;

                let data = match datatype {
                    Datatype::Bool => OutputData::Bool(decode_bool(raw)),
                    Datatype::Uint8 => OutputData::Uint8(raw.to_vec()),
                    Datatype::Int8 => OutputData::Int8(decode_int8(raw)),
                    Datatype::Fp16 => OutputData::Fp16(raw.to_vec()),
                    Datatype::Bf16 => OutputData::Bf16(raw.to_vec()),
                    Datatype::Uint16 => OutputData::Uint16(decode_u16(raw)?),
                    Datatype::Uint32 => OutputData::Uint32(decode_u32(raw)?),
                    Datatype::Uint64 => OutputData::Uint64(decode_u64(raw)?),
                    Datatype::Int16 => OutputData::Int16(decode_i16(raw)?),
                    Datatype::Int32 => OutputData::Int32(decode_i32(raw)?),
                    Datatype::Int64 => OutputData::Int64(decode_i64(raw)?),
                    Datatype::Fp32 => OutputData::Fp32(decode_f32(raw)?),
                    Datatype::Fp64 => OutputData::Fp64(decode_f64(raw)?),
                    Datatype::Bytes => OutputData::Bytes(decode_bytes(raw)?),
                };

                Ok(InferOutput {
                    name: meta.name.clone(),
                    datatype,
                    shape: meta.shape.clone(),
                    data,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(InferResponse {
            id: response.id,
            outputs,
        })
    }
}

// ── Options ───────────────────────────────────────────────────────────────────

/// Per-request options for [`crate::TritonClient::infer`].
#[derive(Debug, Default)]
pub struct InferOptions {
    /// Optional request identifier echoed back in [`InferResponse::id`].
    pub id: Option<String>,
    /// Request timeout. `None` means no deadline.
    pub timeout: Option<Duration>,
}

// ── Decode helpers ────────────────────────────────────────────────────────────

fn check_elem_size(data: &[u8], elem_size: usize, typename: &str) -> Result<()> {
    if !data.len().is_multiple_of(elem_size) {
        Err(Error::Decode(format!(
            "{typename} raw data length {} is not a multiple of {elem_size}",
            data.len()
        )))
    } else {
        Ok(())
    }
}

fn decode_bool(data: &[u8]) -> Vec<bool> {
    data.iter().map(|&b| b != 0).collect()
}

fn decode_int8(data: &[u8]) -> Vec<i8> {
    data.iter().map(|&b| b as i8).collect()
}

fn decode_u16(data: &[u8]) -> Result<Vec<u16>> {
    check_elem_size(data, 2, "UINT16")?;
    Ok(data
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn decode_u32(data: &[u8]) -> Result<Vec<u32>> {
    check_elem_size(data, 4, "UINT32")?;
    Ok(data
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn decode_u64(data: &[u8]) -> Result<Vec<u64>> {
    check_elem_size(data, 8, "UINT64")?;
    Ok(data
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

fn decode_i16(data: &[u8]) -> Result<Vec<i16>> {
    check_elem_size(data, 2, "INT16")?;
    Ok(data
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn decode_i32(data: &[u8]) -> Result<Vec<i32>> {
    check_elem_size(data, 4, "INT32")?;
    Ok(data
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn decode_i64(data: &[u8]) -> Result<Vec<i64>> {
    check_elem_size(data, 8, "INT64")?;
    Ok(data
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

fn decode_f32(data: &[u8]) -> Result<Vec<f32>> {
    check_elem_size(data, 4, "FP32")?;
    Ok(data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn decode_f64(data: &[u8]) -> Result<Vec<f64>> {
    check_elem_size(data, 8, "FP64")?;
    Ok(data
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

fn decode_bytes(data: &[u8]) -> Result<Vec<Vec<u8>>> {
    let mut cur = data;
    let mut out = Vec::new();
    while !cur.is_empty() {
        let (prefix, rest) = cur
            .split_at_checked(4)
            .ok_or_else(|| Error::Decode("BYTES: truncated length prefix".into()))?;
        let len = u32::from_le_bytes([prefix[0], prefix[1], prefix[2], prefix[3]]) as usize;
        let (elem, rest) = rest.split_at_checked(len).ok_or_else(|| {
            Error::Decode(format!(
                "BYTES: element claims {len} bytes but only {} remain",
                rest.len()
            ))
        })?;
        out.push(elem.to_vec());
        cur = rest;
    }
    Ok(out)
}
