pub struct ServerInfo {
    pub name: String,
    pub version: String,
    pub extensions: Vec<String>,
}

pub struct ModelInfo {
    pub name: String,
    pub versions: Vec<String>,
    pub platform: String,
    pub inputs: Vec<TensorMeta>,
    pub outputs: Vec<TensorMeta>,
}

pub struct TensorMeta {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
}

pub struct ModelConfig {
    pub name: String,
    pub platform: String,
}

pub struct ModelStats {
    pub name: String,
    pub version: String,
    pub inference_count: u64,
}

pub struct ModelEntry {
    pub name: String,
    pub state: String,
}

pub struct InferInput {
    pub name: String,
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
}

pub struct InferResult {
    pub id: String,
    pub outputs: Vec<OutputTensor>,
}

pub struct OutputTensor {
    pub name: String,
    pub shape: Vec<i64>,
    pub data_f32: Vec<f32>,
}

pub struct ShmRegion {
    pub key: String,
    pub offset: u64,
    pub byte_size: u64,
}

#[allow(dead_code)]
pub struct CudaRegion {
    pub device_id: u64,
    pub byte_size: u64,
}

#[derive(Debug)]
pub enum LogValue {
    Bool(bool),
    Uint32(u32),
    Text(String),
}
