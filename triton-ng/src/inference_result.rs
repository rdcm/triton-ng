use crate::utils::decode_le;
use crate::{DataType, Error, GpuBuffer, PinnedRamBuffer, RamBuffer, WrittenBuffer};

/// Where the data of an output is.
pub(crate) enum OutputData {
    /// Copied to CPU memory owned by the result.
    Host(Vec<u8>),
    /// In the buffer given with `add_requested_output_into`.
    Buffer(WrittenBuffer),
    /// The buffer was handed over with one of the `take_*` methods.
    Taken,
}

pub struct OutputTensor {
    pub name: String,
    pub shape: Vec<i64>,
    pub datatype: crate::types::DataType,
    pub(crate) data: OutputData,
}

impl OutputTensor {
    /// The bytes of the output.
    ///
    /// Fails when the output is in a GPU buffer, which the CPU cannot read,
    /// or when its buffer was taken.
    pub fn bytes(&self) -> Result<&[u8], Error> {
        match &self.data {
            OutputData::Host(data) => Ok(data),
            OutputData::Buffer(WrittenBuffer::Ram(buffer)) => Ok(buffer),
            OutputData::Buffer(WrittenBuffer::Pinned(buffer)) => Ok(buffer),
            OutputData::Buffer(WrittenBuffer::Gpu(_)) => {
                Err(format!("output '{}' is in GPU memory", self.name).into())
            }
            OutputData::Taken => Err(format!("buffer of output '{}' is taken", self.name).into()),
        }
    }

    pub fn as_fp32_vec(&self) -> Result<Vec<f32>, Error> {
        self.expect_datatype(DataType::Fp32)?;
        decode_le(self.bytes()?, f32::from_le_bytes)
    }

    pub fn as_i32_vec(&self) -> Result<Vec<i32>, Error> {
        self.expect_datatype(DataType::Int32)?;
        decode_le(self.bytes()?, i32::from_le_bytes)
    }

    pub fn as_i64_vec(&self) -> Result<Vec<i64>, Error> {
        self.expect_datatype(DataType::Int64)?;
        decode_le(self.bytes()?, i64::from_le_bytes)
    }

    /// Integer tensor widened to `i64`, accepts both INT32 and INT64.
    pub fn as_index_vec(&self) -> Result<Vec<i64>, Error> {
        match self.datatype {
            DataType::Int32 => Ok(decode_le(self.bytes()?, i32::from_le_bytes)?
                .into_iter()
                .map(i64::from)
                .collect()),
            _ => self.as_i64_vec(),
        }
    }

    fn expect_datatype(&self, expected: DataType) -> Result<(), Error> {
        if self.datatype != expected {
            return Err(format!(
                "output '{}' has datatype {:?}, expected {:?}",
                self.name, self.datatype, expected
            )
            .into());
        }
        Ok(())
    }
}

pub struct InferenceResult {
    pub outputs: Vec<OutputTensor>,
    pub error: Option<String>,
}

impl InferenceResult {
    /// Returns the output tensor named `name`.
    pub fn output(&self, name: &str) -> Result<&OutputTensor, Error> {
        self.outputs
            .iter()
            .find(|o| o.name == name)
            .ok_or_else(|| format!("missing output '{name}'").into())
    }

    /// Hands over the GPU buffer the output `name` was written into, see
    /// [`InferenceRequest::add_requested_output_into`].
    ///
    /// Fails when the output is missing, is not in a GPU buffer, or was taken.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut result = ctx.server().infer_async(inference).await?;
    /// let state: GpuBuffer = result.take_gpu("ho")?;
    /// ```
    pub fn take_gpu(&mut self, name: &str) -> Result<GpuBuffer, Error> {
        match self.take(name)? {
            WrittenBuffer::Gpu(buffer) => Ok(buffer),
            other => self.put_back(name, other, "GPU"),
        }
    }

    /// Hands over the pinned buffer the output `name` was written into, see
    /// [`InferenceResult::take_gpu`].
    pub fn take_pinned(&mut self, name: &str) -> Result<PinnedRamBuffer, Error> {
        match self.take(name)? {
            WrittenBuffer::Pinned(buffer) => Ok(buffer),
            other => self.put_back(name, other, "pinned"),
        }
    }

    /// Hands over the CPU buffer the output `name` was written into, see
    /// [`InferenceResult::take_gpu`].
    pub fn take_ram(&mut self, name: &str) -> Result<RamBuffer, Error> {
        match self.take(name)? {
            WrittenBuffer::Ram(buffer) => Ok(buffer),
            other => self.put_back(name, other, "CPU"),
        }
    }

    fn output_mut(&mut self, name: &str) -> Result<&mut OutputTensor, Error> {
        self.outputs
            .iter_mut()
            .find(|o| o.name == name)
            .ok_or_else(|| format!("missing output '{name}'").into())
    }

    fn take(&mut self, name: &str) -> Result<WrittenBuffer, Error> {
        let output = self.output_mut(name)?;
        match std::mem::replace(&mut output.data, OutputData::Taken) {
            OutputData::Buffer(buffer) => Ok(buffer),
            OutputData::Host(data) => {
                output.data = OutputData::Host(data);
                Err(format!("output '{name}' has no buffer").into())
            }
            OutputData::Taken => Err(format!("buffer of output '{name}' is taken").into()),
        }
    }

    fn put_back<T>(&mut self, name: &str, buffer: WrittenBuffer, kind: &str) -> Result<T, Error> {
        self.output_mut(name)?.data = OutputData::Buffer(buffer);
        Err(format!("output '{name}' is not in a {kind} buffer").into())
    }
}

impl InferenceResult {
    pub(crate) fn failed(message: impl Into<String>) -> Self {
        Self {
            outputs: vec![],
            error: Some(message.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{InferenceResult, OutputData, OutputTensor};
    use crate::{DataType, MemoryManager, OutputBuffer};

    fn result_with_buffers() -> InferenceResult {
        let memory = MemoryManager::heap();
        let output = |name: &str, data| OutputTensor {
            name: name.to_string(),
            shape: vec![2],
            datatype: DataType::Fp32,
            data,
        };
        let written = |buffer: OutputBuffer| OutputData::Buffer(buffer.into_written());
        InferenceResult {
            outputs: vec![
                output("gpu", written(memory.gpu(0, 8).unwrap().into())),
                output("pinned", written(memory.pinned(8).unwrap().into())),
                output("host", OutputData::Host(vec![0; 8])),
            ],
            error: None,
        }
    }

    #[test]
    fn output_buffer_is_taken_once() {
        let mut result = result_with_buffers();

        assert_eq!(result.take_gpu("gpu").unwrap().len(), 8);
        assert!(result.take_gpu("gpu").is_err());
        assert!(result.output("gpu").unwrap().bytes().is_err());
    }

    #[test]
    fn buffer_of_other_memory_stays_in_the_result() {
        let mut result = result_with_buffers();

        assert!(result.take_gpu("pinned").is_err());
        assert_eq!(
            result.output("pinned").unwrap().as_fp32_vec().unwrap(),
            [0.0, 0.0]
        );
        assert_eq!(result.take_pinned("pinned").unwrap().len(), 8);
    }

    #[test]
    fn host_outputs_have_no_buffer_to_take() {
        let mut result = result_with_buffers();

        assert!(result.take_ram("host").is_err());
        assert_eq!(result.output("host").unwrap().bytes().unwrap().len(), 8);
        assert!(result.take_ram("missing").is_err());
    }

    #[test]
    fn gpu_outputs_cannot_be_read_on_the_cpu() {
        let result = result_with_buffers();

        assert!(result.output("gpu").unwrap().as_fp32_vec().is_err());
    }
}
