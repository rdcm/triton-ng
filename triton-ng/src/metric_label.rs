use crate::TritonError;
use crate::utils::cstring_from_str;

/// A label of a metric, a string `TRITONSERVER_Parameter` passed to
/// `TRITONSERVER_MetricNew`, which copies the name and value into the metric
/// (core `r25.09`, `Metric::Metric`).
///
/// Owns the parameter and deletes it on drop. Neither `Send` nor `Sync`: it
/// lives only while a metric is made.
pub(crate) struct MetricLabel {
    ptr: *mut triton_ng_sys::TRITONSERVER_Parameter,
}

impl MetricLabel {
    pub(crate) fn new(name: &str, value: &str) -> Result<Self, TritonError> {
        let name = cstring_from_str(name)?;
        let value = cstring_from_str(value)?;
        // SAFETY: both strings are NUL-terminated and copied by Triton.
        let ptr = unsafe {
            triton_ng_sys::TRITONSERVER_ParameterNew(
                name.as_ptr(),
                triton_ng_sys::TRITONSERVER_parametertype_enum_TRITONSERVER_PARAMETER_STRING,
                value.as_ptr().cast(),
            )
        };
        if ptr.is_null() {
            return Err(TritonError::from_message(
                "Triton failed to create a metric label",
            ));
        }
        Ok(Self { ptr })
    }

    pub(crate) fn as_ptr(&self) -> *const triton_ng_sys::TRITONSERVER_Parameter {
        self.ptr
    }
}

impl Drop for MetricLabel {
    fn drop(&mut self) {
        unsafe { triton_ng_sys::TRITONSERVER_ParameterDelete(self.ptr) };
    }
}
