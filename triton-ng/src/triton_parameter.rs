use crate::TritonError;
use crate::utils::cstring_from_str;
use std::marker::PhantomData;

/// A `TRITONSERVER_Parameter`: a label of a metric, passed to
/// `TRITONSERVER_MetricNew`, or a parameter of
/// `TRITONSERVER_ServerLoadModelWithParameters`.
///
/// Owns the parameter and deletes it on drop. Triton copies the name and a
/// string value into it (core `r25.09`, `InferenceParameter`), but only
/// points at the data of a bytes value: `'a` keeps that data alive as long
/// as the parameter. Neither `Send` nor `Sync`: it lives only for one call.
pub(crate) struct TritonParameter<'a> {
    ptr: *mut triton_ng_sys::TRITONSERVER_Parameter,
    _data: PhantomData<&'a [u8]>,
}

impl TritonParameter<'static> {
    pub(crate) fn string(name: &str, value: &str) -> Result<Self, TritonError> {
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
        Self::from_ptr(ptr)
    }
}

impl<'a> TritonParameter<'a> {
    pub(crate) fn bytes(name: &str, value: &'a [u8]) -> Result<Self, TritonError> {
        let name = cstring_from_str(name)?;
        // SAFETY: the name is NUL-terminated and copied by Triton; the bytes
        // are not copied and outlive the parameter through `'a`.
        let ptr = unsafe {
            triton_ng_sys::TRITONSERVER_ParameterBytesNew(
                name.as_ptr(),
                value.as_ptr().cast(),
                value.len() as u64,
            )
        };
        Self::from_ptr(ptr)
    }

    fn from_ptr(ptr: *mut triton_ng_sys::TRITONSERVER_Parameter) -> Result<Self, TritonError> {
        if ptr.is_null() {
            return Err(TritonError::from_message(
                "Triton failed to create a parameter",
            ));
        }
        Ok(Self {
            ptr,
            _data: PhantomData,
        })
    }

    pub(crate) fn as_ptr(&self) -> *const triton_ng_sys::TRITONSERVER_Parameter {
        self.ptr
    }
}

impl Drop for TritonParameter<'_> {
    fn drop(&mut self) {
        unsafe { triton_ng_sys::TRITONSERVER_ParameterDelete(self.ptr) };
    }
}
