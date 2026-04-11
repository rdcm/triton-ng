use crate::error::TritonError;
use crate::types::InstanceKind;
use crate::utils::cstr_to_string;
use libc::c_char;
use triton_ng_macros::triton_call;

pub struct ModelInstance {
    ptr: *mut triton_sys::TRITONBACKEND_ModelInstance,
}

impl ModelInstance {
    pub(crate) fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_ModelInstance) -> Self {
        Self { ptr }
    }

    pub fn name(&self) -> Result<String, TritonError> {
        let mut name: *const c_char = std::ptr::null();
        triton_call!(triton_sys::TRITONBACKEND_ModelInstanceName(
            self.ptr, &mut name
        ))?;
        Ok(unsafe { cstr_to_string(name) })
    }

    pub fn device_id(&self) -> Result<i32, TritonError> {
        let mut device_id: i32 = 0;
        triton_call!(triton_sys::TRITONBACKEND_ModelInstanceDeviceId(
            self.ptr,
            &mut device_id
        ))?;
        Ok(device_id)
    }

    pub fn kind(&self) -> Result<InstanceKind, TritonError> {
        let mut kind: triton_sys::TRITONSERVER_InstanceGroupKind = 0;
        triton_call!(triton_sys::TRITONBACKEND_ModelInstanceKind(
            self.ptr, &mut kind
        ))?;
        Ok(InstanceKind::from_sys(kind))
    }
}
