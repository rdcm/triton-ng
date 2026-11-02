use crate::backend_attributes::BackendAttributes;
use crate::error::TritonError;
use triton_ng_macros::triton_call;

/// The attributes object Triton gives `TRITONBACKEND_GetBackendAttribute` to
/// fill.
///
/// Borrows the `TRITONBACKEND_BackendAttribute`: it is a local of the core
/// (`TritonBackend::UpdateAttributes`, backend_manager.cc) that lives for the
/// call, so there is nothing to drop. Neither `Send` nor `Sync`.
pub(crate) struct BackendAttribute {
    ptr: *mut triton_ng_sys::TRITONBACKEND_BackendAttribute,
}

impl BackendAttribute {
    pub(crate) fn from_ptr(
        ptr: *mut triton_ng_sys::TRITONBACKEND_BackendAttribute,
    ) -> Result<Self, TritonError> {
        if ptr.is_null() {
            return Err(TritonError::from_message("backend attributes are null"));
        }
        Ok(Self { ptr })
    }

    /// Writes `attributes` into the object. The core applies them only when
    /// the entry point returns success, so a failure here leaves Triton with
    /// the attributes it had.
    pub(crate) fn set(&self, attributes: &BackendAttributes) -> Result<(), TritonError> {
        for group in &attributes.preferred_instance_groups {
            let devices = if group.devices.is_empty() {
                std::ptr::null()
            } else {
                group.devices.as_ptr()
            };
            triton_call!(
                triton_ng_sys::TRITONBACKEND_BackendAttributeAddPreferredInstanceGroup(
                    self.ptr,
                    group.kind.to_sys(),
                    group.count,
                    devices,
                    group.devices.len() as u64,
                )
            )?;
        }
        triton_call!(
            triton_ng_sys::TRITONBACKEND_BackendAttributeSetParallelModelInstanceLoading(
                self.ptr,
                attributes.parallel_model_instance_loading,
            )
        )
    }
}
