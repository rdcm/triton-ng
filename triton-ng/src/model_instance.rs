use crate::error::{Error, TritonError};
use crate::executor::InstanceExecutor;
use crate::model::Model;
use crate::types::InstanceKind;
use crate::utils::cstr_to_string;
use libc::c_char;
use std::any::Any;
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;
use triton_ng_macros::triton_call;

/// A model instance: Triton runs one per `instance_group` entry and device.
///
/// Every instance has its own runtime that runs the batches given to it, see
/// [`crate::Backend::model_instance_execute`].
///
/// Borrows the `TRITONBACKEND_ModelInstance`: Triton owns it and keeps it
/// alive for the hook, so there is nothing to drop. Neither `Send` nor
/// `Sync`: use it only inside the hook; batches get the instance state
/// through [`crate::Context::instance_state`].
///
/// # Examples
///
/// ```ignore
/// fn model_instance_initialize(instance: &ModelInstance) -> Result<(), Error> {
///     log::info(&format!("instance {} on device {}", instance.name()?, instance.device_id()?));
///     Ok(())
/// }
/// ```
pub struct ModelInstance {
    ptr: *mut triton_ng_sys::TRITONBACKEND_ModelInstance,
}

impl ModelInstance {
    pub(crate) fn from_ptr(ptr: *mut triton_ng_sys::TRITONBACKEND_ModelInstance) -> Self {
        Self { ptr }
    }

    pub fn name(&self) -> Result<String, TritonError> {
        let mut name: *const c_char = std::ptr::null();
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceName(
            self.ptr, &mut name
        ))?;
        Ok(unsafe { cstr_to_string(name) })
    }

    pub fn device_id(&self) -> Result<i32, TritonError> {
        let mut device_id: i32 = 0;
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceDeviceId(
            self.ptr,
            &mut device_id
        ))?;
        Ok(device_id)
    }

    pub fn kind(&self) -> Result<InstanceKind, TritonError> {
        let mut kind: triton_ng_sys::TRITONSERVER_InstanceGroupKind = 0;
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceKind(
            self.ptr, &mut kind
        ))?;
        Ok(InstanceKind::from_sys(kind))
    }

    /// The model this instance belongs to.
    pub fn model(&self) -> Result<Model, TritonError> {
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceModel(
            self.ptr,
            &mut _
        ))?;
        Ok(Model::from_ptr(ptr))
    }

    /// Stores backend-defined state for this instance.
    ///
    /// Call it once, from `Backend::model_instance_initialize`; batches of the
    /// instance get it through [`crate::Context::instance_state`]. Use it for
    /// what must not be shared between instances, like caches of GPU buffers
    /// of the instance's device. The state is dropped after
    /// `Backend::model_instance_finalize`, once every batch of the instance
    /// has finished. A second call fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn model_instance_initialize(instance: &ModelInstance) -> Result<(), Error> {
    ///     instance.set_state(Mutex::new(GpuCache::new(64 << 20)))?;
    ///     Ok(())
    /// }
    /// ```
    pub fn set_state<T: Any + Send + Sync>(&self, state: T) -> Result<(), Error> {
        self.executor()?.set_state(Arc::new(state))
    }

    /// Stores the executor of this instance. Fails if one is already stored:
    /// replacing it would free it under the references returned by
    /// [`ModelInstance::executor`].
    pub(crate) fn set_executor(&self, executor: InstanceExecutor) -> Result<(), TritonError> {
        if !self.raw_state()?.is_null() {
            return Err(TritonError::from_message(
                "model instance executor is already set",
            ));
        }
        let state_ptr = Box::into_raw(Box::new(executor)) as *mut c_void;
        if let Err(e) = triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceSetState(
            self.ptr, state_ptr
        )) {
            unsafe { drop(Box::from_raw(state_ptr as *mut InstanceExecutor)) };
            return Err(e);
        }
        Ok(())
    }

    pub(crate) fn executor(&self) -> Result<&InstanceExecutor, TritonError> {
        let state_ptr = self.raw_state()?;
        if state_ptr.is_null() {
            return Err(TritonError::from_message(
                "model instance executor is not initialized",
            ));
        }
        // SAFETY: the pointer was produced by `set_executor` from a
        // `Box<InstanceExecutor>` and stays valid until `take_executor`, which
        // only runs on instance finalize.
        Ok(unsafe { &*(state_ptr as *const InstanceExecutor) })
    }

    /// Removes the executor of this instance, if any, and hands it over.
    pub(crate) fn take_executor(&self) -> Result<Option<InstanceExecutor>, TritonError> {
        let state_ptr = self.raw_state()?;
        if state_ptr.is_null() {
            return Ok(None);
        }
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceSetState(
            self.ptr,
            ptr::null_mut()
        ))?;
        Ok(Some(*unsafe {
            Box::from_raw(state_ptr as *mut InstanceExecutor)
        }))
    }

    fn raw_state(&self) -> Result<*mut c_void, TritonError> {
        let mut state_ptr: *mut c_void = ptr::null_mut();
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceState(
            self.ptr,
            &mut state_ptr
        ))?;
        Ok(state_ptr)
    }
}
