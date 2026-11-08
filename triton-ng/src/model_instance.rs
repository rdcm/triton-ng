use crate::buffer_attributes::BufferAttributes;
use crate::error::{Error, TritonError};
use crate::executor::InstanceExecutor;
use crate::host_policy::HostPolicy;
use crate::message::serialize_to_json;
use crate::model::Model;
use crate::types::InstanceGroupKind;
use crate::utils::cstr_to_string;
use libc::c_char;
use std::any::Any;
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;
use triton_ng_macros::triton_call;

/// A secondary device configured for a model instance, returned by
/// [`ModelInstance::secondary_devices`].
///
/// Owns its values and holds nothing of Triton.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecondaryDevice {
    /// The device kind, as named by Triton's model config, such as `KIND_NVDLA`.
    pub kind: String,
    /// The identifier of the device within its kind.
    pub id: i64,
}

/// A model instance: Triton runs one per `instance_group` entry and device.
///
/// By default every instance has its own Triton thread that runs the batches
/// given to it one after another, see [`crate::CustomBackend::model_instance_execute`].
/// [`crate::ExecutionPolicy::DeviceBlocking`] makes the non-passive GPU
/// instances of one model and device share a thread.
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

    /// The name of the instance, `<model>_<index>` by default.
    pub fn name(&self) -> Result<String, TritonError> {
        let mut name: *const c_char = std::ptr::null();
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceName(
            self.ptr, &mut name
        ))?;
        Ok(unsafe { cstr_to_string(name) })
    }

    /// The device of the instance: the GPU id for `KIND_GPU`, 0 otherwise.
    pub fn device_id(&self) -> Result<i32, TritonError> {
        let mut device_id: i32 = 0;
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceDeviceId(
            self.ptr,
            &mut device_id
        ))?;
        Ok(device_id)
    }

    /// Kind of the instance group of this instance. Fails on a kind this
    /// version of triton-ng does not know.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if instance.kind()? == InstanceGroupKind::Gpu {
    ///     log::info(&format!("instance on GPU {}", instance.device_id()?));
    /// }
    /// ```
    pub fn kind(&self) -> Result<InstanceGroupKind, TritonError> {
        let mut kind: triton_ng_sys::TRITONSERVER_InstanceGroupKind = 0;
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceKind(
            self.ptr, &mut kind
        ))?;
        InstanceGroupKind::from_sys(kind).map_err(TritonError::from_message)
    }

    /// The host policy Triton selected for this instance.
    ///
    /// The name is the `host_policy` field of the instance's `instance_group`,
    /// or the default Triton picks for the kind of the group when the config
    /// leaves it out: `cpu`, `gpu_<device-id>` or `model`. The settings come
    /// from the `--host-policy` options Triton was started with, so a policy
    /// Triton does not know gives an empty map.
    ///
    /// The returned [`HostPolicy`] owns its strings. Triton retains the
    /// underlying message, which stays alive with the instance.
    ///
    /// # Errors
    ///
    /// Fails when Triton cannot return or serialize the host policy, when
    /// Triton returns a null message, or when the message is not the documented
    /// `{"<policy-name>": {"<setting>": "<value>"}}` JSON.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let policy = instance.host_policy()?;
    /// log::info(&format!("host policy: {}", policy.name()));
    /// if let Some(node) = policy.setting("numa-node") {
    ///     log::info(&format!("NUMA node: {node}"));
    /// }
    /// ```
    pub fn host_policy(&self) -> Result<HostPolicy, TritonError> {
        let message = triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceHostPolicy(
            self.ptr,
            &mut _
        ))?;
        if message.is_null() {
            return Err(TritonError::from_message(
                "TRITONBACKEND_ModelInstanceHostPolicy returned null",
            ));
        }

        // SAFETY: the message is a member of the TritonModelInstance (core
        // backend_model_instance.cc, `&ti->HostPolicyMessage()`), which stays
        // alive for this borrowed `ModelInstance`. Triton keeps owning it:
        // serialization only reads it and never deletes it.
        let json = unsafe { serialize_to_json(message) }?;
        HostPolicy::from_json(&json).map_err(TritonError::from_message)
    }

    /// The model this instance belongs to.
    pub fn model(&self) -> Result<Model, TritonError> {
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceModel(
            self.ptr,
            &mut _
        ))?;
        Ok(Model::from_ptr(ptr))
    }

    /// Names of the optimization profiles configured for this instance, in the
    /// order of `instance_group.profile` in the model config.
    ///
    /// The returned strings are copies; Triton keeps the originals for as long
    /// as the instance lives.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn model_instance_initialize(instance: &ModelInstance) -> Result<(), Error> {
    ///     let profiles = instance.profile_names()?;
    ///     instance.set_state(ProfileCache::new(&profiles))?;
    ///     Ok(())
    /// }
    /// ```
    pub fn profile_names(&self) -> Result<Vec<String>, TritonError> {
        collect_entries(&Profiles { instance: self }).map_err(list_error)
    }

    /// Whether this instance is passive: Triton does not schedule inference
    /// requests on it (`instance_group.passive` in the model config).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if !instance.is_passive()? {
    ///     log::info("this instance accepts scheduled requests");
    /// }
    /// ```
    pub fn is_passive(&self) -> Result<bool, TritonError> {
        let mut is_passive = false;
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceIsPassive(
            self.ptr,
            &mut is_passive
        ))?;
        Ok(is_passive)
    }

    /// Secondary devices configured for this instance through
    /// `instance_group.secondary_devices` in the model config.
    ///
    /// The returned devices own their strings; Triton keeps the originals for
    /// as long as the instance lives.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn model_instance_initialize(instance: &ModelInstance) -> Result<(), Error> {
    ///     for device in instance.secondary_devices()? {
    ///         log::info(&format!("using {} {}", device.kind, device.id));
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn secondary_devices(&self) -> Result<Vec<SecondaryDevice>, TritonError> {
        collect_entries(&SecondaryDevices { instance: self }).map_err(list_error)
    }

    /// Reports the memory the instance itself holds, for the `memory_usage` of
    /// `/v2/models/<model>/stats`.
    ///
    /// Pass one [`BufferAttributes`] per memory type and id, each holding the
    /// total of that kind; entries of the same memory type and id are summed
    /// before they are reported, since Triton keeps one entry per memory type
    /// and id. Call it from `CustomBackend::model_instance_initialize` or any
    /// later hook of the instance: each call replaces what the previous one
    /// reported and an empty slice clears it. Triton copies the entries while
    /// the call runs, so the slice can be dropped afterwards. The memory the
    /// model itself holds is reported through
    /// [`crate::Model::report_memory_usage`].
    ///
    /// # Errors
    ///
    /// Fails when the entries of one memory type and id add up to more than a
    /// `u64`, when `usage` holds more entries than Triton can count, or when
    /// Triton rejects the report.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn model_instance_initialize(instance: &ModelInstance) -> Result<(), Error> {
    ///     let device = MemoryType::Gpu { device_id: instance.device_id()? };
    ///     instance.report_memory_usage(&[BufferAttributes::new(device, 8 << 20)?])?;
    ///     Ok(())
    /// }
    /// ```
    pub fn report_memory_usage(&self, usage: &[BufferAttributes]) -> Result<(), TritonError> {
        let (entries, count) = BufferAttributes::report(usage)?;
        let mut pointers: Vec<_> = entries.iter().map(BufferAttributes::as_ptr).collect();
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceReportMemoryUsage(
            self.ptr,
            pointers.as_mut_ptr(),
            count
        ))
    }

    /// Stores backend-defined state for this instance.
    ///
    /// Call it once, from `CustomBackend::model_instance_initialize`; batches of the
    /// instance get it through [`crate::Context::instance_state`]. Use it for
    /// what must not be shared between instances, like caches of GPU buffers
    /// of the instance's device. The state is dropped after
    /// `CustomBackend::model_instance_finalize`, once every batch of the instance
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

/// Why one of the lists Triton keeps for an instance could not be read.
///
/// Carries text rather than a [`TritonError`], so the unit tests exercise the
/// handling of a failing entry without a Triton server.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ListError {
    /// The Triton call that reads the list failed.
    Triton(String),
    /// Triton returned no value where its documentation promises one.
    Missing(String),
}

impl std::fmt::Display for ListError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Triton(message) | Self::Missing(message) => f.write_str(message),
        }
    }
}

/// A list Triton keeps for an instance, read entry by entry.
pub(crate) trait InstanceList {
    /// One entry of the list.
    type Item;

    /// How many entries Triton reports.
    fn count(&self) -> Result<u32, ListError>;

    /// The entry at `index`, which is lower than [`InstanceList::count`].
    fn entry(&self, index: u32) -> Result<Self::Item, ListError>;
}

/// Every entry of `list`, stopping at the first one that fails.
///
/// The entries read before the failing one are dropped, so a caller never sees
/// a partial list and nothing is left behind.
pub(crate) fn collect_entries<L: InstanceList>(list: &L) -> Result<Vec<L::Item>, ListError> {
    let count = list.count()?;
    let mut entries = Vec::with_capacity(count as usize);
    for index in 0..count {
        entries.push(list.entry(index)?);
    }
    Ok(entries)
}

/// The list error as the Triton error the public methods return.
fn list_error(e: ListError) -> TritonError {
    TritonError::from_message(e.to_string())
}

/// The optimization profiles of an instance, from the Triton calls of
/// [`ModelInstance::profile_names`].
struct Profiles<'a> {
    instance: &'a ModelInstance,
}

impl InstanceList for Profiles<'_> {
    type Item = String;

    fn count(&self) -> Result<u32, ListError> {
        let mut count = 0;
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceProfileCount(
            self.instance.ptr,
            &mut count
        ))
        .map_err(|e| ListError::Triton(e.to_string()))?;
        Ok(count)
    }

    fn entry(&self, index: u32) -> Result<String, ListError> {
        let mut name = ptr::null();
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceProfileName(
            self.instance.ptr,
            index,
            &mut name
        ))
        .map_err(|e| ListError::Triton(e.to_string()))?;
        if name.is_null() {
            return Err(ListError::Missing(format!(
                "TRITONBACKEND_ModelInstanceProfileName returned null for index {index}"
            )));
        }
        // SAFETY: the core returns `Profiles()[index].c_str()`, owned by the
        // instance. It is copied here and never modified or freed.
        Ok(unsafe { cstr_to_string(name) })
    }
}

/// The secondary devices of an instance, from the Triton calls of
/// [`ModelInstance::secondary_devices`].
struct SecondaryDevices<'a> {
    instance: &'a ModelInstance,
}

impl InstanceList for SecondaryDevices<'_> {
    type Item = SecondaryDevice;

    fn count(&self) -> Result<u32, ListError> {
        let mut count = 0;
        triton_call!(
            triton_ng_sys::TRITONBACKEND_ModelInstanceSecondaryDeviceCount(
                self.instance.ptr,
                &mut count
            )
        )
        .map_err(|e| ListError::Triton(e.to_string()))?;
        Ok(count)
    }

    fn entry(&self, index: u32) -> Result<SecondaryDevice, ListError> {
        let mut kind = ptr::null();
        let mut id = 0;
        triton_call!(
            triton_ng_sys::TRITONBACKEND_ModelInstanceSecondaryDeviceProperties(
                self.instance.ptr,
                index,
                &mut kind,
                &mut id
            )
        )
        .map_err(|e| ListError::Triton(e.to_string()))?;
        if kind.is_null() {
            return Err(ListError::Missing(format!(
                "TRITONBACKEND_ModelInstanceSecondaryDeviceProperties returned a null kind for index {index}"
            )));
        }
        // SAFETY: the core returns `SecondaryDevices()[index].kind_.c_str()`,
        // owned by the instance. It is copied here and never modified or freed.
        let kind = unsafe { cstr_to_string(kind) };
        Ok(SecondaryDevice { kind, id })
    }
}

#[cfg(test)]
mod tests {
    use super::{InstanceList, ListError, SecondaryDevice, collect_entries};

    /// A list that answers like the C API of an instance whose profile names
    /// are readable, except for the entries a test marks as unusable.
    struct FakeProfiles {
        names: Vec<Option<&'static str>>,
        failing_index: Option<u32>,
    }

    impl InstanceList for FakeProfiles {
        type Item = String;

        fn count(&self) -> Result<u32, ListError> {
            Ok(self.names.len() as u32)
        }

        fn entry(&self, index: u32) -> Result<String, ListError> {
            if self.failing_index == Some(index) {
                return Err(ListError::Triton(format!(
                    "the call failed for index {index}"
                )));
            }
            match self.names.get(index as usize) {
                Some(Some(name)) => Ok((*name).to_string()),
                Some(None) => Err(ListError::Missing(format!(
                    "the name is null for index {index}"
                ))),
                None => Err(ListError::Missing(format!("no entry at index {index}"))),
            }
        }
    }

    /// A list of secondary devices, to check that a device keeps its kind and
    /// its id.
    struct FakeDevices {
        devices: Vec<SecondaryDevice>,
    }

    impl InstanceList for FakeDevices {
        type Item = SecondaryDevice;

        fn count(&self) -> Result<u32, ListError> {
            Ok(self.devices.len() as u32)
        }

        fn entry(&self, index: u32) -> Result<SecondaryDevice, ListError> {
            match self.devices.get(index as usize) {
                Some(device) => Ok(device.clone()),
                None => Err(ListError::Missing(format!("no device at index {index}"))),
            }
        }
    }

    #[test]
    fn every_entry_is_collected_in_the_order_of_the_list() -> Result<(), String> {
        let profiles = FakeProfiles {
            names: vec![Some("profile_0"), Some("profile_1"), Some("profile_2")],
            failing_index: None,
        };
        assert_eq!(
            collect_entries(&profiles).map_err(|e| e.to_string())?,
            vec!["profile_0", "profile_1", "profile_2"]
        );

        let empty = FakeProfiles {
            names: Vec::new(),
            failing_index: None,
        };
        assert_eq!(
            collect_entries(&empty).map_err(|e| e.to_string())?,
            Vec::<String>::new()
        );
        Ok(())
    }

    #[test]
    fn a_null_entry_fails_instead_of_becoming_an_empty_name() {
        let profiles = FakeProfiles {
            names: vec![Some("profile_0"), None],
            failing_index: None,
        };
        let collected = collect_entries(&profiles);
        assert!(
            matches!(collected, Err(ListError::Missing(message)) if message.contains("index 1")),
            "a null name is reported with its index"
        );
    }

    #[test]
    fn a_failing_call_stops_the_enumeration() {
        let profiles = FakeProfiles {
            names: vec![Some("profile_0"), Some("profile_1"), Some("profile_2")],
            failing_index: Some(1),
        };
        let collected = collect_entries(&profiles);
        assert!(
            matches!(collected, Err(ListError::Triton(message)) if message.contains("index 1")),
            "the error of the failing entry is the one that comes back"
        );
    }

    #[test]
    fn devices_keep_their_kind_and_id() -> Result<(), String> {
        let devices = FakeDevices {
            devices: vec![
                SecondaryDevice {
                    kind: "KIND_NVDLA".to_string(),
                    id: 3,
                },
                SecondaryDevice {
                    kind: "KIND_NVDLA".to_string(),
                    id: 0,
                },
            ],
        };
        let collected = collect_entries(&devices).map_err(|e| e.to_string())?;
        assert_eq!(
            collected,
            vec![
                SecondaryDevice {
                    kind: "KIND_NVDLA".to_string(),
                    id: 3
                },
                SecondaryDevice {
                    kind: "KIND_NVDLA".to_string(),
                    id: 0
                },
            ]
        );
        Ok(())
    }
}
