use crate::error::Error;

/// Kind of a model instance group, mirrors `TRITONSERVER_InstanceGroupKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceGroupKind {
    /// Triton picks GPU when there is one, CPU otherwise.
    Auto,
    Cpu,
    Gpu,
    /// The model decides where it runs.
    Model,
}

impl InstanceGroupKind {
    pub(crate) fn to_sys(self) -> triton_ng_sys::TRITONSERVER_InstanceGroupKind {
        use triton_ng_sys::*;
        match self {
            Self::Auto => TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_AUTO,
            Self::Cpu => TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_CPU,
            Self::Gpu => TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_GPU,
            Self::Model => TRITONSERVER_instancegroupkind_enum_TRITONSERVER_INSTANCEGROUPKIND_MODEL,
        }
    }
}

/// An instance group the backend prefers, see
/// [`BackendAttributes::preferred_instance_group`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PreferredInstanceGroup {
    pub(crate) kind: InstanceGroupKind,
    pub(crate) count: u64,
    pub(crate) devices: Vec<u64>,
}

/// Attributes of the backend that Triton uses when it loads models, returned
/// by [`crate::CustomBackend::attributes`].
///
/// The default sets nothing: Triton picks instance groups itself and
/// initializes instances one after another.
///
/// A plain value: holds nothing of Triton. `Send + Sync`.
///
/// # Examples
///
/// ```ignore
/// fn attributes(_backend: &Backend) -> Result<BackendAttributes, Error> {
///     Ok(BackendAttributes::new()
///         .preferred_instance_group(InstanceGroupKind::Gpu, 1, &[])?
///         .preferred_instance_group(InstanceGroupKind::Cpu, 2, &[])?
///         .parallel_model_instance_loading(true))
/// }
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BackendAttributes {
    pub(crate) preferred_instance_groups: Vec<PreferredInstanceGroup>,
    pub(crate) parallel_model_instance_loading: bool,
}

impl BackendAttributes {
    /// Attributes that set nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an instance group the backend prefers for models whose config has
    /// no `instance_group`. The first one added is the most preferred.
    ///
    /// `count` is the number of instances per device, `0` leaves it to
    /// Triton. `devices` are the GPUs of a [`InstanceGroupKind::Gpu`] or
    /// [`InstanceGroupKind::Auto`] group, empty leaves them to Triton.
    ///
    /// Triton also takes the count and the GPUs from the first preferred
    /// group of the same kind for an `instance_group` of a config that leaves
    /// them out.
    ///
    /// Fails when `devices` are given for a CPU or model group, repeat a
    /// device, or `count` or a device does not fit the `int32` of the model
    /// config.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let attributes = BackendAttributes::new()
    ///     .preferred_instance_group(InstanceGroupKind::Gpu, 2, &[0, 1])?;
    /// ```
    pub fn preferred_instance_group(
        mut self,
        kind: InstanceGroupKind,
        count: u32,
        devices: &[u32],
    ) -> Result<Self, Error> {
        self.preferred_instance_groups
            .push(preferred_instance_group(kind, count, devices)?);
        Ok(self)
    }

    /// Whether Triton may initialize several instances of a model at once.
    /// With `true`, [`crate::CustomBackend::model_instance_initialize`] and
    /// [`crate::CustomBackend::model_instance_finalize`] of one model run on
    /// several threads at the same time and must not race on shared state.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let attributes = BackendAttributes::new().parallel_model_instance_loading(true);
    /// ```
    pub fn parallel_model_instance_loading(mut self, enabled: bool) -> Self {
        self.parallel_model_instance_loading = enabled;
        self
    }
}

fn preferred_instance_group(
    kind: InstanceGroupKind,
    count: u32,
    devices: &[u32],
) -> Result<PreferredInstanceGroup, String> {
    let fits = |value: u32| i32::try_from(value).is_ok();
    if !fits(count) {
        return Err(format!("instance count {count} does not fit int32"));
    }
    if !devices.is_empty() && !matches!(kind, InstanceGroupKind::Gpu | InstanceGroupKind::Auto) {
        return Err(format!("devices are given for a {kind:?} instance group"));
    }
    for (i, device) in devices.iter().enumerate() {
        if !fits(*device) {
            return Err(format!("device {device} does not fit int32"));
        }
        if devices.iter().take(i).any(|seen| seen == device) {
            return Err(format!("device {device} is given twice"));
        }
    }
    Ok(PreferredInstanceGroup {
        kind,
        count: u64::from(count),
        devices: devices.iter().map(|&device| u64::from(device)).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::{InstanceGroupKind, PreferredInstanceGroup, preferred_instance_group};

    #[test]
    fn group_keeps_its_kind_count_and_devices() {
        assert_eq!(
            preferred_instance_group(InstanceGroupKind::Gpu, 2, &[0, 3]),
            Ok(PreferredInstanceGroup {
                kind: InstanceGroupKind::Gpu,
                count: 2,
                devices: vec![0, 3],
            })
        );
    }

    #[test]
    fn devices_belong_to_gpu_and_auto_groups_only() {
        assert!(preferred_instance_group(InstanceGroupKind::Cpu, 1, &[0]).is_err());
        assert!(preferred_instance_group(InstanceGroupKind::Model, 1, &[0]).is_err());
        assert!(preferred_instance_group(InstanceGroupKind::Cpu, 1, &[]).is_ok());
        assert!(preferred_instance_group(InstanceGroupKind::Auto, 1, &[0]).is_ok());
    }

    #[test]
    fn repeated_device_is_rejected() {
        assert!(preferred_instance_group(InstanceGroupKind::Gpu, 1, &[1, 1]).is_err());
    }

    #[test]
    fn values_beyond_int32_are_rejected() {
        assert!(preferred_instance_group(InstanceGroupKind::Gpu, u32::MAX, &[]).is_err());
        assert!(preferred_instance_group(InstanceGroupKind::Gpu, 1, &[u32::MAX]).is_err());
    }
}
