//! Model and instance memory-reporting fixture; common states also report their sizes.
use super::*;

pub(super) fn reports_memory_check(model: &Model) -> Result<bool, Error> {
    Ok(model.config_parameter("report_memory_check")?.as_deref() == Some("true"))
}

pub(super) fn report_model(model: &Model) -> Result<(), Error> {
    if reports_memory_check(model)? {
        model.report_memory_usage(&[
            BufferAttributes::new(MemoryType::Cpu, CHECK_MODEL_BYTES[0])?,
            BufferAttributes::new(MemoryType::Cpu, CHECK_MODEL_BYTES[1])?,
        ])?;
    } else {
        model.report_memory_usage(&[BufferAttributes::new(
            MemoryType::Cpu,
            size_of::<State>() as u64,
        )?])?;
    }
    Ok(())
}

pub(super) fn initialize_buffers(
    instance: &ModelInstance,
    host_policy: HostPolicy,
) -> Result<(), Error> {
    let model = instance.model()?;
    if !reports_memory_check(&model)? {
        instance.set_state(Buffers {
            host_policy: Some(host_policy),
            ..Default::default()
        })?;
        instance.report_memory_usage(&[BufferAttributes::new(
            MemoryType::Cpu,
            size_of::<Buffers>() as u64,
        )?])?;
        return Ok(());
    }
    // A pinned buffer that really exists, unlike the size of the Rust state.
    let pinned = model.memory_manager()?.pinned(CHECK_PINNED_BYTES)?;
    instance.set_state(Buffers {
        host_policy: Some(host_policy),
        check: Some(pinned),
        ..Default::default()
    })?;
    instance.report_memory_usage(&[BufferAttributes::new(
        MemoryType::CpuPinned,
        CHECK_PINNED_BYTES as u64,
    )?])?;
    Ok(())
}
