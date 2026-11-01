use crate::memory_manager::MemoryType;
use std::ffi::c_void;

/// Memory Triton reads or writes: where it is, its address and size. A
/// descriptor of memory owned by someone else: it frees nothing and is
/// neither `Send` nor `Sync`.
#[derive(Clone, Copy)]
pub struct RawBuffer {
    pub(crate) ptr: *mut c_void,
    pub(crate) len: usize,
    pub(crate) memory_type: MemoryType,
    pub(crate) memory_type_id: i64,
}

/// Seals [`crate::InputBuffer`]: the module is private, so no type outside
/// the crate can implement it.
pub trait Sealed {
    fn raw(&self) -> RawBuffer;
}
