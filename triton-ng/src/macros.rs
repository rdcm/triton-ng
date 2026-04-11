macro_rules! ffi_call {
    ($expr: expr) => {{
        #[allow(clippy::macro_metavars_in_unsafe)]
        let res = unsafe { $expr };

        if res.is_null() {
            std::result::Result::<(), $crate::error::TritonError>::Ok(())
        } else {
            std::result::Result::<(), $crate::error::TritonError>::Err(
                unsafe { $crate::error::TritonError::new(res) },
            )
        }
    }};
    ($expr: expr, $val: expr) => {{
        #[allow(clippy::macro_metavars_in_unsafe)]
        let res = unsafe { $expr };

        if res.is_null() {
            std::result::Result::<_, $crate::error::TritonError>::Ok($val)
        } else {
            std::result::Result::<_, $crate::error::TritonError>::Err(
                unsafe { $crate::error::TritonError::new(res) },
            )
        }
    }};
}

macro_rules! ensure_ptr {
    ($ptr:expr) => {{
        if $ptr.is_null() {
            std::result::Result::<_, $crate::error::TritonError>::Err(
                $crate::error::TritonError::from_message(concat!(
                    "Pointer is null: ",
                    stringify!($ptr)
                )),
            )
        } else {
            std::result::Result::<_, $crate::error::TritonError>::Ok($ptr)
        }
    }};
}
