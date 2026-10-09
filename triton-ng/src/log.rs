//! Logging through the Triton server logger (`TRITONSERVER_LogMessage`),
//! so backend messages end up in the server log with the usual formatting.

use std::ffi::CString;

fn log(level: triton_ng_sys::TRITONSERVER_LogLevel, message: &str) {
    let Ok(message) = CString::new(message.replace('\0', "?")) else {
        return;
    };

    let error = unsafe {
        triton_ng_sys::TRITONSERVER_LogMessage(level, c"triton-ng".as_ptr(), 0, message.as_ptr())
    };

    if !error.is_null() {
        unsafe { triton_ng_sys::TRITONSERVER_ErrorDelete(error) };
    }
}

pub fn info(message: &str) {
    log(
        triton_ng_sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_INFO,
        message,
    );
}

pub fn warn(message: &str) {
    log(
        triton_ng_sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_WARN,
        message,
    );
}

pub fn error(message: &str) {
    log(
        triton_ng_sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_ERROR,
        message,
    );
}

pub fn verbose(message: &str) {
    log(
        triton_ng_sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_VERBOSE,
        message,
    );
}
