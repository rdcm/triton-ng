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

/// Logs `message` at the INFO level of the Triton server log.
pub fn info(message: &str) {
    log(
        triton_ng_sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_INFO,
        message,
    );
}

/// Logs `message` at the WARN level of the Triton server log.
pub fn warn(message: &str) {
    log(
        triton_ng_sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_WARN,
        message,
    );
}

/// Logs `message` at the ERROR level of the Triton server log.
pub fn error(message: &str) {
    log(
        triton_ng_sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_ERROR,
        message,
    );
}

/// Logs `message` at the VERBOSE level of the Triton server log, shown with `--log-verbose=1` or higher.
pub fn verbose(message: &str) {
    log(
        triton_ng_sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_VERBOSE,
        message,
    );
}
