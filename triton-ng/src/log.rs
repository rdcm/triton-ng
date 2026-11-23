//! Logging through the Triton server logger (`TRITONSERVER_LogMessage`),
//! so backend messages end up in the server log with the usual formatting.

use std::ffi::CString;
use std::panic::Location;

/// Builds the NUL-terminated string the C API takes, with the NUL bytes of
/// `text` replaced so they cannot end the string early.
fn c_string(text: &str) -> CString {
    let mut bytes = text.as_bytes().to_vec();
    for byte in &mut bytes {
        if *byte == b'\0' {
            *byte = b'?';
        }
    }

    // SAFETY: the loop replaced every NUL byte, so there is no interior NUL.
    unsafe { CString::from_vec_unchecked(bytes) }
}

#[track_caller]
fn log(level: triton_ng_sys::TRITONSERVER_LogLevel, message: &str) {
    let message = c_string(message);
    let caller = Location::caller();
    let file = c_string(caller.file());
    let line = i32::try_from(caller.line()).unwrap_or(i32::MAX);

    let error = unsafe {
        triton_ng_sys::TRITONSERVER_LogMessage(level, file.as_ptr(), line, message.as_ptr())
    };

    if !error.is_null() {
        unsafe { triton_ng_sys::TRITONSERVER_ErrorDelete(error) };
    }
}

/// Logs `message` at the INFO level of the Triton server log.
#[track_caller]
pub fn info(message: &str) {
    log(
        triton_ng_sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_INFO,
        message,
    );
}

/// Logs `message` at the WARN level of the Triton server log.
#[track_caller]
pub fn warn(message: &str) {
    log(
        triton_ng_sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_WARN,
        message,
    );
}

/// Logs `message` at the ERROR level of the Triton server log.
#[track_caller]
pub fn error(message: &str) {
    log(
        triton_ng_sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_ERROR,
        message,
    );
}

/// Logs `message` at the VERBOSE level of the Triton server log, shown with `--log-verbose=1` or higher.
///
/// The message is formatted before Triton drops it, so guard the construction with
/// [`verbose_enabled`] on a path that runs per request.
#[track_caller]
pub fn verbose(message: &str) {
    log(
        triton_ng_sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_VERBOSE,
        message,
    );
}

/// Whether Triton logs [`verbose`] messages, that is, whether the server runs with
/// `--log-verbose=1` or higher. False when Triton was built without logging.
///
/// The other levels are on unless the hosting server turns them off when it embeds Triton, so
/// they need no predicate.
///
/// # Examples
///
/// ```ignore
/// if log::verbose_enabled() {
///     log::verbose(&format!("request {}: {} elements", request.id()?, elements));
/// }
/// ```
pub fn verbose_enabled() -> bool {
    // SAFETY: the function reads the verbose level of the global logger and takes no pointer.
    unsafe {
        triton_ng_sys::TRITONSERVER_LogIsEnabled(
            triton_ng_sys::TRITONSERVER_loglevel_enum_TRITONSERVER_LOG_VERBOSE,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nul_bytes_in_a_message_are_replaced() {
        let expected = c"plain message?and more";
        let actual = c_string("plain message\u{0}and more");

        assert_eq!(expected, actual.as_c_str());
    }
}
