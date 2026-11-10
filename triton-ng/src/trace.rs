use crate::TritonError;
use crate::error::ErrorCode;
use crate::request::Request;
use std::ffi::CString;
use std::marker::PhantomData;
use std::time::{SystemTime, UNIX_EPOCH};
use triton_ng_macros::triton_call;

/// The trace of a [`Request`], from [`Request::trace`].
///
/// Owns nothing: Triton keeps the `TRITONSERVER_InferenceTrace` and frees it
/// with the request, so the trace cannot outlive the borrow of the request it
/// came from. Neither `Send` nor `Sync`, like the request.
///
/// A request that Triton does not sample, and a server built without tracing,
/// give a trace whose [`Trace::report_activity`] does nothing and reports
/// success, so a backend can report activities without checking first.
///
/// # Examples
///
/// ```ignore
/// let trace = request.trace()?;
/// trace.report_activity("tokenize_start")?;
/// let tokens = tokenize(text)?;
/// trace.report_activity("tokenize_end")?;
/// ```
#[derive(Debug)]
pub struct Trace<'a> {
    ptr: *mut triton_ng_sys::TRITONSERVER_InferenceTrace,
    request: PhantomData<&'a Request>,
}

impl<'a> Trace<'a> {
    /// The trace behind `ptr`, null when the request is not traced or Triton
    /// was built without tracing.
    pub(crate) fn from_ptr(ptr: *mut triton_ng_sys::TRITONSERVER_InferenceTrace) -> Self {
        Self {
            ptr,
            request: PhantomData,
        }
    }

    /// Reports an activity of the backend into the trace, under the name given.
    ///
    /// Triton shows the activity next to the ones it reports itself, timed with
    /// the moment of the call, so a backend can show where the time inside a
    /// request goes.
    ///
    /// Fails when the name contains a NUL byte or when Triton rejects the
    /// report; a request that is not traced reports success without doing
    /// anything.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// request.trace()?.report_activity("preprocess")?;
    /// ```
    pub fn report_activity(&self, activity: &str) -> Result<(), TritonError> {
        let name = activity_name(activity)
            .map_err(|message| TritonError::with_code(ErrorCode::InvalidArg, message))?;
        if self.ptr.is_null() {
            return Ok(());
        }

        // SAFETY: `ptr` is the trace of the request this value borrows, which
        // Triton keeps alive until the request is released; reporting only
        // reads it.
        let timestamp = now_nanos().map_err(TritonError::from_message)?;
        triton_call!(triton_ng_sys::TRITONSERVER_InferenceTraceReportActivity(
            self.ptr,
            timestamp,
            name.as_ptr()
        ))
    }
}

/// The C string of an activity name, a message when it contains a NUL byte.
///
/// Carries text rather than a [`TritonError`], so the unit tests check the
/// validation without a Triton server.
fn activity_name(activity: &str) -> Result<CString, String> {
    CString::new(activity).map_err(|_| "an activity name must not contain a NUL byte".to_string())
}

/// The timestamp of a report: nanoseconds since the epoch, the unit of the
/// timestamps Triton reports itself.
///
/// Carries text rather than a [`TritonError`], so the unit test checks it
/// without a Triton server.
fn now_nanos() -> Result<u64, String> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("the clock is before the epoch: {e}"))?;
    Ok(elapsed.as_nanos() as u64)
}

#[cfg(test)]
mod tests {
    use super::{activity_name, now_nanos};
    use std::ffi::CString;

    #[test]
    fn a_name_with_a_nul_byte_is_refused() {
        assert!(activity_name("tok\0enize").is_err());
        assert_eq!(
            activity_name("tokenize").ok(),
            CString::new("tokenize").ok()
        );
    }

    #[test]
    fn a_timestamp_is_nanoseconds_since_the_epoch() -> Result<(), String> {
        let nanos = now_nanos()?;
        // Later than 2020 and earlier than 2100, in nanoseconds.
        assert!(nanos > 1_577_836_800_000_000_000, "{nanos}");
        assert!(nanos < 4_102_444_800_000_000_000, "{nanos}");
        Ok(())
    }
}
