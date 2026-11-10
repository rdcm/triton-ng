use crate::TritonError;
use crate::error::ErrorCode;
use crate::request::Request;
use std::ffi::CString;
use std::marker::PhantomData;
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
    /// The name is written into the JSON of the trace collector as it is, so a
    /// name that would break that JSON is refused: no double quote, no
    /// backslash, no NUL byte and no other control character. Triton reports
    /// `INVALID_ARG` then, and nothing is recorded.
    ///
    /// A request that is not traced reports success without doing anything,
    /// after the name was checked.
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
        let timestamp = crate::batch_statistics::now_ns();
        triton_call!(triton_ng_sys::TRITONSERVER_InferenceTraceReportActivity(
            self.ptr,
            timestamp,
            name.as_ptr()
        ))
    }
}

/// The C string of an activity name, a message when the trace collector cannot
/// write it into its JSON.
///
/// Carries text rather than a [`TritonError`], so the unit tests check the
/// validation without a Triton server.
fn activity_name(activity: &str) -> Result<CString, String> {
    if let Some(c) = activity
        .chars()
        .find(|c| c.is_control() || *c == '"' || *c == '\\')
    {
        return Err(format!(
            "an activity name must not contain {c:?}: the trace collector writes the name into a \
             JSON string without escaping it"
        ));
    }
    CString::new(activity).map_err(|_| "an activity name must not contain a NUL byte".to_string())
}

#[cfg(test)]
mod tests {
    use super::activity_name;
    use std::ffi::CString;

    #[test]
    fn a_name_the_trace_collector_cannot_write_is_refused() {
        for name in [
            "tok\0enize",
            "decode \"first\"",
            "path\\to\\activity",
            "line\nbreak",
            "tab\there",
        ] {
            assert!(activity_name(name).is_err(), "{name:?} was accepted");
        }
        assert_eq!(
            activity_name("decode (batch 1)").ok(),
            CString::new("decode (batch 1)").ok()
        );
    }

    #[test]
    fn a_timestamp_is_taken_from_the_monotonic_clock() -> Result<(), String> {
        let first = crate::batch_statistics::now_ns();
        let second = crate::batch_statistics::now_ns();
        assert!(second >= first, "{second} < {first}");
        // The clock counts from boot, like the one Triton stamps its own events
        // with: a Unix timestamp is three orders of magnitude above it and would
        // put every activity after the end of its request.
        assert!(second < 1_577_836_800_000_000_000, "{second}");
        Ok(())
    }
}
