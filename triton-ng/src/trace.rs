use crate::TritonError;
use crate::error::ErrorCode;
use crate::request::Request;
use libc::c_char;
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::ptr;
use triton_ng_macros::triton_call;

/// The trace of a [`Request`], from [`Request::trace`].
///
/// Owns nothing: Triton keeps the `TRITONSERVER_InferenceTrace` and frees it
/// with the request, so the trace cannot outlive the borrow of the request it
/// came from. Neither `Send` nor `Sync`, like the request.
///
/// A request that Triton does not sample, and a server built without tracing,
/// give a trace whose [`Trace::report_activity`] does nothing and reports
/// success, so a backend can report activities without checking first, and
/// whose [`Trace::context`] is `None`.
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

    /// The context Triton keeps for the trace: a JSON object with the W3C
    /// `traceparent` and `tracestate` of the server span, for a model that
    /// reports spans of its own into the same trace.
    ///
    /// `None` when the server runs in `triton` mode, when the request is not
    /// sampled, and when Triton was built without tracing: there is no server
    /// span to join, and a fresh context is the right start.
    ///
    /// The JSON is copied, because [`Trace::report_activity`] replaces the
    /// string Triton keeps, so a slice borrowed from it would not stay valid.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let trace = request.trace()?;
    /// let parent = match trace.context()? {
    ///     Some(json) => propagator.extract(&JsonCarrier::new(&json)),
    ///     None => Context::new(),
    /// };
    /// let span = tracer.start_with_context("preprocess", &parent);
    /// ```
    pub fn context(&self) -> Result<Option<String>, TritonError> {
        if self.ptr.is_null() {
            return Ok(None);
        }

        let mut context: *const c_char = ptr::null();
        // SAFETY: `ptr` is the trace of the request this value borrows, which
        // Triton keeps alive until the request is released; reading the
        // context only reads it.
        triton_call!(triton_ng_sys::TRITONSERVER_InferenceTraceContext(
            self.ptr,
            &mut context
        ))?;
        // SAFETY: the context string belongs to the trace, which is alive for
        // the whole call, and the copy taken here is the one that is returned.
        unsafe { context_string(context) }
            .map_err(|message| TritonError::with_code(ErrorCode::Internal, message))
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

/// The context of a trace, `None` when the trace has none.
///
/// Carries text rather than a [`TritonError`], so the unit tests check the
/// conversion without a Triton server.
///
/// # Safety
///
/// `ptr` must be null or a NUL-terminated string that stays alive and unchanged
/// for the whole call.
unsafe fn context_string(ptr: *const c_char) -> Result<Option<String>, String> {
    if ptr.is_null() {
        return Ok(None);
    }
    let context = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|_| "the context of a trace is not UTF-8".to_string())?
        .to_owned();
    Ok((!context.is_empty()).then_some(context))
}

#[cfg(test)]
mod tests {
    use super::{activity_name, context_string};
    use std::ffi::CString;
    use std::ptr;

    #[test]
    fn a_trace_context_is_copied_and_an_empty_one_is_none() -> Result<(), String> {
        let empty = CString::new("").map_err(|e| e.to_string())?;
        let json = CString::new(r#"{"traceparent":"00-abcd","tracestate":""}"#)
            .map_err(|e| e.to_string())?;
        let invalid = CString::new(vec![0x66u8, 0xff]).map_err(|e| e.to_string())?;

        let expected = (
            Ok(None),
            Ok(None),
            Ok(Some(
                r#"{"traceparent":"00-abcd","tracestate":""}"#.to_string(),
            )),
            Err("the context of a trace is not UTF-8".to_string()),
        );
        // SAFETY: each pointer is either null or the C string of a binding that
        // outlives the call.
        let actual = unsafe {
            (
                context_string(ptr::null()),
                context_string(empty.as_ptr()),
                context_string(json.as_ptr()),
                context_string(invalid.as_ptr()),
            )
        };
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn a_name_the_trace_collector_cannot_write_is_refused() {
        let mut actual_0 = Vec::new();
        let mut expected_0 = Vec::new();
        let mut actual_1 = Vec::new();
        let mut expected_1 = Vec::new();

        for name in [
            "tok\0enize",
            "decode \"first\"",
            "path\\to\\activity",
            "line\nbreak",
            "tab\there",
        ] {
            actual_0.push(activity_name(name).is_err());
            expected_0.push(true);
        }
        actual_1.push(activity_name("decode (batch 1)").ok());
        expected_1.push(CString::new("decode (batch 1)").ok());
        let expected = (expected_0, expected_1);
        let actual = (actual_0, actual_1);
        assert_eq!(actual, expected);
    }

    #[test]
    fn a_timestamp_is_taken_from_the_monotonic_clock() -> Result<(), String> {
        let first = crate::batch_statistics::now_ns();
        let second = crate::batch_statistics::now_ns();
        let expected = (true, true);
        let actual = (second >= first, second < 1_577_836_800_000_000_000);
        assert_eq!(actual, expected);
        Ok(())
    }
}
