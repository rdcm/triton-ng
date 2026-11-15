#[path = "common/mod.rs"]
mod common;

use crate::common::sut::Sut;
use anyhow::Result;
use tonic::Code;

#[tokio::test]
async fn custom_backend_answers_with_the_requested_error_code() -> Result<()> {
    let sut = Sut::new("fixture_errors", &[]).await?;

    let mut actual = Vec::new();
    let mut expected = Vec::new();
    for (code, expected_code) in [
        ("INTERNAL", Code::Internal),
        ("NOT_FOUND", Code::NotFound),
        ("INVALID_ARG", Code::InvalidArgument),
        ("UNAVAILABLE", Code::Unavailable),
        ("UNSUPPORTED", Code::Unimplemented),
        ("ALREADY_EXISTS", Code::AlreadyExists),
        ("CANCELLED", Code::Cancelled),
    ] {
        let actual_code = sut
            .grpc_code_of_failed_inference("error_code", code)
            .await?;
        actual.push(actual_code);
        expected.push(expected_code);
    }
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn custom_backend_passes_on_the_code_of_a_failed_bls_call() -> Result<()> {
    let sut = Sut::new("fixture_errors", &[]).await?;

    let code = sut
        .grpc_code_of_failed_inference("infer_model", "missing_model")
        .await?;

    let actual = code;
    let expected = Code::NotFound;
    assert_eq!(actual, expected);
    Ok(())
}
