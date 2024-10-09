use crate::schema::generic_result;
use std::{io, time};
use tonic::codegen::http::uri;
use tonic::metadata::errors;
use tonic::transport;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("call error (status: {status:?}): {message}")]
    Call { status: Status, message: String },
    #[error("time error: {0}")]
    Time(#[from] time::SystemTimeError),
    #[error("decode error: {0}")]
    ProtoDecode(#[from] prost::DecodeError),
    #[error("gRPC status error (status: {0})")]
    GrpcStatus(#[from] tonic::Status),
    #[error("gRPC transport error: {0}")]
    GrpcTransport(#[from] transport::Error),
    #[error("gRPC invalid metadata value: {0}")]
    GrpcInvalidMetadataValue(#[from] errors::InvalidMetadataValue),
    #[error("gRPC invalid URI: {0}")]
    GrpcInvalidUri(#[from] uri::InvalidUri),
    #[error("I/O error: {0}")]
    IoError(#[from] io::Error),
    #[error("TOML decode error: {0}")]
    TomlDecodeError(#[from] toml::de::Error),
    #[error("unsupported value encoding")]
    UnsupportedValueEncoding,
    #[error("HOME env var not set")]
    HomeEnvVarNotSet,
    #[error("no active config found")]
    NoActiveConfigFound,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Status {
    Unspecified = 0,
    Success = 1,
    Failure = 2,
    /// Used when a task was killed using an external signal.
    Terminated = 3,
    Timeout = 4,
    /// Used when the user's function fails to initialize (ex. S3 mount failed due to invalid credentials).
    /// Terminates the function and all remaining inputs.
    InitFailure = 5,
}

impl From<generic_result::GenericStatus> for Status {
    fn from(value: generic_result::GenericStatus) -> Self {
        match value {
            generic_result::GenericStatus::Unspecified => Status::Unspecified,
            generic_result::GenericStatus::Success => Status::Success,
            generic_result::GenericStatus::Failure => Status::Failure,
            generic_result::GenericStatus::Terminated => Status::Terminated,
            generic_result::GenericStatus::Timeout => Status::Timeout,
            generic_result::GenericStatus::InitFailure => Status::InitFailure,
        }
    }
}
