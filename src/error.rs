/// Errors that can occur in this crate and in the backends it drives.
///
/// Marked `#[non_exhaustive]` because the set of variants grows as new
/// backends bridge their errors through [`Error::Backend`], and consumer
/// crates layer their own typed variants on top (for example an ONNX
/// model backend adds ORT-typed model-load / inference variants);
/// downstream `match`es must include a `_` arm.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
  /// An error raised by a [`VadBackend`] implementation.
  ///
  /// The transparent bridge for the [`VadBackend::Error`] associated
  /// type: the detector wraps a backend's error here so a single
  /// [`Error`] type covers every backend. Its [`Display`] and
  /// [`source`](std::error::Error::source) delegate to the wrapped
  /// error.
  ///
  /// [`VadBackend`]: crate::VadBackend
  /// [`VadBackend::Error`]: crate::VadBackend::Error
  /// [`Display`]: std::fmt::Display
  #[error(transparent)]
  Backend(Box<dyn std::error::Error + Send + Sync + 'static>),

  /// Errors related to unsupported or incompatible sample rates.
  #[error("unsupported sample rate: {rate} Hz (only 8 kHz and 16 kHz are supported directly)")]
  UnsupportedSampleRate {
    /// The unsupported sample rate in Hz.
    rate: u32,
  },

  /// Errors related to a sample-rate mismatch between the current operation/component and the provided stream or input.
  #[error(
    "stream sample rate {actual} Hz does not match expected {expected} Hz for this operation"
  )]
  IncompatibleSampleRate {
    /// The sample rate in Hz required by the current operation or component.
    expected: u32,
    /// The sample rate in Hz provided by the stream or input that caused the mismatch.
    actual: u32,
  },

  /// Errors related to invalid chunk lengths that do not match the expected chunk size for the sample rate.
  #[error("invalid chunk length: expected {expected} samples, got {actual}")]
  InvalidChunkLength {
    /// The expected chunk length in samples for the given sample rate.
    expected: usize,
    /// The actual chunk length in samples that was provided.
    actual: usize,
  },
}

/// A convenient alias for results that carry this crate's [`Error`] type.
pub type Result<T> = std::result::Result<T, Error>;
