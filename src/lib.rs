//! Domain-neutral run segmenter — any frame-probability sequence to
//! contiguous runs — with a backend-agnostic voice-activity-detection (VAD)
//! shell over it.
//!
//! The segmenter at `zuoer`'s centre is domain-neutral; VAD is the first and
//! best-validated shell over it, and sound-event detection the second.
//! `zuoer` is the model-free half of either pipeline. Two layers:
//!
//! - a **neutral core** — [`RunSegmenter`], the hysteresis state machine
//!   that turns any frame-probability sequence into contiguous [`Run`]s,
//!   configured by [`RunOptions`]. It knows nothing about speech: the same
//!   machine segments sound-event probabilities, one instance per class.
//! - a **VAD shell** — the [`VadBackend`] seam that accepts PCM and emits
//!   speech probabilities through a `sink`, the [`SpeechSegment`] /
//!   [`SpeechSegmenter`] / [`SpeechOptions`] names for the core, and the
//!   one-shot [`detect_speech_with`] helper.
//!
//! It owns **no model, no inference runtime, and no audio I/O**. A model
//! crate implements [`VadBackend`] over its own inference — for example an
//! ONNX Silero backend or a CoreML backend — and drives the segmenter with
//! this crate's post-processing. The segmentation semantics are the
//! Silero-VAD-derived hysteresis rules; a backend declaring a different
//! frame geometry reuses them unchanged (see [`RunSegmenter`]).
//!
//! # Shell correspondence
//!
//! The `Speech*` surface is a set of **plain type aliases** plus
//! forwarding accessors — no wrapper, no conversion, no behavioural
//! difference. Code written against either spelling interoperates freely.
//!
//! | speech name | neutral name |
//! |---|---|
//! | [`SpeechSegment`] | [`Run`] |
//! | [`SpeechSegmenter`] / [`SpeechDetector`] | [`RunSegmenter`] |
//! | [`SpeechOptions`] | [`RunOptions`] |
//! | [`SpeechOptions::min_speech_duration`] | [`RunOptions::min_run_duration`] |
//! | [`SpeechOptions::min_silence_duration`] | [`RunOptions::min_gap_duration`] |
//! | [`SpeechOptions::min_silence_at_max_speech`] | [`RunOptions::min_gap_at_max_run`] |
//! | [`SpeechOptions::max_speech_duration`] | [`RunOptions::max_run_duration`] |
//! | [`SpeechOptions::speech_pad`] | [`RunOptions::pad`] |
//! | [`SpeechOptions::min_speech_samples`] | [`RunOptions::min_run_samples`] |
//! | [`SpeechOptions::min_silence_samples`] | [`RunOptions::min_gap_samples`] |
//! | [`SpeechOptions::min_silence_at_max_speech_samples`] | [`RunOptions::min_gap_at_max_run_samples`] |
//! | [`SpeechOptions::max_speech_samples`] | [`RunOptions::max_run_samples`] |
//! | [`SpeechOptions::speech_pad_samples`] | [`RunOptions::pad_samples`] |
//!
//! Each duration accessor's `with_*` / `set_*` builder pair follows the
//! same mapping (`with_min_speech_duration` →
//! [`with_min_run_duration`](RunOptions::with_min_run_duration), and so
//! on). The segmenter's own method names — `push_probability`,
//! `pop_pending`, `finish`, `reset`, `set_sample_rate`, `set_frame_hop` —
//! and [`Run`]'s accessors are already neutral and are spelled the same in
//! both surfaces.
//!
//! # Backend seam
//!
//! A backend owns its input windowing and its end-of-stream policy:
//! [`VadBackend::push`] feeds PCM and invokes a `sink` callback once per
//! completed model frame (zero, one, or many per call — so overlapping
//! windows and delayed first output are expressible), and
//! [`VadBackend::finish`] applies the trailing-frame policy (zero-pad the
//! last partial frame, or drop it). The detector needs only the backend's
//! [`frame_hop`](VadBackend::frame_hop) — the samples one probability
//! advances the timeline — and its [`sample_rate`](VadBackend::sample_rate).
//! [`detect_speech_with`] drives `push` then `finish` for one-shot offline
//! detection.
//!
//! # Streaming seam
//!
//! [`RunSegmenter::push_probability`] consumes one frame probability and
//! returns any run it closes directly. A streaming backend driver runs
//! the backend over incoming PCM to obtain frame probabilities, hands them
//! to [`RunSegmenter::push_probabilities`] (which buffers closed runs),
//! and drains them in order with [`RunSegmenter::pop_pending`] /
//! [`RunSegmenter::finish`]. This is the sans-I/O plumbing the backend
//! crates build their probability feeders on top of.
//!
//! # Feature flags
//!
//! - `serde` — derive `Serialize`/`Deserialize` for [`RunOptions`] and
//!   [`SampleRate`] (`Duration` fields via `humantime-serde`).
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod backend;
mod detector;
mod error;
mod options;
mod run;

pub use backend::VadBackend;
pub use detector::{SpeechDetector, SpeechSegment, SpeechSegmenter, detect_speech_with};
pub use error::{Error, Result};
pub use options::{RunOptions, SampleRate, SpeechOptions};
pub use run::{Run, RunSegmenter};

/// Compile and run the `README.md` examples as doctests.
///
/// The README is the crate's front page and its examples are the first
/// thing a reader copies, so they are held to the same standard as the
/// rustdoc examples: `cargo test --doc` compiles and runs every fenced
/// `rust` block in it. Gated on `cfg(doctest)` so the README text is not
/// also rendered into the crate documentation, which carries its own
/// hand-written overview above.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
