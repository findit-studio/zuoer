//! Backend-agnostic voice-activity-detection (VAD) core.
//!
//! `zuoer` is the model-free heart of a VAD pipeline: the [`VadBackend`]
//! seam that accepts PCM and emits speech probabilities through a `sink`,
//! plus the backend-agnostic post-processing that turns a stream of those
//! probabilities into [`SpeechSegment`]s — the [`SpeechSegmenter`]
//! hysteresis state machine, its [`SpeechOptions`] timing/threshold
//! configuration, and the one-shot [`detect_speech_with`] helper.
//!
//! It owns **no model, no inference runtime, and no audio I/O**. A model
//! crate implements [`VadBackend`] over its own inference — for example an
//! ONNX Silero backend or a CoreML backend — and drives the segmenter with
//! this crate's post-processing. The segmentation semantics are the
//! Silero-VAD-derived hysteresis rules; a backend declaring a different
//! frame geometry reuses them unchanged (see [`SpeechSegmenter`]).
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
//! [`SpeechSegmenter::push_probability`] consumes one frame probability and
//! returns any segment it closes directly. A streaming backend driver runs
//! the backend over incoming PCM to obtain frame probabilities, hands them
//! to [`SpeechSegmenter::push_probabilities`] (which buffers closed
//! segments), and drains them in order with
//! [`SpeechSegmenter::pop_pending`] / [`SpeechSegmenter::finish`]. This is
//! the sans-I/O plumbing the backend crates build their probability feeders
//! on top of.
//!
//! # Feature flags
//!
//! - `serde` — derive `Serialize`/`Deserialize` for [`SpeechOptions`] and
//!   [`SampleRate`] (`Duration` fields via `humantime-serde`).
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod backend;
mod detector;
mod error;
mod options;

pub use backend::VadBackend;
pub use detector::{SpeechDetector, SpeechSegment, SpeechSegmenter, detect_speech_with};
pub use error::{Error, Result};
pub use options::{SampleRate, SpeechOptions};
