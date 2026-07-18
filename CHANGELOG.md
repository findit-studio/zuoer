# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

### Added

- Initial release. The backend-agnostic VAD core extracted from the
  [`silero`](https://github.com/Findit-AI/silero) crate (0.5.0), with full
  git history preserved:
  - `VadBackend` — the push-based backend seam (`frame_hop`,
    `sample_rate`, `push`, `finish`, `reset`, associated `Error`): the
    backend owns its input windowing (overlapping windows and delayed
    first output are expressible) and its end-of-stream trailing-frame
    policy (zero-pad or drop), emitting probabilities through a `sink`.
  - `SpeechSegmenter` / `SpeechDetector` — the Silero-VAD-derived hysteresis
    state machine that turns frame probabilities into `SpeechSegment`s, with
    the `push_probabilities` / `pop_pending` streaming seam.
  - `SpeechOptions` — segmentation timing and threshold configuration
    (`serde`-optional).
  - `SampleRate` — the 8 kHz / 16 kHz sample-rate contract.
  - `detect_speech_with` — one-shot offline detection over any `VadBackend`.
  - `Error` / `Result` — the backend-free VAD error surface, with the
    transparent `Error::Backend` bridge for backend-specific errors.
