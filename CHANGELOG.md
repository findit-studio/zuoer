# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `RunSegmenter::push_probability` now canonicalizes every frame probability
  into `[0, 1]` before either the hysteresis comparisons or the aggregates see
  it, so `Run::mean_probability` / `Run::peak_probability` on an emitted run
  are always finite and in range. `VadBackend` documented `[0, 1]` but nothing
  enforced it: a `NaN` frame inside an active run — including one bridged by
  `min_gap_duration`, or flushed by `finish` — made the emitted mean `NaN`.
  Before the 0.2 aggregates a malformed probability only ever reached a
  threshold comparison and degraded safely to silence; the aggregates let it
  escape as a value a consumer reads.
  - `NaN` maps to `0.0`. Every comparison against `NaN` is false, so such a
    frame already behaved as below-threshold — `0.0` preserves the 0.1
    segmentation rather than changing it. (`f32::clamp` returns `NaN` for a
    `NaN` input, so the mapping is explicit rather than a bare clamp.)
  - Infinities and finite out-of-range values are clamped: at or above `1.0`
    to `1.0`, at or below `0.0` to `0.0`.
  - One behavioural exception, now a pinned contract — and it is **not** about
    `NaN`. It covers every input whose canonical value is `0.0`: `NaN`,
    `f32::NEG_INFINITY`, and any finite negative alike. All of them are now
    compared as `0.0`, and `0.0 >= 0.0` holds where the raw comparison did
    not, so against a threshold of exactly `0.0` such a frame opens a run
    (zero `start_threshold` — including the continuation run a
    `max_run_duration` force-split decides on from the same comparison) or
    sustains one (zero effective `end_threshold`) where it previously did
    neither. Above-range inputs have no matching exception: `1.0` and
    everything above it compare identically against any threshold in
    `[0, 1]`. Every configuration with positive thresholds — including all
    defaults — segments identically.
- **serde:** `RunOptions` / `SpeechOptions` deserialization no longer bypasses
  threshold sanitization. `set_start_threshold` / `set_end_threshold` sanitize
  into `[0, 1]`, but the derived `Deserialize` wrote fields directly, so a
  persisted profile carrying a negative or non-finite threshold installed a
  value no setter would have stored — `start_threshold` was then read back
  raw, and a raw `end_threshold` re-serialized as garbage (or as `null`, for a
  non-finite value JSON cannot represent). Both fields now run the same
  `sanitize_probability` on the way in, so the setter path and the serde path
  store the same value for the same input.
- **serde:** a default-constructed `RunOptions` now survives its own
  round-trip. `max_run_duration` is `skip_serializing_if = "Option::is_none"`
  and carries a `deserialize_with` (through `humantime_serde::option`), which
  suppresses serde's implicit "a missing `Option` field is `None`" rule — so
  serializing any options value with no `max_run_duration` (the default)
  produced JSON that failed to deserialize with ``missing field
  `max_run_duration` ``. Both `Option` fields now carry an explicit
  `#[serde(default)]`.

## [0.2.0]

### Added

- `RunSegmenter` / `Run` / `RunOptions` — the hysteresis state machine lifted
  out of `SpeechSegmenter` into a domain-neutral core. Input is a
  frame-probability sequence whose timeline unit is `frame_hop`; output is
  contiguous runs. Threshold hysteresis, min-duration, min-gap bridging,
  force-splitting, the incremental `push_probability -> Option<Run>` close, the
  `push_probabilities` / `pop_pending` / `finish` / `reset` seam, and the
  `set_sample_rate` / `set_frame_hop` time conversions are unchanged — the 23
  segmenter tests and 5 options tests from 0.1 pass verbatim.
- `Run::mean_probability` / `Run::peak_probability` — mean and peak of the
  frame probabilities the run was built from, accumulated in O(1) per frame.
  The single source for VAD segment confidence and sound-event confidence.
  The aggregated span is the run's **raw model-frame span**:
  - `pad` extension is **excluded** — padding is a timeline courtesy, not an
    observation.
  - frames bridged by `min_gap_duration` are **included**; they occurred inside
    the run and correctly pull the mean down.
  - a `max_run_duration` force-split cuts the accumulator too: the emitted run
    carries what was accumulated up to the split point, and the continuation
    restarts from its own resume point. Frames inside the gap that the split
    landed on belong to neither run and appear in neither aggregate.
- `Run::with_mean_probability` / `Run::with_peak_probability` — attach
  aggregates to a hand-built `Run`. `Run::new` leaves both at `0.0`.

### Changed

- **Breaking:** `SpeechSegment` no longer implements `Eq`. It now carries
  `f32` aggregates, which have no total equality. `PartialEq` is unchanged and
  now also compares the aggregates.
- **Breaking (serde):** `RunOptions` fields serialize under their neutral names
  (`min_run_duration`, `min_gap_duration`, `min_gap_at_max_run`,
  `max_run_duration`, `pad`). Each accepts its 0.1 speech-flavoured name as a
  deserialization alias, so 0.1-era configuration profiles still load; only the
  serialized output changed.
- `SpeechSegment`, `SpeechSegmenter`, `SpeechDetector`, and `SpeechOptions` are
  now plain type aliases for `Run`, `RunSegmenter`, and `RunOptions`. The
  speech-flavoured options accessors (`min_speech_duration`, `speech_pad`, the
  `*_samples` getters, and their `with_*` / `set_*` pairs) are kept as
  forwarding accessors. Source-compatible: no consumer needs an edit.

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
