# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-09-06

### Added

- `deny_unknown_fields` on `RunOptions` and `SampleRate` under the `serde`
  feature — a misspelled key in a hand-edited JSON/YAML/TOML profile is now
  refused by name at deserialize time instead of being silently ignored.
  This is the last gap in the two types' document faces: both already
  derived `Serialize`/`Deserialize`, defaulted every field
  (`RunOptions::default()` for the whole table, `SampleRate::default()` for
  its own field) from one source shared with `RunOptions::new()`, carried
  `Duration` fields through `humantime-serde` (`"2s"`, `"250ms"`), and
  accepted the 0.1-era speech-flavoured field names as deserialization
  aliases.
- `toml` added as a dev-dependency to round-trip-test the document face
  through TOML alongside the existing `serde_json` coverage (no new
  run-time dependency; the crate stays serde-format-agnostic).

## [0.2.0] - 2026-08-22

### Added

- `RunSegmenter` / `Run` / `RunOptions` — the hysteresis state machine lifted
  out of `SpeechSegmenter` into a domain-neutral core. Input is a
  frame-probability sequence whose timeline unit is `frame_hop`; output is
  contiguous runs. Threshold hysteresis, min-duration, min-gap bridging,
  force-splitting, the incremental `push_probability -> Option<Run>` close, the
  `push_probabilities` / `pop_pending` / `finish` / `reset` seam, and the
  `set_sample_rate` / `set_frame_hop` time conversions carry over from
  `SpeechSegmenter` unchanged — the 23 segmenter tests and 5 options tests from
  0.1 pass verbatim, bodies included. The segmentation *mechanics* are what is
  unchanged; the permitted threshold *range* is not — see the threshold-floor
  entry under **Fixed**.
- `Run::mean_probability` / `Run::peak_probability` — mean and peak of the
  frame probabilities the run was built from, accumulated in O(1) per frame.
  The single source for VAD segment confidence and sound-event confidence. On
  a segmenter-emitted run both are always finite and inside `[0, 1]`, whatever
  the probability producer fed in (see **Fixed**). The aggregated span is the
  run's **raw model-frame span**:
  - `pad` extension is **excluded** — padding is a timeline courtesy, not an
    observation.
  - frames bridged by `min_gap_duration` are **included**; they occurred inside
    the run and correctly pull the mean down.
  - a `max_run_duration` force-split cuts the accumulator too: the emitted run
    carries what was accumulated up to the split point, and the continuation
    restarts from its own resume point. Frames inside the gap that the split
    landed on belong to neither run and appear in neither aggregate.
- `Run::with_mean_probability` / `Run::with_peak_probability` — attach
  aggregates to a hand-built `Run`. `Run::new` leaves both at `0.0`. These
  store what the caller hands them, so the range guarantee above covers
  segmenter-emitted runs only.
- `RunOptions::MIN_THRESHOLD` (`0.01`): the lowest threshold the options type
  will store, and the floor that makes both the strictly-positive-threshold
  and the never-inverted-hysteresis properties hold by construction.

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
- `Error::InvalidChunkLength`'s message drops the `VAD` qualifier: `invalid
  chunk length: expected N samples, got M`. The variant is never constructed by
  this crate — it exists for backends to raise, including the non-speech ones
  the neutral core now serves.

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
  - Canonicalization changes no segmentation, in any configuration. Every
    effective threshold is now strictly positive (see the threshold-floor
    entry below), so a frame canonicalized to `0.0` fails every threshold
    comparison exactly as the raw `NaN` or negative did, and a frame
    canonicalized to `1.0` passes exactly the comparisons the raw
    out-of-range value passed. The only case where the two could ever have
    differed — an effective threshold of exactly `0.0`, where `0.0 >= 0.0`
    holds but `NaN >= 0.0` does not — is no longer reachable.
- Thresholds are clamped into `[RunOptions::MIN_THRESHOLD, 1]` rather than
  `[0, 1]`, on the setter path and the `serde` path alike, where the new
  public `RunOptions::MIN_THRESHOLD` is `0.01`. A `start_threshold` of `0.0`
  was accepted and was pathological: every frame satisfies `>= 0.0`, so every
  frame opened a run, cleared the tentative gap, and restarted it on the same
  frame — pinning the gap age at zero so an active run could be extended for
  as long as frames kept arriving, and never closing it. `0.0` also derived an
  `end_threshold` of `0.01`, i.e. ABOVE the start threshold, inverting the
  hysteresis window. Neither configuration has a legitimate use — "every frame
  is speech" is not a detector — so they are excluded rather than documented.
  Two properties now hold for every options value the crate can produce, and
  the canonicalization contract above rests on the first:
  - every effective threshold is strictly positive;
  - `end_threshold() <= start_threshold()`, because the derived end threshold
    (`start - 0.15`) bottoms out at the same `0.01` the start threshold does.

  A start threshold in `(0.0, 0.01)`, and an explicit end threshold in the
  same band, are lifted to `0.01`; a non-finite threshold now becomes `0.01`
  rather than `0.0`, so a mistyped `-1.0` can no longer land on the most
  permissive setting there is. Callers using the defaults, or any threshold at
  or above `0.01`, are unaffected.
- **serde:** `RunOptions` / `SpeechOptions` deserialization no longer bypasses
  threshold sanitization. `set_start_threshold` / `set_end_threshold` clamp
  into the permitted range, but the derived `Deserialize` wrote fields
  directly, so a persisted profile carrying a negative or non-finite threshold
  installed a value no setter would have stored — `start_threshold` was then
  read back raw, and a raw `end_threshold` re-serialized as garbage (or as
  `null`, for a non-finite value JSON cannot represent). Both fields now run
  the same threshold sanitizer on the way in, so the setter path and the serde
  path store the same value for the same input.
- **serde:** a default-constructed `RunOptions` now survives its own
  round-trip. `max_run_duration` is `skip_serializing_if = "Option::is_none"`
  and carries a `deserialize_with` (through `humantime_serde::option`), which
  suppresses serde's implicit "a missing `Option` field is `None`" rule — so
  serializing any options value with no `max_run_duration` (the default)
  produced JSON that failed to deserialize with ``missing field
  `max_run_duration` ``. Both `Option` fields now carry an explicit
  `#[serde(default)]`.

## [0.1.0] - 2026-07-18

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
