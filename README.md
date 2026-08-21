<div align="center">
<h1>zuoer</h1>
</div>
<div align="center">

Backend-agnostic voice-activity-detection (VAD) core.

[<img alt="github" src="https://img.shields.io/badge/github-findit--studio/zuoer-8da0cb?style=for-the-badge&logo=Github" height="22">][Github-url]
[<img alt="Build" src="https://img.shields.io/github/actions/workflow/status/findit-studio/zuoer/ci.yml?logo=Github-Actions&style=for-the-badge" height="22">][CI-url]
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-zuoer-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="22">][doc-url]
[<img alt="crates.io" src="https://img.shields.io/crates/v/zuoer?style=for-the-badge&logo=rust" height="22">][crates-url]
<img alt="license" src="https://img.shields.io/badge/License-Apache%202.0/MIT-blue.svg?style=for-the-badge" height="22">

</div>

## Introduction

`zuoer` is the model-free heart of a VAD pipeline, built on a domain-neutral
core. It provides:

- the **neutral run segmenter** — `RunSegmenter`, the hysteresis state machine
  that turns any frame-probability sequence into contiguous `Run`s, configured
  by `RunOptions`. It knows nothing about speech: the same machine segments
  sound-event probabilities, one instance per class;
- the **`VadBackend` seam** — the push-based contract (`frame_hop`,
  `sample_rate`, `push`, `finish`, `reset`, and an associated `Error`) that
  feeds PCM to a backend and emits speech probabilities through a `sink`, one
  per completed model frame;
- the **VAD shell** — `SpeechSegment` / `SpeechSegmenter` / `SpeechOptions`,
  plain aliases for the neutral types, plus the one-shot `detect_speech_with`
  helper.

Every emitted `Run` carries the **mean and peak** of the frame probabilities
over its raw model-frame span — padding excluded, bridged frames included,
restarted at a force-split. That is the single source for VAD segment
confidence and sound-event confidence alike. Both are always finite and inside
`[0, 1]`: `push_probability` canonicalizes each frame first — `NaN` to `0.0`
(silence), everything else clamped — and the same canonical value drives the
hysteresis comparisons and the aggregates, so a malformed frame can never
reach a consumer as a `NaN` confidence.

Canonicalization changes no segmentation. Thresholds are clamped into
`[RunOptions::MIN_THRESHOLD, 1]` (`0.01` at the bottom), so a frame
canonicalized to `0.0` satisfies no threshold and a frame canonicalized to
`1.0` satisfies every threshold it already satisfied — exactly as the raw
value compared. The floor also keeps the hysteresis window from inverting:
`end_threshold() <= start_threshold()` for every configuration the setters
and the `serde` path can produce.

### Shell correspondence

The `Speech*` surface is plain type aliases plus forwarding accessors — no
wrapper, no conversion, no behavioural difference:

| speech name | neutral name |
|---|---|
| `SpeechSegment` | `Run` |
| `SpeechSegmenter` / `SpeechDetector` | `RunSegmenter` |
| `SpeechOptions` | `RunOptions` |
| `min_speech_duration` | `min_run_duration` |
| `min_silence_duration` | `min_gap_duration` |
| `min_silence_at_max_speech` | `min_gap_at_max_run` |
| `max_speech_duration` | `max_run_duration` |
| `speech_pad` | `pad` |

The `*_samples` getters and the `with_*` / `set_*` builder pairs follow the
same mapping. The segmenter's own methods (`push_probability`, `pop_pending`,
`finish`, `reset`, `set_sample_rate`, `set_frame_hop`) and `Run`'s accessors
are already neutral and are spelled the same in both surfaces.

It owns **no model, no inference runtime, and no audio I/O**. A model crate
implements `VadBackend` over its own inference — an ONNX Silero backend, a
CoreML backend, or any other — and drives the segmenter with this crate's
post-processing. The segmentation rules are the Silero-VAD-derived hysteresis
semantics; a backend that declares a different frame geometry (via
`VadBackend::frame_hop`) reuses them unchanged.

## Usage

Implement `VadBackend` over your model, then run it one-shot or streaming:

```rust
use zuoer::{SampleRate, SpeechOptions, VadBackend, detect_speech_with};

struct MyBackend { /* model + recurrent state + partial-frame buffer */ }

impl VadBackend for MyBackend {
    type Error = zuoer::Error;

    // Samples one emitted probability advances the timeline (the frame hop).
    fn frame_hop(&self) -> usize { 512 }

    fn sample_rate(&self) -> SampleRate { SampleRate::Rate16k }

    // Feed PCM; invoke `sink` once per completed model frame. A call may
    // emit zero, one, or many probabilities — buffer the trailing partial
    // frame for the next call.
    fn push(&mut self, samples: &[f32], sink: &mut dyn FnMut(f32)) -> Result<(), Self::Error> {
        let _ = (samples, sink); // run inference over each complete window
        Ok(())
    }

    // End-of-stream: emit any trailing probability your policy produces —
    // zero-pad the last partial frame, or drop it (snip_edges) and emit
    // nothing.
    fn finish(&mut self, sink: &mut dyn FnMut(f32)) -> Result<(), Self::Error> {
        let _ = sink;
        Ok(())
    }

    fn reset(&mut self) { /* clear recurrent state + partial-frame buffer */ }
}

let mut backend = MyBackend { /* .. */ };
let audio = vec![0.0_f32; 512 * 16];
let segments = detect_speech_with(&mut backend, &audio, SpeechOptions::default())?;
# Ok::<(), zuoer::Error>(())
```

For streaming, feed frame probabilities to `SpeechSegmenter::push_probabilities`
and drain closed segments with `pop_pending` (and `finish` at end-of-stream).

### The neutral core, without audio

Nothing above the frame probabilities is required. Drive `RunSegmenter` with
`RunOptions` directly and it segments any per-frame score — one instance per
sound-event class, no `VadBackend` and no PCM in sight. Every emitted `Run`
carries the mean and peak of the frames it was built from, which is where a
consumer reads its confidence:

```rust
use std::time::Duration;

use zuoer::{RunOptions, RunSegmenter, SampleRate};

// One segmenter per class. The input is whatever per-frame probabilities
// the classifier produced; the timeline unit is the classifier's hop.
let options = RunOptions::default()
    .with_sample_rate(SampleRate::Rate16k)
    .with_start_threshold(0.6) // end threshold derives as 0.45
    .with_min_run_duration(Duration::from_millis(40))
    .with_min_gap_duration(Duration::from_millis(40))
    .with_pad(Duration::ZERO);

let mut dog_bark = RunSegmenter::new(options);
dog_bark.set_frame_hop(320); // 20 ms per frame at 16 kHz

dog_bark.push_probabilities(&[0.1, 0.9, 0.8, 0.95, 0.2, 0.05, 0.0]);

// The run opened on the 0.9 frame and closed once the trailing gap
// reached `min_gap_duration`.
let run = dog_bark.pop_pending().expect("the gap closed the run");
assert_eq!(run.start_sample(), 320); // 0.02 s
assert_eq!(run.end_sample(), 1280); // 0.08 s

// The aggregates span the run's raw frames only: the three above-threshold
// frames, not the sub-threshold gap that closed it and not any padding.
assert_eq!(run.peak_probability(), 0.95);
assert!((run.mean_probability() - 0.883).abs() < 1e-3);

// Nothing was left open at end of stream.
assert!(dog_bark.finish().is_none());
```

## Feature flags

- `serde` — derive `Serialize`/`Deserialize` for `RunOptions` and `SampleRate`
  (`Duration` fields via `humantime-serde`). Fields serialize under their
  neutral names and also accept the 0.1 speech-flavoured names as
  deserialization aliases.

## Consumers

- [`silero`](https://github.com/Findit-AI/silero) — the ONNX Silero VAD backend.
- [`coremlit`](https://github.com/findit-studio/coremlit) — the CoreML VAD
  backend (not yet published to crates.io).

#### License

`zuoer` is under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE), [LICENSE-MIT](LICENSE-MIT) for details.

Copyright (c) 2026 FinDIT studio authors.

[Github-url]: https://github.com/findit-studio/zuoer
[CI-url]: https://github.com/findit-studio/zuoer/actions/workflows/ci.yml
[doc-url]: https://docs.rs/zuoer
[crates-url]: https://crates.io/crates/zuoer
