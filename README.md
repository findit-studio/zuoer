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

`zuoer` is the model-free heart of a VAD pipeline. It provides:

- the **`VadBackend` seam** — the minimal per-frame contract (`frame_samples`,
  `sample_rate`, `predict`, `reset`, and an associated `Error`) that turns one
  exact-size frame of PCM into a single speech probability;
- the **backend-agnostic post-processing** that turns a stream of those
  probabilities into speech segments: the `SpeechSegmenter` hysteresis state
  machine, its `SpeechOptions` timing/threshold configuration, `SpeechSegment`,
  and the one-shot `detect_speech_with` helper.

It owns **no model, no inference runtime, and no audio I/O**. A model crate
implements `VadBackend` over its own inference — an ONNX Silero backend, a
CoreML backend, or any other — and drives the segmenter with this crate's
post-processing. The segmentation rules are the Silero-VAD-derived hysteresis
semantics; a backend that declares a different frame geometry (via
`VadBackend::frame_samples`) reuses them unchanged.

## Usage

Implement `VadBackend` over your model, then run it one-shot or streaming:

```rust
use zuoer::{SampleRate, SpeechOptions, VadBackend, detect_speech_with};

struct MyBackend { /* model + recurrent state */ }

impl VadBackend for MyBackend {
    type Error = zuoer::Error;
    fn frame_samples(&self) -> usize { 512 }
    fn sample_rate(&self) -> SampleRate { SampleRate::Rate16k }
    fn predict(&mut self, frame: &[f32]) -> Result<f32, Self::Error> {
        // run inference over exactly `frame_samples()` samples
        Ok(0.0)
    }
    fn reset(&mut self) { /* clear recurrent state */ }
}

let mut backend = MyBackend { /* .. */ };
let audio = vec![0.0_f32; 512 * 16];
let segments = detect_speech_with(&mut backend, &audio, SpeechOptions::default())?;
# Ok::<(), zuoer::Error>(())
```

For streaming, feed frame probabilities to `SpeechSegmenter::push_probabilities`
and drain closed segments with `pop_pending` (and `finish` at end-of-stream).

## Feature flags

- `serde` — derive `Serialize`/`Deserialize` for `SpeechOptions` and
  `SampleRate` (`Duration` fields via `humantime-serde`).

## Consumers

- [`silero`](https://github.com/Findit-AI/silero) — the ONNX Silero VAD backend.

#### License

`zuoer` is under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE), [LICENSE-MIT](LICENSE-MIT) for details.

Copyright (c) 2026 FinDIT studio authors.

[Github-url]: https://github.com/findit-studio/zuoer
[CI-url]: https://github.com/findit-studio/zuoer/actions/workflows/ci.yml
[doc-url]: https://docs.rs/zuoer
[crates-url]: https://crates.io/crates/zuoer
