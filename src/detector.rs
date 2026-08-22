use crate::{
  Result,
  backend::VadBackend,
  options::SpeechOptions,
  run::{Run, RunSegmenter},
};

/// One speech segment on the stream timeline.
///
/// A plain alias for the neutral [`Run`]: identical type, identical
/// constructors, identical accessors — including the
/// [`mean_probability`](Run::mean_probability) /
/// [`peak_probability`](Run::peak_probability) aggregates, which are the
/// source of a segment's VAD confidence. See the correspondence table in
/// the [crate docs](crate).
pub type SpeechSegment = Run;

/// The hysteresis state machine that turns speech probabilities into
/// [`SpeechSegment`]s.
///
/// A plain alias for the neutral [`RunSegmenter`], which owns the
/// semantics and the timeline contract. See the correspondence table in
/// the [crate docs](crate).
pub type SpeechSegmenter = RunSegmenter;

/// Backwards-compatible alias for callers that think in
/// "detector" rather than "segmenter" terms.
pub type SpeechDetector = SpeechSegmenter;

/// One-shot offline speech detection over any [`VadBackend`].
///
/// Feeds the whole `samples` buffer to the backend via
/// [`push`](VadBackend::push), then [`finish`](VadBackend::finish), and
/// applies the same segmentation rules as [`SpeechSegmenter`] to every
/// probability the backend emits. Input windowing and the end-of-stream
/// trailing-frame policy (zero-pad the last partial frame, or drop it) are
/// the backend's own — this helper imposes neither. The backend is *not*
/// [`reset`](VadBackend::reset): pass a freshly constructed or reset
/// backend to start a new stream.
///
/// # Sample rate
///
/// The backend is authoritative for its own stream: the segmenter's
/// duration thresholds and emitted [`SpeechSegment`] stamps are taken
/// from [`backend.sample_rate()`](VadBackend::sample_rate), overriding
/// whatever `sample_rate` the passed `options` carried. Configure the
/// rate on the backend, not on `options`, when driving this helper.
///
/// # Errors
///
/// Returns the backend's error, bridged into [`Error`](crate::Error), if
/// any inference fails.
///
/// # Panics
///
/// Panics if the backend reports a zero
/// [`frame_hop`](VadBackend::frame_hop).
pub fn detect_speech_with<B: VadBackend>(
  backend: &mut B,
  samples: &[f32],
  options: SpeechOptions,
) -> Result<Vec<SpeechSegment>> {
  let hop = backend.frame_hop();
  assert!(hop != 0, "VadBackend::frame_hop() must be non-zero");
  let mut segmenter = SpeechSegmenter::new(options);
  // The backend owns its stream's sample rate: align the segmenter's
  // duration conversions and segment stamps to it, overriding the rate
  // the passed options carried. `set_sample_rate` also resets the hop to
  // that rate's model chunk, so re-apply the backend's hop afterward.
  segmenter.set_sample_rate(backend.sample_rate());
  segmenter.set_frame_hop(hop);
  let mut segments = Vec::new();

  // The backend owns input windowing and its end-of-stream policy; drive
  // `push` then `finish`, segmenting each emitted probability through the
  // sink. The sink form keeps a streaming backend's hot path
  // allocation-free — no intermediate probability buffer. The two inline
  // closures are separate so each releases its borrow of `segmenter` /
  // `segments` before the trailing `segmenter.finish()`.
  backend
    .push(samples, &mut |probability| {
      if let Some(segment) = segmenter.push_probability(probability) {
        segments.push(segment);
      }
    })
    .map_err(Into::into)?;
  backend
    .finish(&mut |probability| {
      if let Some(segment) = segmenter.push_probability(probability) {
        segments.push(segment);
      }
    })
    .map_err(Into::into)?;

  if let Some(segment) = segmenter.finish() {
    segments.push(segment);
  }
  Ok(segments)
}

#[cfg(test)]
mod tests {
  use std::time::Duration;

  use crate::{SampleRate, SpeechOptions, VadBackend};

  use super::{SpeechSegment, SpeechSegmenter, detect_speech_with};

  fn frame_count(duration_ms: u32, sample_rate: SampleRate) -> usize {
    let frame_ms = (sample_rate.chunk_samples() as u32 * 1_000) / sample_rate.hz();
    (duration_ms / frame_ms) as usize
  }

  fn collect(segmenter: &mut SpeechSegmenter, probabilities: &[f32]) -> Vec<SpeechSegment> {
    let mut segments = Vec::new();
    for probability in probabilities {
      if let Some(segment) = segmenter.push_probability(*probability) {
        segments.push(segment);
      }
    }
    if let Some(segment) = segmenter.finish() {
      segments.push(segment);
    }
    segments
  }

  #[test]
  fn closes_segment_after_confirmed_silence() {
    let config = SpeechOptions::default();
    let mut segmenter = SpeechSegmenter::new(config.clone());
    let mut probabilities = vec![0.9; frame_count(320, SampleRate::Rate16k)];
    probabilities.extend(vec![0.0; frame_count(128, SampleRate::Rate16k)]);

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(segments.len(), 1);
    assert!(segments[0].start_sample() <= config.speech_pad_samples());
    assert!(segments[0].sample_count() >= config.min_speech_samples());
  }

  #[test]
  fn drops_short_bursts() {
    let config = SpeechOptions::default();
    let mut segmenter = SpeechSegmenter::new(config.clone());
    let mut probabilities = vec![0.9; frame_count(64, SampleRate::Rate16k)];
    probabilities.extend(vec![0.0; frame_count(160, SampleRate::Rate16k)]);
    let segments = collect(&mut segmenter, &probabilities);
    assert!(segments.is_empty());
  }

  #[test]
  fn middle_band_frames_do_not_reset_tentative_end() {
    // Verifies that mid-band probabilities (between the end_threshold and
    // start_threshold, e.g. `0.4` against the default `0.5` start) do NOT
    // reset the silence accumulator — they're treated as "not yet
    // confirmed speech".
    //
    // The segment closes after FIVE consecutive low-or-mid-band frames
    // at the default `min_silence_duration_ms = 100` (1600 samples / 512
    // per frame = 3.125 → 4 prior frames + the close-firing 5th frame),
    // matching upstream Python silero-vad. The upstream `silero` crate
    // closed after FOUR frames (one frame too eager) until its 0.3.0
    // silence-counter off-by-one fix, which this crate inherited.
    let config = SpeechOptions::default()
      .with_min_speech_duration(Duration::ZERO)
      .with_speech_pad(Duration::ZERO)
      .with_min_silence_duration(Duration::from_millis(100));
    let mut segmenter = SpeechSegmenter::new(config);

    let mut probabilities = vec![0.9; 4];
    // Five low/mid frames so the segment closes via push_probability.
    // The mid-band 0.4 frame in the middle must NOT reset the silence
    // accumulator — that's the actual property under test.
    probabilities.extend([0.0, 0.4, 0.0, 0.0, 0.0]);
    probabilities.extend(vec![0.9; 4]);

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].start_sample(), 0);
    assert_eq!(segments[0].end_sample(), 2_048);
    // Segment two starts on the first speech frame after the closed
    // silence (4 high + 5 silence = frame index 9, sample 4_608).
    assert_eq!(segments[1].start_sample(), 4_608);
  }

  #[test]
  fn min_speech_duration_is_checked_before_padding() {
    // A speech burst of 6 frames * 32 ms = 192 ms is shorter than the
    // default `min_speech_duration_ms = 250`, so the segment that the
    // trailing silence closes must be dropped — `min_speech` is checked
    // against the raw speech window (raw_end - raw_start), not against
    // the padded boundaries.
    //
    // Push-based close requires FIVE consecutive low-probability frames
    // at the default `min_silence_duration_ms = 100` (4 in the upstream
    // `silero` crate, until its 0.3.0 silence-counter off-by-one fix).
    // The trailing silence below is therefore 5 frames, not 4, so the
    // close still fires via `push_probability` — otherwise `finish()`
    // would emit the burst-plus-trailing-silence as a single trailing
    // segment that satisfies the 250 ms duration check, which is a
    // different (and correct, but separate) behaviour.
    let config = SpeechOptions::default();
    let mut segmenter = SpeechSegmenter::new(config);

    let mut probabilities = vec![0.0; 4];
    probabilities.extend(vec![0.9; 6]);
    probabilities.extend(vec![0.0; 5]);

    let segments = collect(&mut segmenter, &probabilities);
    assert!(segments.is_empty());
  }

  #[test]
  fn finish_flushes_trailing_active_segment() {
    let config = SpeechOptions::default();
    let mut segmenter = SpeechSegmenter::new(config);
    let probabilities = vec![0.9; frame_count(320, SampleRate::Rate16k)];
    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(segments.len(), 1);
    assert!(segments[0].end_sample() > segments[0].start_sample());
  }

  #[test]
  fn reset_clears_runtime_state() {
    let mut segmenter = SpeechSegmenter::new(SpeechOptions::default());
    let _ = segmenter.push_probability(0.9);
    assert!(segmenter.is_active());
    segmenter.reset();
    assert!(!segmenter.is_active());
  }

  #[test]
  fn set_sample_rate_resets_runtime_state_and_updates_timeline_rate() {
    let mut segmenter = SpeechSegmenter::new(SpeechOptions::default());
    let _ = segmenter.push_probability(0.9);
    assert!(segmenter.is_active());

    segmenter.set_sample_rate(SampleRate::Rate8k);
    assert_eq!(segmenter.sample_rate(), SampleRate::Rate8k);
    assert!(!segmenter.is_active());

    for _ in 0..frame_count(320, SampleRate::Rate8k) {
      let _ = segmenter.push_probability(0.9);
    }
    let segment = segmenter.finish().expect("trailing segment");
    assert_eq!(segment.sample_rate(), SampleRate::Rate8k);
  }

  #[test]
  fn force_splits_long_speech_when_max_duration_is_reached() {
    let config = SpeechOptions::default()
      .with_min_speech_duration(Duration::ZERO)
      .with_speech_pad(Duration::ZERO)
      .with_max_speech_duration(Duration::from_millis(160));
    let mut segmenter = SpeechSegmenter::new(config);
    let probabilities = vec![0.9; 8];

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].start_sample(), 0);
    assert_eq!(segments[0].end_sample(), 2_560);
    assert_eq!(segments[1].start_sample(), 2_560);
    assert_eq!(segments[1].end_sample(), 4_096);
  }

  #[test]
  fn prefers_recorded_silence_when_splitting_long_speech() {
    let config = SpeechOptions::default()
      .with_min_speech_duration(Duration::ZERO)
      .with_speech_pad(Duration::ZERO)
      .with_min_silence_duration(Duration::from_millis(300))
      .with_min_silence_at_max_speech(Duration::from_millis(64))
      .with_max_speech_duration(Duration::from_millis(256));
    let mut segmenter = SpeechSegmenter::new(config);
    let mut probabilities = vec![0.9; 4];
    probabilities.extend(vec![0.0; 4]);
    probabilities.extend(vec![0.9; 4]);

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].start_sample(), 0);
    assert_eq!(segments[0].end_sample(), 2_048);
    assert_eq!(segments[1].start_sample(), 4_096);
    assert_eq!(segments[1].end_sample(), 6_144);
  }

  #[test]
  fn non_qualifying_silence_does_not_overwrite_next_start() {
    let config = SpeechOptions::default()
      .with_min_speech_duration(Duration::ZERO)
      .with_speech_pad(Duration::ZERO)
      .with_min_silence_duration(Duration::from_millis(10_000))
      .with_min_silence_at_max_speech(Duration::from_millis(64))
      .with_max_speech_duration(Duration::from_millis(512));
    let mut segmenter = SpeechSegmenter::new(config);

    let mut probabilities = vec![0.9; 4];
    probabilities.extend(vec![0.0; 4]);
    probabilities.extend(vec![0.9; 4]);
    probabilities.extend(vec![0.0; 1]);
    probabilities.extend(vec![0.9; 20]);

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(segments[0].end_sample(), 2_048);
    assert_eq!(segments[1].start_sample(), 4_096);
  }

  #[test]
  fn force_split_during_silence_closes_without_restarting() {
    // `max_speech_duration` is 256 ms rather than 224 ms so the
    // max-speech split fires one frame later, after `max_split_end` has
    // been recorded by the silence-counter logic. That logic sets
    // `max_split_end` on the 4th low-probability frame, not the 3rd, so
    // a 224 ms ceiling would split at sample 3_584 with
    // `max_split_end == None` — falling back to `frame_start` and
    // closing at sample 3_584 instead of at the recorded silence
    // boundary 2_048. The 256 ms ceiling keeps the property under test
    // reachable: that a force-split during silence closes at the
    // silence boundary, not at the current frame, and does NOT restart
    // a new segment afterwards.
    let config = SpeechOptions::default()
      .with_min_speech_duration(Duration::ZERO)
      .with_speech_pad(Duration::ZERO)
      .with_min_silence_duration(Duration::from_millis(10_000))
      .with_min_silence_at_max_speech(Duration::from_millis(64))
      .with_max_speech_duration(Duration::from_millis(256));
    let mut segmenter = SpeechSegmenter::new(config);

    let mut probabilities = vec![0.9; 4];
    probabilities.extend(vec![0.0; 8]);

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start_sample(), 0);
    assert_eq!(segments[0].end_sample(), 2_048);
  }

  #[test]
  fn four_frame_silence_dip_does_not_close_segment_at_default_min_silence() {
    // Regression guard for the silence-counter off-by-one fix this
    // crate inherited from the upstream `silero` crate (fixed there in
    // its 0.3.0).
    //
    // At the default `min_silence_duration_ms = 100` (1600 samples at
    // 16 kHz) and the default 32 ms / 512-sample frame, upstream Python
    // `silero-vad` (`get_speech_timestamps`) closes a segment after
    // FIVE consecutive low-probability frames — `sil_dur_now =
    // cur_sample - temp_end` is evaluated BEFORE the current frame is
    // consumed, so the comparator sees `(k-1) * 512` on the k-th
    // low-prob frame and only crosses the 1600-sample threshold at
    // k = 5.
    //
    // Before its 0.3.0 fix the upstream `silero` crate evaluated the
    // same counter AFTER the current frame was added to
    // `current_sample`, so it saw `k * 512` and closed at k = 4. A
    // 4-frame (128 ms) silence dip would therefore split a segment in
    // the crate but be tolerated by Python.
    //
    // This test pins the post-fix behaviour: a 4-frame silence dip must
    // be tolerated. The 30-frame speech runs ensure both halves
    // individually clear `min_speech_duration_ms = 250` (8 frames),
    // so neither would be dropped by the min-speech filter if the
    // segment did split.
    let config = SpeechOptions::default();
    let mut segmenter = SpeechSegmenter::new(config.clone());

    let mut probabilities = vec![1.0; 30];
    probabilities.extend(vec![0.0; 4]);
    probabilities.extend(vec![1.0; 30]);

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(
      segments.len(),
      1,
      "4-frame silence dip must be tolerated at default min_silence_duration_ms = 100; \
       got {} segments",
      segments.len()
    );
    // Sanity: the (one) segment must start at 0 (the start-pad
    // saturates against the timeline's zero) and span the full
    // 30 + 4 + 30 = 64 frame window — at 512 samples / frame, that
    // ends at 32_768.
    assert_eq!(segments[0].start_sample(), 0);
    assert_eq!(segments[0].end_sample(), 32_768);
  }

  #[test]
  fn five_frame_silence_dip_closes_segment_at_default_min_silence() {
    // Companion to `four_frame_silence_dip_does_not_close_segment_*`.
    // At the same defaults, FIVE consecutive low-prob frames must close
    // the segment — matching upstream Python silero-vad's
    // `sil_dur_now >= 1600` firing on the 5th frame.
    let config = SpeechOptions::default();
    let mut segmenter = SpeechSegmenter::new(config);

    let mut probabilities = vec![1.0; 30];
    probabilities.extend(vec![0.0; 5]);
    probabilities.extend(vec![1.0; 30]);

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(
      segments.len(),
      2,
      "5-frame silence dip must close the segment at default \
       min_silence_duration_ms = 100; got {} segments",
      segments.len()
    );
  }

  #[test]
  fn force_split_applies_speech_pad_to_split_boundaries() {
    let config = SpeechOptions::default()
      .with_min_speech_duration(Duration::ZERO)
      .with_speech_pad(Duration::from_millis(32))
      .with_min_silence_duration(Duration::from_millis(10_000))
      .with_min_silence_at_max_speech(Duration::from_millis(64))
      .with_max_speech_duration(Duration::from_millis(512));
    let mut segmenter = SpeechSegmenter::new(config);

    let mut probabilities = vec![0.9; 4];
    probabilities.extend(vec![0.0; 4]);
    probabilities.extend(vec![0.9; 8]);

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(segments[0].end_sample(), 2_560);
    assert_eq!(segments[1].start_sample(), 3_584);
  }

  #[test]
  fn finish_preserves_undrained_queued_segments() {
    // `push_probabilities` can queue multiple segments per call (rare
    // but possible — a long buffer with a force-split + close in one
    // push). A `finish()` that reset the segmenter would clear the
    // queue and silently lose any segments the caller had not popped
    // yet.
    //
    // The contract this pins: `finish()` enqueues the trailing segment
    // (if any) at the back of the queue and pops the front, so undrained
    // segments come out in order before the trailing one.
    let config = SpeechOptions::default();
    let mut segmenter = SpeechSegmenter::new(config);

    // Simulate the post-`push_probabilities` state where two segments
    // closed in one call but the caller only popped the first: stage
    // two segments in the queue directly via the (private) field.
    let queued_a = SpeechSegment::new(0, 1_000, segmenter.sample_rate());
    let queued_b = SpeechSegment::new(2_000, 3_000, segmenter.sample_rate());
    segmenter.pending_segments.push_back(queued_a);
    segmenter.pending_segments.push_back(queued_b);

    // First finish(): must return the head of the queue, NOT silently
    // drop the rest.
    assert_eq!(segmenter.finish(), Some(queued_a));
    assert_eq!(segmenter.pending_segment_count(), 1);

    // Second finish(): must drain the next queued segment.
    assert_eq!(segmenter.finish(), Some(queued_b));
    assert_eq!(segmenter.pending_segment_count(), 0);

    // Queue exhausted, no trailing → None.
    assert_eq!(segmenter.finish(), None);

    // Active state cleared by finish() so a subsequent push could
    // start a fresh segment cleanly.
    assert!(!segmenter.is_active());
  }

  // ── Backend seam: hermetic mock (no ORT) ──────────────────────────

  /// A backend error distinct from `crate::Error`, exercising the
  /// associated-error bridge an out-of-tree backend would use.
  #[derive(Debug)]
  struct MockError(&'static str);

  impl std::fmt::Display for MockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      f.write_str(self.0)
    }
  }

  impl std::error::Error for MockError {}

  impl From<MockError> for crate::Error {
    fn from(error: MockError) -> Self {
      crate::Error::Backend(Box::new(error))
    }
  }

  /// End-of-stream policy for [`MockBackend`]: whether an incomplete
  /// trailing window is zero-padded into a final probability (Silero) or
  /// dropped (FireRed's `snip_edges=true`).
  #[derive(Debug, Clone, Copy, PartialEq, Eq)]
  enum EofPolicy {
    Pad,
    Snip,
  }

  /// A push-based `VadBackend` that returns canned probabilities — one per
  /// completed analysis window — at a caller-declared window/hop geometry.
  /// It authors no detection logic; it exists to drive the segmenter's hop
  /// math, the emit-zero/one/many contract, the end-of-stream policy, and
  /// the error bridge without ORT.
  ///
  /// `window` samples make one analysis window; after emitting a window the
  /// head advances by `hop`. `window > hop` means overlapping windows
  /// (FireRed geometry); `window == hop` means disjoint frames (Silero
  /// geometry). PCM shorter than one window is buffered in `tail` until a
  /// later `push` completes it, or handled by `finish` per [`EofPolicy`].
  struct MockBackend {
    window: usize,
    hop: usize,
    sample_rate: SampleRate,
    eof: EofPolicy,
    tail: Vec<f32>,
    probabilities: Vec<f32>,
    // Index of the next canned probability; also the running count of
    // windows emitted, which the FireRed-geometry proofs assert against.
    cursor: usize,
    fail_at: Option<usize>,
  }

  impl MockBackend {
    fn new(window: usize, hop: usize, eof: EofPolicy, probabilities: Vec<f32>) -> Self {
      assert!(window != 0 && hop != 0 && window >= hop);
      Self {
        window,
        hop,
        sample_rate: SampleRate::Rate16k,
        eof,
        tail: Vec::new(),
        probabilities,
        cursor: 0,
        fail_at: None,
      }
    }

    /// Silero-style geometry: window == hop == `frame`, zero-padding the
    /// trailing partial frame at end-of-stream.
    fn silero_like(frame: usize, probabilities: Vec<f32>) -> Self {
      Self::new(frame, frame, EofPolicy::Pad, probabilities)
    }

    fn with_sample_rate(mut self, sample_rate: SampleRate) -> Self {
      self.sample_rate = sample_rate;
      self
    }

    fn with_fail_at(mut self, cursor: usize) -> Self {
      self.fail_at = Some(cursor);
      self
    }

    /// Emit the next canned probability (`0.0` past the end) via `sink`,
    /// failing if this window index was marked. Advances `cursor`.
    fn emit(&mut self, sink: &mut dyn FnMut(f32)) -> Result<(), MockError> {
      if self.fail_at == Some(self.cursor) {
        return Err(MockError("mock predict failure"));
      }
      let probability = self.probabilities.get(self.cursor).copied().unwrap_or(0.0);
      self.cursor += 1;
      sink(probability);
      Ok(())
    }
  }

  impl VadBackend for MockBackend {
    type Error = MockError;

    fn frame_hop(&self) -> usize {
      self.hop
    }

    fn sample_rate(&self) -> SampleRate {
      self.sample_rate
    }

    fn push(&mut self, samples: &[f32], sink: &mut dyn FnMut(f32)) -> Result<(), MockError> {
      self.tail.extend_from_slice(samples);
      // Emit one probability per complete window, advancing the head by
      // `hop` (not `window`) so overlapping windows are honored.
      while self.tail.len() >= self.window {
        self.emit(sink)?;
        self.tail.drain(..self.hop);
      }
      Ok(())
    }

    fn finish(&mut self, sink: &mut dyn FnMut(f32)) -> Result<(), MockError> {
      match self.eof {
        // Silero policy: zero-pad the trailing partial window into one
        // final probability.
        EofPolicy::Pad if !self.tail.is_empty() => {
          self.tail.resize(self.window, 0.0);
          self.emit(sink)?;
          self.tail.clear();
        }
        // FireRed's snip_edges=true: drop the incomplete trailing window.
        _ => self.tail.clear(),
      }
      Ok(())
    }

    fn reset(&mut self) {
      self.tail.clear();
      self.cursor = 0;
    }
  }

  #[test]
  fn set_frame_hop_overrides_the_sample_rate_default() {
    let mut segmenter = SpeechSegmenter::new(SpeechOptions::default());
    // The 16 kHz model chunk is the default hop.
    assert_eq!(segmenter.frame_hop(), 512);
    segmenter.set_frame_hop(4096);
    assert_eq!(segmenter.frame_hop(), 4096);
  }

  #[test]
  fn detect_speech_with_uses_backend_sample_rate_not_options() {
    // Regression (backend-seam High): the backend is authoritative for
    // its stream's sample rate. An 8 kHz / 256-sample backend feeding
    // 320 ms of speech (ten 0.9 frames over 2_560 samples) must produce
    // one segment — 320 ms clears the 250 ms minimum at 8 kHz (2_000
    // samples) — stamped 8 kHz, even though the passed options carry the
    // default 16 kHz. Before the fix `detect_speech_with` built the
    // segmenter straight from the 16 kHz options: it converted 250 ms to
    // 4_000 samples, dropped the 2_560-sample run, and would have stamped
    // any segment 16 kHz. Mutation: drop the `set_sample_rate` derivation
    // in `detect_speech_with` → zero segments (and a 16 kHz stamp) → red.
    let mut backend =
      MockBackend::silero_like(256, vec![0.9; 10]).with_sample_rate(SampleRate::Rate8k);
    let samples = vec![0.0_f32; 10 * 256];
    let segments =
      detect_speech_with(&mut backend, &samples, SpeechOptions::default()).expect("detect");
    assert_eq!(
      segments.len(),
      1,
      "320 ms of speech at the backend's 8 kHz must yield one segment"
    );
    assert_eq!(
      segments[0].sample_rate(),
      SampleRate::Rate8k,
      "segment must be stamped with the backend's rate"
    );
  }

  #[test]
  fn mock_geometry_closes_after_two_256ms_low_frames() {
    // A backend declaring 4096-sample frames (window == hop) makes each
    // frame 256 ms at 16 kHz. The default `min_silence_duration_ms = 100`
    // is 1600 samples. Because the silence counter is measured BEFORE the
    // current frame is consumed, the FIRST low frame only establishes
    // the silence start (counter 0) and the SECOND low frame sees a full
    // 4096-sample (256 ms) gap — which already exceeds 1600 — so the
    // segment closes on the second low frame. This is the above-the-
    // threshold side of the duration→hop rounding, and every boundary
    // lands on a 4096-sample multiple.
    let mut backend = MockBackend::silero_like(4096, vec![0.9, 0.9, 0.9, 0.0, 0.0]);
    let samples = vec![0.0_f32; 5 * 4096];
    let segments =
      detect_speech_with(&mut backend, &samples, SpeechOptions::default()).expect("detect");

    assert_eq!(
      segments.len(),
      1,
      "two 256 ms low frames must close one segment"
    );
    assert_eq!(segments[0].start_sample(), 0);
    // raw_end = 3 * 4096 (silence start), + 30 ms speech_pad (480).
    assert_eq!(segments[0].end_sample(), 3 * 4096 + 480);
    // Every window handed to the backend was consumed at 4096 samples.
    assert_eq!(backend.cursor, 5);
  }

  #[test]
  fn mock_geometry_holds_open_through_one_256ms_low_frame() {
    // The below-the-threshold side: a SINGLE 256 ms low frame only
    // establishes the silence start (counter 0 < 1600), so no segment
    // closes mid-stream. The open segment is emitted by the end-of-
    // stream `finish`, spanning to the raw current sample — a 4096-
    // sample multiple with no trailing pad. Hardcoding a 512-sample
    // hop here would advance the timeline too slowly to satisfy the
    // 250 ms (4000-sample) minimum-speech gate and drop the segment.
    let mut backend = MockBackend::silero_like(4096, vec![0.9, 0.9, 0.9, 0.0]);
    let samples = vec![0.0_f32; 4 * 4096];
    let segments =
      detect_speech_with(&mut backend, &samples, SpeechOptions::default()).expect("detect");

    assert_eq!(
      segments.len(),
      1,
      "one 256 ms low frame must not close the segment"
    );
    assert_eq!(segments[0].start_sample(), 0);
    assert_eq!(segments[0].end_sample(), 4 * 4096);
  }

  #[test]
  fn mock_geometry_max_speech_lookahead_is_frame_aware() {
    // Regression (backend-seam Medium): the max-speech force-split
    // lookahead must subtract the ACTIVE hop, not the sample rate's
    // model chunk. A 4096-sample backend at 16 kHz with a 1 s max-speech
    // ceiling (speech_pad 0) has a hop-aware threshold of
    // 16_000 − 4_096 = 11_904, so the first frame_start past it is 12_288
    // (768 ms) and the split lands there. The old chunk-based threshold
    // (16_000 − 512 = 15_488) split one frame later, at frame_start
    // 16_384 (1.024 s) — overshooting the configured 1 s maximum. Mutation:
    // revert the lookahead to `chunk_samples()` → the split moves to
    // 16_384 → red.
    let mut backend = MockBackend::silero_like(4096, vec![0.9; 6]);
    let samples = vec![0.0_f32; 6 * 4096];
    let options = SpeechOptions::default()
      .with_min_speech_duration(Duration::ZERO)
      .with_speech_pad(Duration::ZERO)
      .with_max_speech_duration(Duration::from_millis(1_000));
    let segments = detect_speech_with(&mut backend, &samples, options).expect("detect");

    assert_eq!(segments[0].start_sample(), 0);
    assert_eq!(
      segments[0].end_sample(),
      12_288,
      "max-speech split must land at the hop-aware 12_288, not the \
       chunk-based overshoot 16_384"
    );
  }

  #[test]
  fn mock_backend_error_bridges_through_backend_variant() {
    // The associated `VadBackend::Error` (a foreign type here) must
    // reach the caller through the transparent `Error::Backend` variant,
    // delegating its `Display` to the wrapped error. The failure fires on
    // the SECOND emitted window (`fail_at: 1`) — mid-`push` — so the
    // error propagates out of `push`, not `finish`.
    let mut backend = MockBackend::silero_like(4096, vec![0.9, 0.9, 0.9]).with_fail_at(1);
    let samples = vec![0.0_f32; 3 * 4096];
    let error = detect_speech_with(&mut backend, &samples, SpeechOptions::default())
      .expect_err("backend failure must propagate");
    assert!(
      matches!(error, crate::Error::Backend(_)),
      "backend error must bridge through Error::Backend, got {error:?}"
    );
    assert_eq!(error.to_string(), "mock predict failure");
  }

  // ── FireRed geometry proof: the push-based contract fits a 400-sample
  //    window / 160-sample hop / delayed-first-output / snip_edges backend
  //    (the fence's exact 640- and 500-sample histories). ────────────────

  #[test]
  fn firered_geometry_640_samples_emit_two_windows_through_detect_speech_with() {
    // The push-based contract must express FireRed's feature geometry: a
    // 400-sample analysis window advanced by a 160-sample hop, with no
    // output until a full window exists and snip_edges=true at the tail.
    // The fence's exact history — 640 PCM samples contain two valid
    // windows (starting at samples 0 and 160) — so driving
    // `detect_speech_with` must run the backend to EXACTLY two
    // probabilities. The retired `predict` / `frame_samples` contract
    // could not express this: one input chunk meant one hop and a
    // mandatory trailing pad, so 640 samples forced either four
    // probabilities at hop 160 (two before any valid window existed) or a
    // missed window at frame 400.
    let mut backend = MockBackend::new(400, 160, EofPolicy::Snip, vec![0.9, 0.9]);
    let samples = vec![0.0_f32; 640];
    let _ = detect_speech_with(&mut backend, &samples, SpeechOptions::default()).expect("detect");
    assert_eq!(
      backend.cursor, 2,
      "640 samples must emit exactly two FireRed windows (at samples 0 and 160)"
    );
  }

  #[test]
  fn firered_geometry_500_sample_tail_is_snipped_through_detect_speech_with() {
    // The snip_edges half of the fence history: 500 samples contain ONE
    // full window (at sample 0); the trailing 340 samples are an
    // incomplete window that FireRed's snip_edges=true DROPS. Driving
    // `detect_speech_with` must run the backend to exactly ONE probability
    // — the old detector-side zero-pad would have fabricated a second
    // FireRed window that upstream never produces.
    let mut backend = MockBackend::new(400, 160, EofPolicy::Snip, vec![0.9, 0.9]);
    let samples = vec![0.0_f32; 500];
    let _ = detect_speech_with(&mut backend, &samples, SpeechOptions::default()).expect("detect");
    assert_eq!(
      backend.cursor, 1,
      "500 samples must emit exactly one window; the 340-sample tail is snipped, not padded"
    );
  }
}
