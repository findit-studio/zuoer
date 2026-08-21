use std::collections::VecDeque;

use crate::options::{RunOptions, SampleRate};

/// One contiguous above-threshold run on the stream timeline.
///
/// Domain-neutral: a run is whatever the frame probabilities were scoring
/// — speech for a VAD backend, one sound-event class for a detector like
/// CED. [`SpeechSegment`](crate::SpeechSegment) is an alias of this type.
///
/// # Probability aggregates
///
/// [`mean_probability`](Self::mean_probability) and
/// [`peak_probability`](Self::peak_probability) summarize the frame
/// probabilities the run was built from — the data source for VAD segment
/// confidence and CED event confidence. Both are accumulated in O(1) per
/// frame over the run's **raw model-frame span**:
///
/// - **Padding is excluded.** [`RunOptions::pad`] extends the emitted
///   `[start_sample, end_sample)` on both sides, but padding is a timeline
///   courtesy, not an observation: there are no probabilities out there to
///   average, so the accumulator covers only `active_raw_start ..
///   raw_end`.
/// - **Bridged frames are included.** Frames inside a gap shorter than
///   [`RunOptions::min_gap_duration`] genuinely occurred inside the run,
///   so they stay in the mean and pull it down. That is the correct
///   signal.
/// - **A force-split cuts the accumulator too.** When
///   [`RunOptions::max_run_duration`] force-splits, the emitted run
///   carries only what was accumulated up to the split point, and the
///   continuation run's accumulator restarts from the frame the
///   continuation starts at. Frames in the gap that the split landed on
///   belong to neither run and appear in neither aggregate.
///
/// Both aggregates on a run the [`RunSegmenter`] emits are always finite
/// and inside `[0, 1]`, whatever the probability producer fed in:
/// [`push_probability`](RunSegmenter::push_probability) canonicalizes
/// every frame — `NaN` to `0.0`, everything else clamped — before it
/// reaches either the state machine or the accumulator.
///
/// A [`Run`] built directly with [`new`](Self::new) has zeroed aggregates;
/// only runs the [`RunSegmenter`] emits carry observed values. The
/// [`with_mean_probability`](Self::with_mean_probability) /
/// [`with_peak_probability`](Self::with_peak_probability) setters store
/// what the caller hands them, so the range guarantee above covers
/// segmenter-emitted runs only.
///
/// `Eq` is deliberately not implemented — the aggregates are floats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Run {
  start_sample: u64,
  end_sample: u64,
  sample_rate: SampleRate,
  mean_probability: f32,
  peak_probability: f32,
}

impl Run {
  /// Create a new run with the given start and end samples and sample
  /// rate, and zeroed probability aggregates.
  ///
  /// Attach observed aggregates with
  /// [`with_mean_probability`](Self::with_mean_probability) /
  /// [`with_peak_probability`](Self::with_peak_probability).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(start_sample: u64, end_sample: u64, sample_rate: SampleRate) -> Self {
    Self {
      start_sample,
      end_sample,
      sample_rate,
      mean_probability: 0.0,
      peak_probability: 0.0,
    }
  }

  /// Set the mean frame probability over this run.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_mean_probability(mut self, mean_probability: f32) -> Self {
    self.mean_probability = mean_probability;
    self
  }

  /// Set the peak frame probability over this run.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_peak_probability(mut self, peak_probability: f32) -> Self {
    self.peak_probability = peak_probability;
    self
  }

  /// Returns the start sample of this run.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn start_sample(&self) -> u64 {
    self.start_sample
  }

  /// Returns the end sample of this run.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn end_sample(&self) -> u64 {
    self.end_sample
  }

  /// Returns the sample rate of this run.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sample_rate(&self) -> SampleRate {
    self.sample_rate
  }

  /// Returns the number of samples in this run.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sample_count(&self) -> u64 {
    self.end_sample.saturating_sub(self.start_sample)
  }

  /// Returns the start time of this run in seconds.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn start_seconds(&self) -> f64 {
    self.start_sample as f64 / self.sample_rate.hz() as f64
  }

  /// Returns the end time of this run in seconds.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn end_seconds(&self) -> f64 {
    self.end_sample as f64 / self.sample_rate.hz() as f64
  }

  /// Returns the mean frame probability over this run's raw model-frame
  /// span — padding excluded, bridged frames included.
  ///
  /// See the [type-level notes](Self#probability-aggregates) for the exact
  /// span the accumulator covers. `0.0` for a run built with
  /// [`new`](Self::new).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn mean_probability(&self) -> f32 {
    self.mean_probability
  }

  /// Returns the highest frame probability over this run's raw
  /// model-frame span — padding excluded, bridged frames included.
  ///
  /// See the [type-level notes](Self#probability-aggregates) for the exact
  /// span the accumulator covers. `0.0` for a run built with
  /// [`new`](Self::new).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn peak_probability(&self) -> f32 {
    self.peak_probability
  }
}

/// O(1) running mean/peak over a set of frame probabilities.
///
/// The sum is kept in `f64` so a long run does not lose the tail of its
/// mean to `f32` rounding. Every observation arrives already canonicalized
/// into `[0, 1]` by [`normalize_probability`] — the accumulator is never
/// handed a `NaN`, an infinity, or an out-of-range value — so `sum`,
/// [`mean`](Self::mean), and [`peak`](Self::peak) are always finite and in
/// range. `peak` still starts at `f32::NEG_INFINITY` so the first
/// observation wins outright instead of tying with a `0.0` seed the run
/// never observed.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Aggregate {
  sum: f64,
  count: u64,
  peak: f32,
}

impl Aggregate {
  const EMPTY: Self = Self {
    sum: 0.0,
    count: 0,
    peak: f32::NEG_INFINITY,
  };

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn observe(&mut self, probability: f32) {
    self.sum += probability as f64;
    self.count += 1;
    if probability > self.peak {
      self.peak = probability;
    }
  }

  /// Fold `other`'s frames into this accumulator. Used when a gap shorter
  /// than `min_gap_duration` is bridged and its frames become part of the
  /// run.
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn absorb(&mut self, other: &Self) {
    self.sum += other.sum;
    self.count += other.count;
    if other.peak > self.peak {
      self.peak = other.peak;
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn mean(&self) -> f32 {
    if self.count == 0 {
      0.0
    } else {
      (self.sum / self.count as f64) as f32
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn peak(&self) -> f32 {
    if self.count == 0 { 0.0 } else { self.peak }
  }
}

/// Canonicalize one raw frame probability into `[0, 1]`.
///
/// `NaN` is mapped to `0.0` rather than clamped: `f32::clamp` returns
/// `NaN` for a `NaN` input, so a bare clamp would let it straight through.
/// See [`RunSegmenter::push_probability`] for the policy this implements
/// and why `0.0` is the mapping that preserves the pre-aggregate
/// behaviour.
#[cfg_attr(not(tarpaulin), inline(always))]
const fn normalize_probability(probability: f32) -> f32 {
  if probability.is_nan() {
    0.0
  } else {
    probability.clamp(0.0, 1.0)
  }
}

/// A recorded preferred force-split boundary, with the aggregate of the
/// frames that precede it.
#[derive(Debug, Clone, Copy)]
struct SplitPoint {
  /// Raw end sample the force-split would close the run at.
  end: u64,
  /// Aggregate over `active_raw_start .. end`.
  aggregate: Aggregate,
}

/// The resume point a force-split would restart the next run at, with the
/// aggregate of the frames observed since it.
#[derive(Debug, Clone, Copy)]
struct PendingRun {
  /// Raw start sample of the continuation run.
  start: u64,
  /// Aggregate over `start .. current frame`.
  aggregate: Aggregate,
}

/// Streaming hysteresis state machine that turns a frame-probability
/// sequence into contiguous [`Run`]s.
///
/// Domain-neutral: it consumes probabilities and knows nothing about what
/// produced them. [`SpeechSegmenter`](crate::SpeechSegmenter) is an alias
/// of this type; a sound-event detector drives the same machine per class.
///
/// # Timeline contract
///
/// The segmenter has no clock of its own. It counts frames and multiplies:
/// each [`push_probability`](Self::push_probability) advances the timeline
/// by exactly [`frame_hop`](Self::frame_hop) samples, so the `n`-th
/// probability covers `[n * frame_hop, (n + 1) * frame_hop)` and every
/// sample boundary a [`Run`] carries is a multiple of the hop (before
/// padding). Duration options are converted against
/// [`sample_rate`](Self::sample_rate) once, so a threshold that is not a
/// whole number of hops is compared at sample resolution, not rounded to
/// frames.
///
/// `frame_hop` is the **hop**, not the analysis window: for a producer
/// whose windows overlap (a 400-sample window advanced by 160 samples) it
/// is 160. It defaults to the sample rate's model chunk size (512 samples
/// at 16 kHz — the Silero geometry); a producer with a different hop
/// declares it through [`set_frame_hop`](Self::set_frame_hop) (done
/// automatically by [`detect_speech_with`](crate::detect_speech_with) from
/// [`VadBackend::frame_hop`](crate::VadBackend::frame_hop)) so the same
/// segmentation rules apply at any frame geometry.
///
/// # Emitted runs
///
/// A run opens on the first frame at or above
/// [`start_threshold`](RunOptions::start_threshold) and stays open while
/// probabilities hold at or above
/// [`end_threshold`](RunOptions::end_threshold). A below-threshold gap
/// shorter than [`min_gap_duration`](RunOptions::min_gap_duration) is
/// bridged; a longer one closes the run at the gap's first frame. A closed
/// run is dropped unless its raw span reaches
/// [`min_run_duration`](RunOptions::min_run_duration), and the surviving
/// run is padded by [`pad`](RunOptions::pad) on both sides. Each emitted
/// run carries the mean and peak of the probabilities over its raw span;
/// see [`Run`](Run#probability-aggregates).
#[derive(Debug, Clone)]
pub struct RunSegmenter {
  options: RunOptions,
  // Frame hop: the timeline advances by this many samples per
  // probability. Defaults to `sample_rate().chunk_samples()`; a backend
  // with a different hop overrides it via `set_frame_hop`.
  frame_hop: u64,
  current_sample: u64,
  // Padded start sample used for emitted runs.
  active_start: Option<u64>,
  // Raw model-frame start sample used for upstream-compatible duration checks.
  active_raw_start: Option<u64>,
  tentative_end: Option<u64>,
  // Aggregate over the run's confirmed frames: `active_raw_start ..
  // tentative_end` when a gap is open, `active_raw_start .. now`
  // otherwise.
  aggregate: Aggregate,
  // Aggregate over the frames of the currently open gap (`tentative_end
  // .. now`). Folded into `aggregate` when the gap is bridged; discarded
  // when the gap closes the run, because those frames sit past `raw_end`.
  tentative_aggregate: Aggregate,
  // Start sample of the most recent gap long enough to be a preferred
  // force-split point, with the aggregate of the frames before it.
  max_split: Option<SplitPoint>,
  // First above-threshold frame after `max_split`; used to resume after a
  // force-split at that gap boundary, with the aggregate accumulated
  // since.
  next_run: Option<PendingRun>,
  // Queue of runs closed by recent push_probabilities calls that have not
  // yet been popped by the caller. Drained one run at a time via
  // `pop_pending`.
  //
  // `pub(crate)` so the shell's inline tests can stage a queue directly.
  pub(crate) pending_segments: VecDeque<Run>,
}

impl RunSegmenter {
  /// Create a new `RunSegmenter` with the given options.
  ///
  /// The frame hop defaults to the options' sample-rate model chunk
  /// size (512 samples at 16 kHz). Override it with
  /// [`set_frame_hop`](Self::set_frame_hop) for a producer that declares a
  /// different hop.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(options: RunOptions) -> Self {
    let frame_hop = options.sample_rate().chunk_samples() as u64;
    Self {
      options,
      frame_hop,
      current_sample: 0,
      active_start: None,
      active_raw_start: None,
      tentative_end: None,
      aggregate: Aggregate::EMPTY,
      tentative_aggregate: Aggregate::EMPTY,
      max_split: None,
      next_run: None,
      pending_segments: VecDeque::new(),
    }
  }

  /// Returns a reference to the [`RunOptions`] used by this segmenter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn options(&self) -> &RunOptions {
    &self.options
  }

  /// Reconfigure the segmenter for a stream with a different sample rate.
  ///
  /// Changing sample rate starts a new logical timeline, so any
  /// in-flight run state is cleared.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_sample_rate(&mut self, sample_rate: SampleRate) {
    if self.sample_rate() != sample_rate {
      self.options.set_sample_rate(sample_rate);
      // A new rate implies a new default hop; a custom backend hop must
      // be re-applied via `set_frame_hop`.
      self.frame_hop = sample_rate.chunk_samples() as u64;
      self.reset();
    }
  }

  /// Returns the sample rate used by this segmenter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sample_rate(&self) -> SampleRate {
    self.options.sample_rate()
  }

  /// Returns the frame hop: the number of samples this segmenter's
  /// timeline advances by per probability.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn frame_hop(&self) -> usize {
    self.frame_hop as usize
  }

  /// Set the frame hop.
  ///
  /// The segmenter advances its sample timeline by `frame_hop` for each
  /// probability fed to [`push_probability`](Self::push_probability), so
  /// this must equal the hop of the producer of those probabilities — the
  /// value a [`VadBackend`](crate::VadBackend) reports from
  /// [`frame_hop`](crate::VadBackend::frame_hop).
  /// [`detect_speech_with`](crate::detect_speech_with) applies it
  /// automatically; call it directly only when driving
  /// [`push_probability`](Self::push_probability) with a producer whose
  /// hop differs from the sample rate's model chunk size.
  ///
  /// [`set_sample_rate`](Self::set_sample_rate) resets the hop to the
  /// rate's model chunk size, so apply this after any rate change.
  ///
  /// # Panics
  ///
  /// Panics if `frame_hop` is zero.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_frame_hop(&mut self, frame_hop: usize) {
    assert!(frame_hop != 0, "frame_hop must be non-zero");
    self.frame_hop = frame_hop as u64;
  }

  /// Returns whether the segmenter is currently active (i.e., has an ongoing run).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_active(&self) -> bool {
    self.active_start.is_some()
  }

  /// Reset the segmenter's internal state: the in-flight run tracker
  /// (active start, tentative end, probability aggregates, force-split
  /// bookkeeping), the running sample counter, and any runs queued for
  /// [`pop_pending`](Self::pop_pending) drain.
  ///
  /// This resets only the segmenter. A backend driving it keeps its own
  /// per-stream memory (recurrent state, rolling context, un-chunked PCM
  /// tail); reset that separately via
  /// [`VadBackend::reset`](crate::VadBackend::reset) when reusing the pair
  /// for a new logical recording.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn reset(&mut self) {
    self.current_sample = 0;
    self.clear_run_memory();
    self.pending_segments.clear();
  }

  /// Number of runs currently queued for drain via
  /// [`pop_pending`](Self::pop_pending).
  ///
  /// Always `0` after a [`pop_pending`](Self::pop_pending) or
  /// [`finish`](Self::finish) call that returned `None`. Useful for tests
  /// that want to assert the caller has drained everything before tearing
  /// down a stream.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn pending_segment_count(&self) -> usize {
    self.pending_segments.len()
  }

  /// Consume one probability for one model frame.
  ///
  /// Returns `Some(run)` only when a run can be closed with the currently
  /// available evidence.
  ///
  /// This returns any closed run directly and does **not** touch the
  /// pending queue. To feed a run of probabilities and buffer the closed
  /// runs for ordered draining — the shape a streaming backend driver
  /// needs — use [`push_probabilities`](Self::push_probabilities)
  /// followed by [`pop_pending`](Self::pop_pending).
  ///
  /// # Malformed probabilities
  ///
  /// [`VadBackend`](crate::VadBackend) documents `[0, 1]`; this is the one
  /// place that contract is enforced rather than assumed. The value is
  /// canonicalized before anything else reads it, and that single
  /// canonical value drives **both** the hysteresis comparisons and the
  /// run's aggregates, so the state machine and the aggregate can never
  /// disagree about a frame:
  ///
  /// - `NaN` becomes `0.0` — silence. Every comparison against `NaN` is
  ///   false, so a `NaN` frame already behaved as below-threshold; `0.0`
  ///   keeps that segmentation instead of letting the frame escape into
  ///   [`mean_probability`](Run::mean_probability) as a `NaN` a consumer
  ///   would read. The sole configuration where the two differ is a
  ///   threshold of exactly `0.0`, which `0.0` satisfies (`0.0 >= 0.0`)
  ///   and `NaN` did not: there, a `NaN` frame now opens or sustains a
  ///   run.
  /// - everything else is clamped into range: `f32::INFINITY` and any
  ///   value above `1.0` become `1.0`; `f32::NEG_INFINITY` and any value
  ///   below `0.0` become `0.0`. Each of those already compared to the
  ///   thresholds the way its clamped form does, so clamping moves no
  ///   boundary at all.
  ///
  /// The repair keeps a malformed frame from corrupting a value a
  /// consumer reads; it is not a licence to emit out-of-range
  /// probabilities.
  pub fn push_probability(&mut self, probability: f32) -> Option<Run> {
    let probability = normalize_probability(probability);
    let frame_hop = self.frame_hop;
    let frame_start = self.current_sample;
    self.current_sample = self.current_sample.saturating_add(frame_hop);

    if probability >= self.options.start_threshold() {
      if let Some(tentative_end) = self.tentative_end.take() {
        let gap_samples = frame_start.saturating_sub(tentative_end);
        if gap_samples > self.options.min_gap_at_max_run_samples() {
          // Snapshot BEFORE the bridge fold: the split point closes at
          // `tentative_end`, so it carries only the frames before the gap.
          self.max_split = Some(SplitPoint {
            end: tentative_end,
            aggregate: self.aggregate,
          });
          self.next_run = Some(PendingRun {
            start: frame_start,
            aggregate: Aggregate::EMPTY,
          });
        }
        // The gap is bridged: its frames occurred inside the run, so they
        // stay in the run's aggregate.
        self.aggregate.absorb(&self.tentative_aggregate);
        self.tentative_aggregate = Aggregate::EMPTY;
      }
      if self.active_start.is_none() {
        self.active_start = Some(frame_start.saturating_sub(self.options.pad_samples()));
        self.active_raw_start = Some(frame_start);
        self.aggregate = Aggregate::EMPTY;
        self.aggregate.observe(probability);
        return None;
      }
    }

    let start = self.active_start?;
    let raw_start = self.active_raw_start?;
    if let Some(max_run_samples) = self.options.max_run_samples_for_frame(self.frame_hop)
      && frame_start.saturating_sub(raw_start) > max_run_samples
    {
      return self.split_at_max_duration(frame_start, probability);
    }

    if probability >= self.options.end_threshold() {
      self.observe(probability);
      return None;
    }

    // Gap counter is evaluated against `frame_start` (the start sample of
    // the current frame), not `current_sample` (which is already the *end*
    // of the current frame). This matches upstream Python `silero-vad`'s
    // `sil_dur_now = cur_sample - temp_end` semantics, where `cur_sample`
    // is read BEFORE the model consumes the current window. Without this,
    // the comparator fires one frame early — a 4-frame (128 ms) silence
    // dip would close a segment at default `min_silence_duration_ms =
    // 100`, where Python tolerates it and closes after 5 consecutive
    // low-probability frames. See the parity harness in `tests/parity/`
    // and the v0.3.0 CHANGELOG entry.
    //
    // `get_or_insert` runs BEFORE `observe` so the frame at `gap_start`
    // lands in the tentative aggregate: `gap_start` is the run's `raw_end`
    // if this gap closes it, and `raw_end` is exclusive.
    let gap_start = *self.tentative_end.get_or_insert(frame_start);
    self.observe(probability);
    let gap_samples = frame_start.saturating_sub(gap_start);
    if gap_samples > self.options.min_gap_at_max_run_samples() {
      self.max_split = Some(SplitPoint {
        end: gap_start,
        aggregate: self.aggregate,
      });
    }
    if gap_samples < self.options.min_gap_samples() {
      return None;
    }

    let aggregate = self.aggregate;
    self.clear_run_memory();
    self.build_run(start, raw_start, gap_start, &aggregate)
  }

  /// Feed a sequence of frame probabilities and buffer any closed runs for
  /// ordered retrieval.
  ///
  /// Each probability is fed through
  /// [`push_probability`](Self::push_probability) in turn; every run it
  /// closes is appended to the internal pending queue rather than
  /// returned. Drain the queue one run at a time with
  /// [`pop_pending`](Self::pop_pending) (or, at end-of-stream,
  /// [`finish`](Self::finish)).
  ///
  /// This is the sans-I/O seam a streaming backend driver builds on: run
  /// the backend over the incoming PCM to obtain the frame probabilities,
  /// hand them here, then pop the closed runs. Passing an empty slice
  /// buffers nothing — a pure drain point when paired with
  /// [`pop_pending`](Self::pop_pending).
  pub fn push_probabilities(&mut self, probabilities: &[f32]) {
    for &probability in probabilities {
      if let Some(run) = self.push_probability(probability) {
        self.pending_segments.push_back(run);
      }
    }
  }

  /// Pop the next buffered run closed by
  /// [`push_probabilities`](Self::push_probabilities), in order, or
  /// `None` when the queue is empty.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn pop_pending(&mut self) -> Option<Run> {
    self.pending_segments.pop_front()
  }

  /// Compute the trailing open run (if any) without resetting.
  /// Helper for `finish`.
  fn take_trailing(&self) -> Option<Run> {
    let start = self.active_start?;
    let raw_start = self.active_raw_start?;
    let end = self.current_sample;
    if end.saturating_sub(raw_start) < self.options.min_run_samples() {
      None
    } else {
      // The trailing run spans every frame pushed since `raw_start`,
      // including an unclosed gap, so its aggregate includes them too.
      let mut aggregate = self.aggregate;
      aggregate.absorb(&self.tentative_aggregate);
      Some(Self::finished_run(
        start,
        end,
        self.sample_rate(),
        &aggregate,
      ))
    }
  }

  /// Finish the current stream and return the next available run.
  ///
  /// Enqueues the trailing open run (if any) onto the pending queue, then
  /// pops and returns the head of that queue. This preserves the order of
  /// any runs that an earlier
  /// [`push_probabilities`](Self::push_probabilities) queued but the
  /// caller hasn't drained yet (the rare force-split case): they come out
  /// before the trailing run.
  ///
  /// The in-flight run tracker is cleared so
  /// [`is_active`](Self::is_active) is `false` afterwards and a follow-up
  /// `finish()` can't re-emit the same trailing run. The pending queue is
  /// left intact so subsequent [`pop_pending`](Self::pop_pending) calls
  /// drain the rest; call [`Self::reset`] explicitly when starting a new
  /// stream.
  ///
  /// This closes the segmenter's own trailing run only. A backend driver
  /// that also needs to flush the backend's un-chunked PCM tail should
  /// first feed the flushed frame's probability through
  /// [`push_probabilities`](Self::push_probabilities), then call this.
  pub fn finish(&mut self) -> Option<Run> {
    if let Some(trailing) = self.take_trailing() {
      self.pending_segments.push_back(trailing);
    }
    self.clear_run_memory();
    self.pending_segments.pop_front()
  }

  fn split_at_max_duration(&mut self, frame_start: u64, probability: f32) -> Option<Run> {
    let start = self.active_start?;
    let raw_start = self.active_raw_start?;
    let (raw_end, aggregate) = match self.max_split {
      // Split back at the recorded gap: the emitted run carries only the
      // frames before that gap.
      Some(split) => (split.end, split.aggregate),
      // No recorded gap: the run closes at the current frame, so every
      // frame pushed so far belongs to it — including an in-flight gap
      // too short to have been recorded as a split point.
      None => {
        let mut aggregate = self.aggregate;
        aggregate.absorb(&self.tentative_aggregate);
        (frame_start, aggregate)
      }
    };
    let run = self.build_run(start, raw_start, raw_end, &aggregate);

    let next_raw_start = if let Some(next) = self.next_run.filter(|next| next.start >= raw_end) {
      self.active_start = Some(next.start.saturating_sub(self.options.pad_samples()));
      // The continuation restarts from the resume point, carrying the
      // frames observed since it. The frames inside the split gap
      // (`raw_end .. next.start`) belong to neither run.
      self.aggregate = next.aggregate;
      Some(next.start)
    } else if self.max_split.is_none() && probability >= self.options.start_threshold() {
      self.active_start = Some(frame_start.saturating_sub(self.options.pad_samples()));
      self.aggregate = Aggregate::EMPTY;
      Some(frame_start)
    } else {
      self.active_start = None;
      self.aggregate = Aggregate::EMPTY;
      None
    };
    self.active_raw_start = next_raw_start;
    self.clear_split_tracking();
    if next_raw_start.is_some() {
      // The current frame was not part of the emitted run (`raw_end <=
      // frame_start`), so it is the continuation's to observe.
      self.observe(probability);
    }

    run
  }

  /// Route one frame's probability into the accumulator that owns it: the
  /// tentative one while a gap is open, the run's own otherwise. A pending
  /// force-split resume point accumulates in parallel.
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn observe(&mut self, probability: f32) {
    if self.tentative_end.is_some() {
      self.tentative_aggregate.observe(probability);
    } else {
      self.aggregate.observe(probability);
    }
    if let Some(next) = self.next_run.as_mut() {
      next.aggregate.observe(probability);
    }
  }

  fn build_run(
    &self,
    start: u64,
    raw_start: u64,
    raw_end: u64,
    aggregate: &Aggregate,
  ) -> Option<Run> {
    let end_sample = raw_end
      .saturating_add(self.options.pad_samples())
      .min(self.current_sample);
    if raw_end.saturating_sub(raw_start) < self.options.min_run_samples() {
      None
    } else {
      Some(Self::finished_run(
        start,
        end_sample,
        self.sample_rate(),
        aggregate,
      ))
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn finished_run(
    start_sample: u64,
    end_sample: u64,
    sample_rate: SampleRate,
    aggregate: &Aggregate,
  ) -> Run {
    Run::new(start_sample, end_sample, sample_rate)
      .with_mean_probability(aggregate.mean())
      .with_peak_probability(aggregate.peak())
  }

  fn clear_run_memory(&mut self) {
    self.active_start = None;
    self.active_raw_start = None;
    self.aggregate = Aggregate::EMPTY;
    self.clear_split_tracking();
  }

  fn clear_split_tracking(&mut self) {
    self.tentative_end = None;
    self.tentative_aggregate = Aggregate::EMPTY;
    self.max_split = None;
    self.next_run = None;
  }
}

#[cfg(test)]
mod tests {
  use std::time::Duration;

  use crate::{RunOptions, SampleRate};

  use super::{Run, RunSegmenter};

  /// The 16 kHz default hop: one frame is 512 samples / 32 ms.
  const HOP: u64 = 512;

  fn collect(segmenter: &mut RunSegmenter, probabilities: &[f32]) -> Vec<Run> {
    let mut runs = Vec::new();
    for probability in probabilities {
      if let Some(run) = segmenter.push_probability(*probability) {
        runs.push(run);
      }
    }
    if let Some(run) = segmenter.finish() {
      runs.push(run);
    }
    runs
  }

  #[track_caller]
  fn assert_close(actual: f32, expected: f32, what: &str) {
    assert!(
      (actual - expected).abs() < 1e-6,
      "{what}: expected {expected}, got {actual}"
    );
  }

  /// Aggregation baseline: a run of exactly one frame reports that frame's
  /// probability as both mean and peak — no off-by-one in the accumulator
  /// and no phantom frame from the gap frame that closed it.
  #[test]
  fn single_frame_run_reports_that_frame_as_mean_and_peak() {
    let options = RunOptions::default()
      .with_min_run_duration(Duration::ZERO)
      .with_pad(Duration::ZERO)
      .with_min_gap_duration(Duration::ZERO);
    let mut segmenter = RunSegmenter::new(options);

    // One above-threshold frame, then one below-threshold frame which (at
    // `min_gap_duration = 0`) closes the run immediately.
    let runs = collect(&mut segmenter, &[0.8, 0.0]);

    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].start_sample(), 0);
    assert_eq!(runs[0].end_sample(), HOP, "one frame wide");
    assert_close(runs[0].mean_probability(), 0.8, "mean");
    assert_close(runs[0].peak_probability(), 0.8, "peak");
  }

  /// Rule 2: frames bridged by `min_gap_duration` are inside the run, so
  /// they stay in the mean and pull it down. Rule 1's other half: the
  /// frames of the gap that finally CLOSES the run sit past `raw_end` and
  /// must not.
  ///
  /// Eight frames span the run — `0.6, 1.0`, four bridged zeros, `0.6,
  /// 0.6` — so the mean is `2.8 / 8 = 0.35` while the peak stays `1.0`.
  ///
  /// Mutations: dropping the bridged frames gives `2.8 / 4 = 0.7`;
  /// including the five closing gap frames gives `2.8 / 13 ≈ 0.215`;
  /// pinning `peak` to the first observation gives `0.6`. All red.
  #[test]
  fn bridged_gap_frames_stay_in_the_mean() {
    // Defaults: start 0.5 / end 0.35, `min_gap_duration = 100 ms` (1600
    // samples), so a four-frame dip (max gap 1536) is bridged and a
    // five-frame dip (2048) closes.
    let options = RunOptions::default()
      .with_min_run_duration(Duration::ZERO)
      .with_pad(Duration::ZERO);
    let mut segmenter = RunSegmenter::new(options);

    let mut probabilities = vec![0.6, 1.0];
    probabilities.extend([0.0; 4]);
    probabilities.extend([0.6, 0.6]);
    probabilities.extend([0.0; 5]);

    let runs = collect(&mut segmenter, &probabilities);

    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].start_sample(), 0);
    assert_eq!(
      runs[0].end_sample(),
      8 * HOP,
      "the run spans the eight frames the aggregate covers"
    );
    assert_close(runs[0].mean_probability(), 0.35, "mean");
    assert_close(runs[0].peak_probability(), 1.0, "peak");
  }

  /// Rule 1: `pad` extends the emitted timeline but contributes no
  /// observations. Here the run is padded by one whole frame on each side,
  /// so it spans four frames of timeline over two frames of evidence — and
  /// the two frames the padding covers are both `0.0`.
  ///
  /// Mutation: folding the closing gap frame (the one the trailing pad
  /// covers) into the aggregate gives `2 / 3 ≈ 0.667`. Red. The leading
  /// pad has no mutation to make: no accumulator exists before the run
  /// opens, so those frames are unreachable by construction.
  #[test]
  fn padding_is_outside_the_aggregated_span() {
    let options = RunOptions::default()
      .with_min_run_duration(Duration::ZERO)
      .with_min_gap_duration(Duration::ZERO)
      .with_pad(Duration::from_millis(32));
    let mut segmenter = RunSegmenter::new(options);

    let runs = collect(&mut segmenter, &[0.0, 1.0, 1.0, 0.0]);

    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].start_sample(), 0, "raw start 512 padded back to 0");
    assert_eq!(runs[0].end_sample(), 4 * HOP, "raw end 1536 padded to 2048");
    assert_eq!(runs[0].sample_count(), 4 * HOP);
    assert_close(runs[0].mean_probability(), 1.0, "mean");
    assert_close(runs[0].peak_probability(), 1.0, "peak");
  }

  /// A run closed by `finish` spans every frame pushed since it opened,
  /// including an unclosed sub-threshold gap — so those frames ARE in its
  /// aggregate. The span and the accumulator must agree: three frames of
  /// timeline, three observations, mean `2 / 3`.
  ///
  /// Mutation: discarding the tentative aggregate in `take_trailing`
  /// gives `1.0`. Red.
  #[test]
  fn trailing_run_aggregates_its_unclosed_gap() {
    let options = RunOptions::default()
      .with_min_run_duration(Duration::ZERO)
      .with_pad(Duration::ZERO);
    let mut segmenter = RunSegmenter::new(options);

    let runs = collect(&mut segmenter, &[1.0, 1.0, 0.0]);

    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].end_sample(), 3 * HOP);
    assert_close(runs[0].mean_probability(), 2.0 / 3.0, "mean");
    assert_close(runs[0].peak_probability(), 1.0, "peak");
  }

  /// Rule 3, no recorded gap: the force-split closes at the current frame,
  /// so the emitted run carries the five frames before it and the
  /// continuation's accumulator restarts at the split.
  ///
  /// Boundaries are the ones `force_splits_long_speech_when_max_duration_is_reached`
  /// already pins (2560 / 4096); this adds the aggregates.
  ///
  /// Mutation: not restarting the accumulator gives the continuation
  /// `(5·0.6 + 3·1.0) / 8 = 0.75` instead of `1.0`. Red.
  #[test]
  fn force_split_at_the_current_frame_restarts_the_aggregate() {
    let options = RunOptions::default()
      .with_min_run_duration(Duration::ZERO)
      .with_pad(Duration::ZERO)
      .with_max_run_duration(Duration::from_millis(160));
    let mut segmenter = RunSegmenter::new(options);

    let mut probabilities = vec![0.6; 5];
    probabilities.extend([1.0; 3]);

    let runs = collect(&mut segmenter, &probabilities);

    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0].end_sample(), 5 * HOP);
    assert_close(runs[0].mean_probability(), 0.6, "split run mean");
    assert_close(runs[0].peak_probability(), 0.6, "split run peak");

    assert_eq!(runs[1].start_sample(), 5 * HOP);
    assert_close(runs[1].mean_probability(), 1.0, "continuation mean");
    assert_close(runs[1].peak_probability(), 1.0, "continuation peak");
  }

  /// Rule 3, recorded gap: the force-split lands back at a gap the
  /// segmenter recorded earlier, so the emitted run carries only the four
  /// frames BEFORE that gap (mean `0.9`) even though the gap had already
  /// been bridged into the live accumulator; the continuation restarts at
  /// the resume point (mean `0.8`). The four gap frames the split landed
  /// on belong to neither run and appear in neither aggregate.
  ///
  /// Boundaries are the ones `prefers_recorded_silence_when_splitting_long_speech`
  /// already pins (2048 / 4096 / 6144); this adds the aggregates.
  ///
  /// Mutations: emitting the live (post-bridge) accumulator instead of the
  /// snapshot gives `3.6 / 8 = 0.45` for the first run; not restarting the
  /// continuation's accumulator gives `(3.6 + 3.2) / 12 ≈ 0.567` for the
  /// second. Both red.
  #[test]
  fn force_split_at_a_recorded_gap_carries_the_pre_gap_aggregate() {
    let options = RunOptions::default()
      .with_min_run_duration(Duration::ZERO)
      .with_pad(Duration::ZERO)
      .with_min_gap_duration(Duration::from_millis(300))
      .with_min_gap_at_max_run(Duration::from_millis(64))
      .with_max_run_duration(Duration::from_millis(256));
    let mut segmenter = RunSegmenter::new(options);

    let mut probabilities = vec![0.9; 4];
    probabilities.extend([0.0; 4]);
    probabilities.extend([0.8; 4]);

    let runs = collect(&mut segmenter, &probabilities);

    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0].start_sample(), 0);
    assert_eq!(runs[0].end_sample(), 4 * HOP);
    assert_close(runs[0].mean_probability(), 0.9, "pre-gap run mean");
    assert_close(runs[0].peak_probability(), 0.9, "pre-gap run peak");

    assert_eq!(runs[1].start_sample(), 8 * HOP);
    assert_eq!(runs[1].end_sample(), 12 * HOP);
    assert_close(runs[1].mean_probability(), 0.8, "continuation mean");
    assert_close(runs[1].peak_probability(), 0.8, "continuation peak");
  }

  /// A hand-built `Run` carries zeroed aggregates until they are set:
  /// only runs the segmenter emits carry observed values.
  #[test]
  fn run_constructor_zeroes_the_aggregates() {
    let run = Run::new(0, 1_000, SampleRate::Rate16k);
    assert_close(run.mean_probability(), 0.0, "mean");
    assert_close(run.peak_probability(), 0.0, "peak");

    let annotated = run.with_mean_probability(0.5).with_peak_probability(0.9);
    assert_close(annotated.mean_probability(), 0.5, "mean");
    assert_close(annotated.peak_probability(), 0.9, "peak");
    assert_eq!(annotated.start_sample(), run.start_sample());
    assert_eq!(annotated.end_sample(), run.end_sample());
  }

  /// Malformed inputs whose canonical value is `0.0`: `NaN` (mapped, not
  /// clamped) and everything at or below the bottom of the range.
  const LOW_INVALID: [(f32, &str); 3] = [
    (f32::NAN, "NaN"),
    (f32::NEG_INFINITY, "-inf"),
    (-1.0, "finite -1.0"),
  ];

  /// Malformed inputs whose canonical value is `1.0`: everything at or
  /// above the top of the range.
  const HIGH_INVALID: [(f32, &str); 2] = [(f32::INFINITY, "+inf"), (2.0, "finite 2.0")];

  /// Every emitted run's aggregates must be finite and inside `[0, 1]`,
  /// whatever the producer fed in. This is the invariant the
  /// `push_probability` normalization exists to hold.
  #[track_caller]
  fn assert_in_range(run: &Run, what: &str) {
    let mean = run.mean_probability();
    let peak = run.peak_probability();
    assert!(
      mean.is_finite() && (0.0..=1.0).contains(&mean),
      "{what}: mean {mean} is not a finite probability in [0, 1]"
    );
    assert!(
      peak.is_finite() && (0.0..=1.0).contains(&peak),
      "{what}: peak {peak} is not a finite probability in [0, 1]"
    );
  }

  #[track_caller]
  fn assert_all_in_range(runs: &[Run], what: &str) {
    for (index, run) in runs.iter().enumerate() {
      assert_in_range(run, &format!("{what} run {index}"));
    }
  }

  fn open_options() -> RunOptions {
    RunOptions::default()
      .with_min_run_duration(Duration::ZERO)
      .with_pad(Duration::ZERO)
  }

  /// Close path 1 (a gap longer than `min_gap_duration` closes the run),
  /// above-range input. `+inf` / `2.0` land on an active frame and both
  /// canonicalize to `1.0`, so the three-frame run means `2.2 / 3` and
  /// peaks at exactly `1.0`.
  ///
  /// Mutation (drop the normalization): `+inf` gives an infinite mean and
  /// peak; `2.0` gives mean `1.0667` and peak `2.0`. Red.
  #[test]
  fn normal_close_clamps_above_range_probabilities() {
    for (invalid, name) in HIGH_INVALID {
      let mut segmenter = RunSegmenter::new(open_options());
      let mut probabilities = vec![0.6, invalid, 0.6];
      probabilities.extend([0.0; 5]);

      let runs = collect(&mut segmenter, &probabilities);

      assert_eq!(runs.len(), 1, "{name}");
      assert_all_in_range(&runs, name);
      assert_eq!(runs[0].start_sample(), 0, "{name}");
      assert_eq!(runs[0].end_sample(), 3 * HOP, "{name}");
      assert_close(
        runs[0].mean_probability(),
        2.2 / 3.0,
        &format!("{name} mean"),
      );
      assert_close(runs[0].peak_probability(), 1.0, &format!("{name} peak"));
    }
  }

  /// Close path 1, below-range input. `NaN` / `-inf` / `-1.0` all
  /// canonicalize to `0.0`, so the malformed frame opens a one-frame gap
  /// that the next frame bridges: three observations, mean `1.2 / 3`,
  /// peak `0.6`.
  ///
  /// Mutation: `NaN` gives a `NaN` mean, `-inf` an infinite one, and
  /// `-1.0` mean `0.0667`. Red.
  #[test]
  fn normal_close_maps_below_range_probabilities_to_silence() {
    for (invalid, name) in LOW_INVALID {
      let mut segmenter = RunSegmenter::new(open_options());
      let mut probabilities = vec![0.6, invalid, 0.6];
      probabilities.extend([0.0; 5]);

      let runs = collect(&mut segmenter, &probabilities);

      assert_eq!(runs.len(), 1, "{name}");
      assert_all_in_range(&runs, name);
      assert_eq!(runs[0].start_sample(), 0, "{name}");
      assert_eq!(runs[0].end_sample(), 3 * HOP, "{name}");
      assert_close(runs[0].mean_probability(), 0.4, &format!("{name} mean"));
      assert_close(runs[0].peak_probability(), 0.6, &format!("{name} peak"));
    }
  }

  /// Close path 2 (a bridged gap), below-range input inside the gap. The
  /// bridged frames are part of the run, so the malformed one reaches the
  /// aggregate as `0.0`: seven observations — `0.6, 0.6`, three bridged
  /// zeros, `0.6, 0.6` — mean `2.4 / 7`, peak `0.6`.
  ///
  /// Mutation: `NaN` / `-inf` poison the mean; `-1.0` gives `0.2`. Red.
  #[test]
  fn bridged_gap_maps_below_range_probabilities_to_silence() {
    for (invalid, name) in LOW_INVALID {
      let mut segmenter = RunSegmenter::new(open_options());
      let mut probabilities = vec![0.6, 0.6, 0.0, invalid, 0.0, 0.6, 0.6];
      probabilities.extend([0.0; 5]);

      let runs = collect(&mut segmenter, &probabilities);

      assert_eq!(runs.len(), 1, "{name}");
      assert_all_in_range(&runs, name);
      assert_eq!(runs[0].start_sample(), 0, "{name}");
      assert_eq!(runs[0].end_sample(), 7 * HOP, "{name}");
      assert_close(
        runs[0].mean_probability(),
        2.4 / 7.0,
        &format!("{name} mean"),
      );
      assert_close(runs[0].peak_probability(), 0.6, &format!("{name} peak"));
    }
  }

  /// Close path 2, above-range input on the frame that BRIDGES the gap.
  /// The malformed frame is what re-opens the run, and it is aggregated as
  /// `1.0`: eight observations, mean `3.4 / 8`, peak `1.0`.
  ///
  /// Mutation: `+inf` poisons mean and peak; `2.0` gives mean `0.55` and
  /// peak `2.0`. Red.
  #[test]
  fn bridged_gap_clamps_the_bridging_frame() {
    for (invalid, name) in HIGH_INVALID {
      let mut segmenter = RunSegmenter::new(open_options());
      let mut probabilities = vec![0.6, 0.6, 0.0, invalid, 0.0, 0.0, 0.6, 0.6];
      probabilities.extend([0.0; 5]);

      let runs = collect(&mut segmenter, &probabilities);

      assert_eq!(runs.len(), 1, "{name}");
      assert_all_in_range(&runs, name);
      assert_eq!(runs[0].start_sample(), 0, "{name}");
      assert_eq!(runs[0].end_sample(), 8 * HOP, "{name}");
      assert_close(runs[0].mean_probability(), 0.425, &format!("{name} mean"));
      assert_close(runs[0].peak_probability(), 1.0, &format!("{name} peak"));
    }
  }

  /// Close path 3 (a `max_run_duration` force-split), above-range input in
  /// the split-off run. Five observations before the split, mean
  /// `3.4 / 5`, peak `1.0`; the continuation is clean.
  ///
  /// Mutation: `+inf` poisons the split run; `2.0` gives mean `0.88` and
  /// peak `2.0`. Red.
  #[test]
  fn force_split_clamps_above_range_probabilities() {
    for (invalid, name) in HIGH_INVALID {
      let options = open_options().with_max_run_duration(Duration::from_millis(160));
      let mut segmenter = RunSegmenter::new(options);
      let mut probabilities = vec![0.6, invalid, 0.6, 0.6, 0.6];
      probabilities.extend([0.8; 3]);

      let runs = collect(&mut segmenter, &probabilities);

      assert_eq!(runs.len(), 2, "{name}");
      assert_all_in_range(&runs, name);
      assert_eq!(runs[0].end_sample(), 5 * HOP, "{name}");
      assert_close(
        runs[0].mean_probability(),
        0.68,
        &format!("{name} split mean"),
      );
      assert_close(
        runs[0].peak_probability(),
        1.0,
        &format!("{name} split peak"),
      );
      assert_eq!(runs[1].start_sample(), 5 * HOP, "{name}");
      assert_close(
        runs[1].mean_probability(),
        0.8,
        &format!("{name} continuation mean"),
      );
    }
  }

  /// Close path 3, below-range input bridged into the split-off run: five
  /// observations, mean `2.4 / 5`, peak `0.6`.
  ///
  /// Mutation: `NaN` / `-inf` poison the split run's mean; `-1.0` gives
  /// `0.28`. Red.
  #[test]
  fn force_split_maps_below_range_probabilities_to_silence() {
    for (invalid, name) in LOW_INVALID {
      let options = open_options().with_max_run_duration(Duration::from_millis(160));
      let mut segmenter = RunSegmenter::new(options);
      let mut probabilities = vec![0.6, invalid, 0.6, 0.6, 0.6];
      probabilities.extend([0.8; 3]);

      let runs = collect(&mut segmenter, &probabilities);

      assert_eq!(runs.len(), 2, "{name}");
      assert_all_in_range(&runs, name);
      assert_eq!(runs[0].end_sample(), 5 * HOP, "{name}");
      assert_close(
        runs[0].mean_probability(),
        0.48,
        &format!("{name} split mean"),
      );
      assert_close(
        runs[0].peak_probability(),
        0.6,
        &format!("{name} split peak"),
      );
      assert_eq!(runs[1].start_sample(), 5 * HOP, "{name}");
      assert_close(
        runs[1].mean_probability(),
        0.8,
        &format!("{name} continuation mean"),
      );
    }
  }

  /// Close path 4 (`finish` flushes the trailing run), below-range input
  /// in the unclosed trailing gap — which `take_trailing` folds in, so the
  /// malformed frame reaches the aggregate as `0.0`: mean `1.2 / 3`.
  ///
  /// Mutation: `NaN` / `-inf` poison the mean; `-1.0` gives `0.0667`. Red.
  #[test]
  fn finish_maps_below_range_probabilities_to_silence() {
    for (invalid, name) in LOW_INVALID {
      let mut segmenter = RunSegmenter::new(open_options());

      let runs = collect(&mut segmenter, &[0.6, 0.6, invalid]);

      assert_eq!(runs.len(), 1, "{name}");
      assert_all_in_range(&runs, name);
      assert_eq!(runs[0].end_sample(), 3 * HOP, "{name}");
      assert_close(runs[0].mean_probability(), 0.4, &format!("{name} mean"));
      assert_close(runs[0].peak_probability(), 0.6, &format!("{name} peak"));
    }
  }

  /// Close path 4, above-range input inside the trailing run.
  ///
  /// Mutation: `+inf` poisons mean and peak; `2.0` gives mean `1.0667`
  /// and peak `2.0`. Red.
  #[test]
  fn finish_clamps_above_range_probabilities() {
    for (invalid, name) in HIGH_INVALID {
      let mut segmenter = RunSegmenter::new(open_options());

      let runs = collect(&mut segmenter, &[0.6, invalid, 0.6]);

      assert_eq!(runs.len(), 1, "{name}");
      assert_all_in_range(&runs, name);
      assert_eq!(runs[0].end_sample(), 3 * HOP, "{name}");
      assert_close(
        runs[0].mean_probability(),
        2.2 / 3.0,
        &format!("{name} mean"),
      );
      assert_close(runs[0].peak_probability(), 1.0, &format!("{name} peak"));
    }
  }

  fn boundaries(options: RunOptions, probabilities: &[f32]) -> Vec<(u64, u64)> {
    let mut segmenter = RunSegmenter::new(options);
    collect(&mut segmenter, probabilities)
      .iter()
      .map(|run| (run.start_sample(), run.end_sample()))
      .collect()
  }

  /// The `NaN` -> `0.0` mapping is behaviour-preserving for the state
  /// machine whenever both effective thresholds are above zero (every
  /// default configuration): a `NaN` compared against a positive threshold
  /// is false, and so is `0.0`, so the identical branch is taken on every
  /// comparison in `push_probability` and `split_at_max_duration`.
  ///
  /// This test is the evidence, and it is the one test here that must stay
  /// GREEN with the normalization removed: it asserts that substituting
  /// `NaN` for a `0.0` frame moves no boundary, mid-run (bridged), inside
  /// a closing gap, and on the frame a force-split decides its
  /// continuation from.
  #[test]
  fn nan_frames_do_not_move_segment_boundaries() {
    // A NaN mid-run (bridged), plus a NaN inside the gap that closes the
    // first run.
    let mut with_nan = vec![0.6, 0.6, f32::NAN, 0.6, 0.6];
    with_nan.extend([0.0, f32::NAN, 0.0, 0.0, 0.0]);
    with_nan.extend([0.6, 0.6]);
    with_nan.extend([0.0; 5]);
    let with_zero: Vec<f32> = with_nan
      .iter()
      .map(|p| if p.is_nan() { 0.0 } else { *p })
      .collect();

    let nan_boundaries = boundaries(open_options(), &with_nan);
    assert_eq!(
      nan_boundaries,
      boundaries(open_options(), &with_zero),
      "a NaN frame must segment exactly like the 0.0 frame it stands for"
    );
    assert_eq!(
      nan_boundaries,
      vec![(0, 5 * HOP), (10 * HOP, 12 * HOP)],
      "and both must match the pinned boundaries"
    );

    // A NaN on the frame that a force-split reads to decide whether to
    // open a continuation run (`probability >= start_threshold`).
    let split_options = open_options().with_max_run_duration(Duration::from_millis(160));
    let mut with_nan = vec![0.6; 5];
    with_nan.push(f32::NAN);
    with_nan.extend([0.6; 3]);
    let with_zero: Vec<f32> = with_nan
      .iter()
      .map(|p| if p.is_nan() { 0.0 } else { *p })
      .collect();

    let nan_boundaries = boundaries(split_options.clone(), &with_nan);
    assert_eq!(
      nan_boundaries,
      boundaries(split_options, &with_zero),
      "a NaN at the split decision frame must segment exactly like 0.0"
    );
    assert_eq!(
      nan_boundaries,
      vec![(0, 5 * HOP), (6 * HOP, 9 * HOP)],
      "and both must match the pinned boundaries"
    );
  }

  /// The documented exception to the paragraph above: a threshold of
  /// exactly `0.0` is the one configuration where `NaN` and `0.0` compare
  /// differently, because `0.0 >= 0.0` holds. With a zero start threshold
  /// a `NaN` frame now OPENS a run; with a zero end threshold it now
  /// SUSTAINS one. Pinned so the exception is a tested contract rather
  /// than a surprise.
  #[test]
  fn zero_thresholds_are_the_documented_exception() {
    let options = open_options()
      .with_start_threshold(0.0)
      .with_min_gap_duration(Duration::ZERO);
    let mut segmenter = RunSegmenter::new(options);

    let runs = collect(&mut segmenter, &[f32::NAN, 0.0]);

    assert_eq!(runs.len(), 1, "a zero start threshold opens on NaN -> 0.0");
    assert_all_in_range(&runs, "zero start threshold");
    assert_eq!(runs[0].start_sample(), 0);
    assert_eq!(runs[0].end_sample(), HOP);

    let options = open_options()
      .with_end_threshold(0.0)
      .with_min_gap_duration(Duration::ZERO);
    let mut segmenter = RunSegmenter::new(options);

    let runs = collect(&mut segmenter, &[0.6, f32::NAN, 0.6]);

    assert_eq!(runs.len(), 1, "a zero end threshold sustains through NaN");
    assert_all_in_range(&runs, "zero end threshold");
    assert_eq!(runs[0].end_sample(), 3 * HOP);
    assert_close(runs[0].mean_probability(), 0.4, "mean");
  }
}
