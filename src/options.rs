use core::time::Duration;

use crate::error::{Error, Result};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Sample rates this crate supports directly (8 kHz and 16 kHz).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum SampleRate {
  /// 8 kHz sample rate, which uses smaller chunks and less context.
  #[cfg_attr(feature = "serde", serde(rename = "8k"))]
  Rate8k,
  /// 16 kHz sample rate, which uses larger chunks and more context for better accuracy.
  #[cfg_attr(feature = "serde", serde(rename = "16k"))]
  #[default]
  Rate16k,
}

impl SampleRate {
  /// Returns the sample rate in Hz.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn hz(self) -> u32 {
    match self {
      Self::Rate8k => 8_000,
      Self::Rate16k => 16_000,
    }
  }

  /// Returns the number of samples in the legacy default model chunk for
  /// this sample rate (`256` at 8 kHz, `512` at 16 kHz).
  ///
  /// This is the default frame hop a [`RunSegmenter`](crate::RunSegmenter)
  /// advances by when no backend overrides it — the Silero geometry. A
  /// backend that declares a different hop drives the segmenter through
  /// [`VadBackend::frame_hop`](crate::VadBackend::frame_hop) instead, so
  /// this value is only the fallback, not a fixed contract.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn chunk_samples(self) -> usize {
    match self {
      Self::Rate8k => 256,
      Self::Rate16k => 512,
    }
  }

  /// Create a `SampleRate` from a raw Hz value, returning an error if the rate is not supported.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn from_hz(rate: u32) -> Result<Self> {
    match rate {
      8_000 => Ok(Self::Rate8k),
      16_000 => Ok(Self::Rate16k),
      other => Err(Error::UnsupportedSampleRate { rate: other }),
    }
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_start_threshold() -> f32 {
  0.5
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_min_run_duration() -> Duration {
  Duration::from_millis(250)
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_min_gap_duration() -> Duration {
  Duration::from_millis(100)
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_min_gap_at_max_run() -> Duration {
  Duration::from_millis(98)
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_pad() -> Duration {
  Duration::from_millis(30)
}

/// Configuration for turning a frame-probability sequence into
/// [`Run`](crate::Run)s.
///
/// Domain-neutral: the same thresholds and durations drive speech
/// segmentation, sound-event detection, or any other "probability
/// sequence to contiguous runs" problem. The speech-flavoured accessor
/// names (`min_speech_duration`, `speech_pad`, ...) are kept as forwarding
/// accessors, so a VAD caller written against [`SpeechOptions`] compiles
/// unchanged; see the correspondence table in the [crate docs](crate).
///
/// # Serde
///
/// With the `serde` feature the fields serialize under their **neutral**
/// names (`min_run_duration`, `min_gap_duration`, `min_gap_at_max_run`,
/// `max_run_duration`, `pad`). Each also accepts its 0.1 speech-flavoured
/// name as a deserialization alias, so configuration profiles persisted by
/// 0.1 still load.
///
/// Deserialization is held to the same contract as the setters:
/// `start_threshold` and `end_threshold` are clamped into
/// [`[MIN_THRESHOLD, 1]`](RunOptions::MIN_THRESHOLD) on the way in
/// exactly as [`Self::set_start_threshold`] / [`Self::set_end_threshold`]
/// clamp them, so a hand-edited profile cannot install a threshold no
/// setter would have stored. Every field the serializer omits is optional
/// on the way back in, so a serialized `RunOptions` always round-trips.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RunOptions {
  #[cfg_attr(feature = "serde", serde(default))]
  sample_rate: SampleRate,
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_start_threshold",
      deserialize_with = "deserialize_threshold"
    )
  )]
  start_threshold: f32,
  #[cfg_attr(
    feature = "serde",
    serde(
      default,
      skip_serializing_if = "Option::is_none",
      deserialize_with = "deserialize_optional_threshold"
    )
  )]
  end_threshold: Option<f32>,
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_min_run_duration",
      alias = "min_speech_duration",
      with = "humantime_serde"
    )
  )]
  min_run_duration: Duration,
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_min_gap_duration",
      alias = "min_silence_duration",
      with = "humantime_serde"
    )
  )]
  min_gap_duration: Duration,
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_min_gap_at_max_run",
      alias = "min_silence_at_max_speech",
      with = "humantime_serde"
    )
  )]
  min_gap_at_max_run: Duration,
  #[cfg_attr(
    feature = "serde",
    serde(
      default,
      skip_serializing_if = "Option::is_none",
      alias = "max_speech_duration",
      with = "humantime_serde::option"
    )
  )]
  max_run_duration: Option<Duration>,
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_pad",
      alias = "speech_pad",
      with = "humantime_serde"
    )
  )]
  pad: Duration,
}

/// The speech-flavoured name for [`RunOptions`].
///
/// A plain alias: identical type, identical constructors, identical
/// defaults. Speech-named accessors (`min_speech_duration`, `speech_pad`,
/// ...) forward to their neutral counterparts.
pub type SpeechOptions = RunOptions;

impl Default for RunOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

impl RunOptions {
  /// The lowest threshold these options will store.
  ///
  /// [`start_threshold`](Self::start_threshold) and an explicitly set
  /// [`end_threshold`](Self::end_threshold) are clamped into
  /// `[MIN_THRESHOLD, 1.0]` — on the setter path and, with the `serde`
  /// feature, on the deserialize path alike — and the derived end
  /// threshold bottoms out at the same value. Two properties follow, and
  /// the rest of the crate depends on both:
  ///
  /// - **Every effective threshold is strictly positive.** A frame that
  ///   [`RunSegmenter::push_probability`](crate::RunSegmenter::push_probability)
  ///   canonicalizes to `0.0` therefore satisfies no threshold, which is
  ///   what makes that canonicalization behaviour-preserving rather than
  ///   an exception to reason about.
  /// - **The hysteresis window is never inverted**:
  ///   `end_threshold() <= start_threshold()` for every start threshold
  ///   these options can hold. An inverted window is not cosmetic: a
  ///   frame in the resulting `[start, end)` band satisfies the start
  ///   comparison — which clears the tentative gap and restarts it on the
  ///   same frame — without satisfying the end comparison that sustains
  ///   the run, so the gap age never grows and the run never closes.
  ///
  /// `0.01` rather than an arbitrarily small positive value: it is
  /// already the floor of the derived end threshold (`start - 0.15`,
  /// floored), so the two agree by construction instead of by
  /// coincidence — which is what makes the non-inversion property
  /// provable. It is also the point below which a threshold stops
  /// discriminating: nothing under one percent rejects a frame a
  /// detector would realistically emit.
  pub const MIN_THRESHOLD: f32 = 0.01;

  /// Create a new `RunOptions` with default values.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      sample_rate: SampleRate::Rate16k,
      start_threshold: default_start_threshold(),
      end_threshold: None,
      min_run_duration: default_min_run_duration(),
      min_gap_duration: default_min_gap_duration(),
      // Matches the upstream silero-vad Python default (0.098 s).
      min_gap_at_max_run: default_min_gap_at_max_run(),
      max_run_duration: None,
      pad: default_pad(),
    }
  }

  /// Returns the sample rate the timeline is measured in.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sample_rate(&self) -> SampleRate {
    self.sample_rate
  }

  /// Returns the start threshold, which is the minimum probability
  /// required to open a run.
  ///
  /// Always in [`[MIN_THRESHOLD, 1]`](Self::MIN_THRESHOLD) — in
  /// particular, always strictly positive.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn start_threshold(&self) -> f32 {
    self.start_threshold
  }

  /// Returns the effective end threshold.
  ///
  /// If a user-supplied end threshold would break the hysteresis
  /// window, this falls back to the same derived threshold used by the
  /// default configuration so behavior stays stable regardless of
  /// builder call order.
  ///
  /// Always in [`[MIN_THRESHOLD, 1]`](Self::MIN_THRESHOLD) and never
  /// above [`start_threshold`](Self::start_threshold), for every
  /// threshold pair these options can hold.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn end_threshold(&self) -> f32 {
    effective_end_threshold(
      self.start_threshold,
      self
        .end_threshold
        .unwrap_or_else(|| default_end_threshold(self.start_threshold)),
    )
  }

  /// Returns the minimum duration of an emitted run.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_run_duration(&self) -> Duration {
    self.min_run_duration
  }

  /// Returns the minimum duration of a below-threshold gap required to close a run.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_gap_duration(&self) -> Duration {
    self.min_gap_duration
  }

  /// Returns the minimum gap duration used as a preferred split point when the maximum run duration is reached.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_gap_at_max_run(&self) -> Duration {
    self.min_gap_at_max_run
  }

  /// Returns the maximum duration of a run before the segmenter force-splits it.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_run_duration(&self) -> Option<Duration> {
    self.max_run_duration
  }

  /// Returns the amount of padding added to each side of an emitted run.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn pad(&self) -> Duration {
    self.pad
  }

  /// Returns the minimum duration of an emitted run, in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn min_run_samples(&self) -> u64 {
    ms_to_samples(self.min_run_duration, self.sample_rate)
  }

  /// Returns the minimum gap required to close a run, in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn min_gap_samples(&self) -> u64 {
    ms_to_samples(self.min_gap_duration, self.sample_rate)
  }

  /// Returns the minimum gap usable as a preferred split point when the maximum run duration is reached, in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn min_gap_at_max_run_samples(&self) -> u64 {
    ms_to_samples(self.min_gap_at_max_run, self.sample_rate)
  }

  /// Returns the maximum run duration before force-splitting, in
  /// samples, for the sample rate's legacy default model-chunk hop.
  ///
  /// This assumes the bundled ONNX backend's hop (`chunk_samples` per
  /// probability). A backend that declares a different frame hop force-
  /// splits on that active hop instead: the
  /// [`RunSegmenter`](crate::RunSegmenter) driving
  /// [`detect_speech_with`](crate::detect_speech_with) applies the
  /// backend's [`frame_hop`](crate::VadBackend::frame_hop) automatically,
  /// so those runs honor the backend's hop rather than this method's
  /// chunk assumption.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn max_run_samples(&self) -> Option<u64> {
    self.max_run_samples_for_frame(self.sample_rate.chunk_samples() as u64)
  }

  /// Returns the maximum run length in samples before force-splitting for
  /// a given frame hop.
  ///
  /// Matches the upstream silero-vad derivation at the active frame hop:
  /// - `- frame_hop` because the split check runs on the next frame after
  ///   the limit is exceeded — the timeline advances by the consuming
  ///   segmenter's hop (the sample rate's model chunk for the ONNX
  ///   backend, but e.g. 4096 for another), not by `chunk_samples`
  /// - `- 2 * pad_samples` because emitted runs pad both the end of the
  ///   current run and the start of the next one
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn max_run_samples_for_frame(&self, frame_hop: u64) -> Option<u64> {
    self.max_run_duration.map(|duration| {
      ms_to_samples(duration, self.sample_rate)
        .saturating_sub(frame_hop)
        .saturating_sub(self.pad_samples().saturating_mul(2))
    })
  }

  /// Returns the amount of padding added to each side of an emitted run, in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn pad_samples(&self) -> u64 {
    ms_to_samples(self.pad, self.sample_rate)
  }

  /// Set the sample rate the timeline is measured in.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_sample_rate(mut self, sample_rate: SampleRate) -> Self {
    self.set_sample_rate(sample_rate);
    self
  }

  /// Set the start threshold, the minimum probability required to open a
  /// run. Defaults to `0.5`.
  ///
  /// The value is clamped into
  /// [`[MIN_THRESHOLD, 1]`](Self::MIN_THRESHOLD); a non-finite one
  /// becomes `MIN_THRESHOLD`. A start threshold is therefore always
  /// strictly positive — `0.0` ("every frame opens a run") is not a
  /// configuration these options can express.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_start_threshold(mut self, threshold: f32) -> Self {
    self.set_start_threshold(threshold);
    self
  }

  /// Set the preferred end threshold.
  ///
  /// The stored value is clamped into
  /// [`[MIN_THRESHOLD, 1]`](Self::MIN_THRESHOLD), so it is always
  /// strictly positive. When the threshold is later read via
  /// [`Self::end_threshold`], it is also checked against the current
  /// start threshold. Invalid combinations fall back to the default
  /// derived hysteresis rule even if builder methods are called in a
  /// different order.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_end_threshold(mut self, threshold: f32) -> Self {
    self.set_end_threshold(threshold);
    self
  }

  /// Clear the end threshold, causing it to be automatically derived from the start threshold with a fixed offset.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn clear_end_threshold(mut self) -> Self {
    self.end_threshold = None;
    self
  }

  /// Set the minimum duration of an emitted run.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_run_duration(mut self, duration: Duration) -> Self {
    self.set_min_run_duration(duration);
    self
  }

  /// Set the minimum duration of a below-threshold gap required to close a run.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_gap_duration(mut self, duration: Duration) -> Self {
    self.set_min_gap_duration(duration);
    self
  }

  /// Set the minimum gap that can be used as a preferred split point when the maximum run duration is reached.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_gap_at_max_run(mut self, duration: Duration) -> Self {
    self.set_min_gap_at_max_run(duration);
    self
  }

  /// Set the maximum duration of a run before the segmenter force-splits it.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_run_duration(mut self, duration: Duration) -> Self {
    self.set_max_run_duration(duration);
    self
  }

  /// Clear the maximum run duration, disabling force-splitting by run length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn clear_max_run_duration(mut self) -> Self {
    self.max_run_duration = None;
    self
  }

  /// Set the amount of padding added to each side of an emitted run.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_pad(mut self, pad: Duration) -> Self {
    self.set_pad(pad);
    self
  }

  /// Set the sample rate the timeline is measured in.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_sample_rate(&mut self, sample_rate: SampleRate) -> &mut Self {
    self.sample_rate = sample_rate;
    self
  }

  /// Set the start threshold, the minimum probability required to open a
  /// run. Defaults to `0.5`.
  ///
  /// The value is clamped into
  /// [`[MIN_THRESHOLD, 1]`](Self::MIN_THRESHOLD); a non-finite one
  /// becomes `MIN_THRESHOLD`. A start threshold is therefore always
  /// strictly positive — `0.0` ("every frame opens a run") is not a
  /// configuration these options can express.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_start_threshold(&mut self, threshold: f32) -> &mut Self {
    self.start_threshold = sanitize_threshold(threshold);
    self
  }

  /// Set the preferred end threshold.
  ///
  /// The stored value is clamped into
  /// [`[MIN_THRESHOLD, 1]`](Self::MIN_THRESHOLD), so it is always
  /// strictly positive. When the threshold is later read via
  /// [`Self::end_threshold`], it is also checked against the current
  /// start threshold. Invalid combinations fall back to the default
  /// derived hysteresis rule even if builder methods are called in a
  /// different order.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_end_threshold(&mut self, threshold: f32) -> &mut Self {
    self.end_threshold = Some(sanitize_threshold(threshold));
    self
  }

  /// Set the minimum duration of an emitted run as a `Duration`.
  /// Sub-second precision is supported according to the precision of `Duration`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_run_duration(&mut self, duration: Duration) -> &mut Self {
    self.min_run_duration = duration;
    self
  }

  /// Set the minimum duration of a below-threshold gap required to close a run, as a `Duration`.
  /// Sub-second precision is supported according to the precision of `Duration`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_gap_duration(&mut self, duration: Duration) -> &mut Self {
    self.min_gap_duration = duration;
    self
  }

  /// Set the minimum gap, as a `Duration`, that can be used as a preferred split point
  /// when the maximum run duration is reached. Sub-second precision is supported according
  /// to the precision of `Duration`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_gap_at_max_run(&mut self, duration: Duration) -> &mut Self {
    self.min_gap_at_max_run = duration;
    self
  }

  /// Set the maximum duration of a run, as a `Duration`, before the segmenter
  /// force-splits it. Sub-second precision is supported according to the precision of `Duration`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_run_duration(&mut self, duration: Duration) -> &mut Self {
    self.max_run_duration = Some(duration);
    self
  }

  /// Set the amount of padding added to each side of an emitted run.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_pad(&mut self, pad: Duration) -> &mut Self {
    self.pad = pad;
    self
  }
}

/// Speech-flavoured shell over the neutral accessors.
///
/// Every method here forwards verbatim to its [`RunOptions`] counterpart —
/// there is no separate state and no behavioural difference. They exist so
/// VAD callers written against 0.1's `SpeechOptions` keep compiling; see
/// the correspondence table in the [crate docs](crate).
impl RunOptions {
  /// Returns the minimum duration of detected speech segments.
  ///
  /// Speech-flavoured name for [`min_run_duration`](Self::min_run_duration).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_speech_duration(&self) -> Duration {
    self.min_run_duration()
  }

  /// Returns the minimum duration of silence required to close a detected speech segment.
  ///
  /// Speech-flavoured name for [`min_gap_duration`](Self::min_gap_duration).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_silence_duration(&self) -> Duration {
    self.min_gap_duration()
  }

  /// Returns the minimum silence duration used as a preferred split point when the maximum speech duration is reached.
  ///
  /// Speech-flavoured name for [`min_gap_at_max_run`](Self::min_gap_at_max_run).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_silence_at_max_speech(&self) -> Duration {
    self.min_gap_at_max_run()
  }

  /// Returns the maximum duration of a speech segment before the segmenter force-splits it.
  ///
  /// Speech-flavoured name for [`max_run_duration`](Self::max_run_duration).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_speech_duration(&self) -> Option<Duration> {
    self.max_run_duration()
  }

  /// Returns the amount of padding to add around detected speech segments.
  ///
  /// Speech-flavoured name for [`pad`](Self::pad).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn speech_pad(&self) -> Duration {
    self.pad()
  }

  /// Returns the minimum duration of detected speech segments, in samples.
  ///
  /// Speech-flavoured name for [`min_run_samples`](Self::min_run_samples).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn min_speech_samples(&self) -> u64 {
    self.min_run_samples()
  }

  /// Returns the minimum duration of silence required to close a detected speech segment, in samples.
  ///
  /// Speech-flavoured name for [`min_gap_samples`](Self::min_gap_samples).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn min_silence_samples(&self) -> u64 {
    self.min_gap_samples()
  }

  /// Returns the minimum silence duration used as a preferred split point when max speech duration is reached, in samples.
  ///
  /// Speech-flavoured name for [`min_gap_at_max_run_samples`](Self::min_gap_at_max_run_samples).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn min_silence_at_max_speech_samples(&self) -> u64 {
    self.min_gap_at_max_run_samples()
  }

  /// Returns the maximum speech duration before force-splitting, in samples.
  ///
  /// Speech-flavoured name for [`max_run_samples`](Self::max_run_samples).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn max_speech_samples(&self) -> Option<u64> {
    self.max_run_samples()
  }

  /// Speech-flavoured name for [`max_run_samples_for_frame`](Self::max_run_samples_for_frame).
  ///
  /// Crate-internal, so the only caller is the pinned options test; gated
  /// on `test` rather than carrying a blanket `dead_code` allow.
  #[cfg(test)]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn max_speech_samples_for_frame(&self, frame_hop: u64) -> Option<u64> {
    self.max_run_samples_for_frame(frame_hop)
  }

  /// Returns the amount of padding to add around detected speech segments, in samples.
  ///
  /// Speech-flavoured name for [`pad_samples`](Self::pad_samples).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn speech_pad_samples(&self) -> u64 {
    self.pad_samples()
  }

  /// Set the minimum duration of detected speech segments.
  ///
  /// Speech-flavoured name for [`with_min_run_duration`](Self::with_min_run_duration).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_speech_duration(self, duration: Duration) -> Self {
    self.with_min_run_duration(duration)
  }

  /// Set the minimum duration of silence required to close a detected speech segment.
  ///
  /// Speech-flavoured name for [`with_min_gap_duration`](Self::with_min_gap_duration).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_silence_duration(self, duration: Duration) -> Self {
    self.with_min_gap_duration(duration)
  }

  /// Set the minimum silence duration that can be used as a preferred split point when maximum speech duration is reached.
  ///
  /// Speech-flavoured name for [`with_min_gap_at_max_run`](Self::with_min_gap_at_max_run).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_silence_at_max_speech(self, duration: Duration) -> Self {
    self.with_min_gap_at_max_run(duration)
  }

  /// Set the maximum duration of a speech segment before the segmenter force-splits it.
  ///
  /// Speech-flavoured name for [`with_max_run_duration`](Self::with_max_run_duration).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_speech_duration(self, duration: Duration) -> Self {
    self.with_max_run_duration(duration)
  }

  /// Clear the maximum speech duration, disabling force-splitting by segment length.
  ///
  /// Speech-flavoured name for [`clear_max_run_duration`](Self::clear_max_run_duration).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn clear_max_speech_duration(self) -> Self {
    self.clear_max_run_duration()
  }

  /// Set the amount of padding to add around detected speech segments.
  ///
  /// Speech-flavoured name for [`with_pad`](Self::with_pad).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_speech_pad(self, pad: Duration) -> Self {
    self.with_pad(pad)
  }

  /// Set the minimum duration of detected speech segments.
  ///
  /// Speech-flavoured name for [`set_min_run_duration`](Self::set_min_run_duration).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_speech_duration(&mut self, duration: Duration) -> &mut Self {
    self.set_min_run_duration(duration)
  }

  /// Set the minimum duration of silence required to close a detected speech segment.
  ///
  /// Speech-flavoured name for [`set_min_gap_duration`](Self::set_min_gap_duration).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_silence_duration(&mut self, duration: Duration) -> &mut Self {
    self.set_min_gap_duration(duration)
  }

  /// Set the minimum silence duration that can be used as a preferred split point
  /// when maximum speech duration is reached.
  ///
  /// Speech-flavoured name for [`set_min_gap_at_max_run`](Self::set_min_gap_at_max_run).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_silence_at_max_speech(&mut self, duration: Duration) -> &mut Self {
    self.set_min_gap_at_max_run(duration)
  }

  /// Set the maximum duration of a speech segment before the segmenter force-splits it.
  ///
  /// Speech-flavoured name for [`set_max_run_duration`](Self::set_max_run_duration).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_speech_duration(&mut self, duration: Duration) -> &mut Self {
    self.set_max_run_duration(duration)
  }

  /// Set the amount of padding to add around detected speech segments.
  ///
  /// Speech-flavoured name for [`set_pad`](Self::set_pad).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_speech_pad(&mut self, pad: Duration) -> &mut Self {
    self.set_pad(pad)
  }
}

#[inline]
pub(crate) const fn ms_to_samples(duration: Duration, sample_rate: SampleRate) -> u64 {
  let samples = (duration.as_millis() * (sample_rate.hz() as u128)) / 1_000;

  if samples > u64::MAX as u128 {
    u64::MAX
  } else {
    samples as u64
  }
}

/// Clamp a threshold into the permitted `[MIN_THRESHOLD, 1.0]` range.
///
/// A non-finite threshold has no ordering against frame probabilities at
/// all, so it becomes the floor — the least permissive interpretation of
/// "unusable input" that is still a threshold. See
/// [`RunOptions::MIN_THRESHOLD`] for why the floor is strictly positive.
#[inline]
const fn sanitize_threshold(value: f32) -> f32 {
  if value.is_finite() {
    value.clamp(RunOptions::MIN_THRESHOLD, 1.0)
  } else {
    RunOptions::MIN_THRESHOLD
  }
}

/// Sanitize a threshold arriving through `Deserialize`.
///
/// The derived `Deserialize` writes fields directly and never calls
/// [`RunOptions::set_start_threshold`], so without this a persisted
/// profile could install a zero, negative, or non-finite threshold that
/// no setter would have stored. Deserialization is an input boundary
/// like any other; it applies the same [`sanitize_threshold`] the
/// setters do, so both paths store the same value for the same input.
#[cfg(feature = "serde")]
fn deserialize_threshold<'de, D>(deserializer: D) -> core::result::Result<f32, D::Error>
where
  D: serde::Deserializer<'de>,
{
  Ok(sanitize_threshold(f32::deserialize(deserializer)?))
}

/// The [`Option`] form of [`deserialize_threshold`], for the
/// [`RunOptions::set_end_threshold`] field.
///
/// A `deserialize_with` on an `Option` field suppresses serde's implicit
/// "a missing `Option` field is `None`" rule, so the field carries an
/// explicit `#[serde(default)]` alongside this.
#[cfg(feature = "serde")]
fn deserialize_optional_threshold<'de, D>(
  deserializer: D,
) -> core::result::Result<Option<f32>, D::Error>
where
  D: serde::Deserializer<'de>,
{
  Ok(Option::<f32>::deserialize(deserializer)?.map(sanitize_threshold))
}

/// The end threshold derived from a start threshold when none was set.
///
/// A fixed `0.15` of hysteresis below the start threshold, floored at
/// [`RunOptions::MIN_THRESHOLD`]. Because the start threshold shares that
/// floor, the result is never above it: for `start >= 0.16` the
/// subtraction wins and is strictly below `start`, and for a smaller
/// `start` the floor wins and equals `MIN_THRESHOLD <= start`.
#[inline]
const fn default_end_threshold(start_threshold: f32) -> f32 {
  sanitize_threshold((sanitize_threshold(start_threshold) - 0.15).max(RunOptions::MIN_THRESHOLD))
}

/// Resolve the end threshold actually used, rejecting a configured value
/// that would invert the hysteresis window in favour of the derived one.
///
/// Both branches return a value at most `start_threshold` and at least
/// [`RunOptions::MIN_THRESHOLD`], so the window is never inverted and the
/// end threshold is never zero, whatever the caller supplied and in
/// whatever builder order.
#[inline]
const fn effective_end_threshold(start_threshold: f32, end_threshold: f32) -> f32 {
  let start_threshold = sanitize_threshold(start_threshold);
  let end_threshold = sanitize_threshold(end_threshold);

  if end_threshold < start_threshold {
    end_threshold
  } else {
    default_end_threshold(start_threshold)
  }
}

#[cfg(test)]
mod tests {
  use std::time::Duration;

  use super::{RunOptions, SampleRate, SpeechOptions, ms_to_samples};

  #[test]
  fn sample_rate_chunk_contract_matches_silero_model() {
    // The legacy default frame hop. Context geometry (32/64) is
    // Session-specific and lives with the Silero backend, not here.
    assert_eq!(SampleRate::Rate16k.chunk_samples(), 512);
    assert_eq!(SampleRate::Rate8k.chunk_samples(), 256);
  }

  #[test]
  fn speech_config_defaults_match_expected_streaming_behavior() {
    let config = SpeechOptions::default();
    assert_eq!(config.sample_rate(), SampleRate::Rate16k);
    assert_eq!(config.start_threshold(), 0.5);
    assert_eq!(config.end_threshold(), 0.35);
    assert_eq!(config.min_speech_duration(), Duration::from_millis(250));
    assert_eq!(config.min_silence_duration(), Duration::from_millis(100));
    assert_eq!(
      config.min_silence_at_max_speech(),
      Duration::from_millis(98)
    );
    assert_eq!(config.max_speech_duration(), None);
    assert_eq!(config.speech_pad(), Duration::from_millis(30));
  }

  #[test]
  fn ms_to_samples_uses_stream_rate() {
    assert_eq!(
      ms_to_samples(Duration::from_millis(100), SampleRate::Rate16k),
      1_600
    );
    assert_eq!(
      ms_to_samples(Duration::from_millis(100), SampleRate::Rate8k),
      800
    );
  }

  #[test]
  fn end_threshold_falls_back_to_default_gap_when_builder_order_would_invert_hysteresis() {
    let options = SpeechOptions::default()
      .with_start_threshold(0.4)
      .with_end_threshold(0.6);
    assert!(options.end_threshold() < options.start_threshold());
    assert!((options.end_threshold() - 0.25).abs() < f32::EPSILON);

    let reordered = SpeechOptions::default()
      .with_end_threshold(0.6)
      .with_start_threshold(0.4);
    assert!(reordered.end_threshold() < reordered.start_threshold());
    assert!((options.end_threshold() - reordered.end_threshold()).abs() < f32::EPSILON);

    let valid = SpeechOptions::default()
      .with_start_threshold(0.6)
      .with_end_threshold(0.2);
    assert!((valid.end_threshold() - 0.2).abs() < f32::EPSILON);
  }

  #[test]
  fn max_speech_duration_converts_to_samples_with_stream_lookahead_and_padding() {
    let options = SpeechOptions::default()
      .with_speech_pad(Duration::from_millis(30))
      .with_max_speech_duration(Duration::from_millis(1_000));
    assert_eq!(
      options.max_speech_duration(),
      Some(Duration::from_millis(1_000))
    );
    assert_eq!(options.min_silence_at_max_speech_samples(), 1_568);
    assert_eq!(options.max_speech_samples(), Some(14_528));
    // The public getter assumes the sample rate's native model chunk
    // (512 at 16 kHz) as the one-frame lookahead.
    assert_eq!(options.max_speech_samples_for_frame(512), Some(14_528));
    // A backend that declares a larger frame subtracts THAT frame for the
    // lookahead, not the 512-sample chunk: 16_000 − 4_096 − 2·480 = 10_944.
    assert_eq!(options.max_speech_samples_for_frame(4_096), Some(10_944));
  }

  /// Every threshold `RunOptions` stores is strictly positive. This is
  /// what makes `push_probability`'s canonicalization behaviour-
  /// preserving: a frame canonicalized to `0.0` is below every permitted
  /// threshold, so it can neither open nor sustain a run.
  ///
  /// Mutation (clamp into `[0.0, 1.0]` as before): `0.0`, `-1.0`,
  /// `-inf` and `NaN` all store `0.0`. Red.
  #[test]
  fn thresholds_are_clamped_strictly_above_zero() {
    for (raw, name) in [
      (0.0_f32, "zero"),
      (-1.0, "finite negative"),
      (f32::NEG_INFINITY, "-inf"),
      (f32::NAN, "NaN"),
      (RunOptions::MIN_THRESHOLD / 2.0, "below the floor"),
    ] {
      let options = SpeechOptions::default()
        .with_start_threshold(raw)
        .with_end_threshold(raw);

      assert_eq!(
        options.start_threshold(),
        RunOptions::MIN_THRESHOLD,
        "{name}: start threshold"
      );
      assert!(
        options.start_threshold() > 0.0,
        "{name}: start threshold must be strictly positive"
      );
      assert!(
        options.end_threshold() > 0.0,
        "{name}: end threshold must be strictly positive"
      );
    }
  }

  /// The hysteresis window is never inverted: for EVERY start threshold
  /// the setters permit, with the end threshold derived or explicitly
  /// set anywhere in the range, `end_threshold() <= start_threshold()`
  /// and both are strictly positive.
  ///
  /// A start threshold below the derived end threshold's `0.01` floor is
  /// what inverts it, and an inverted window is not cosmetic: a frame in
  /// the `[start, end)` band satisfies the start comparison — which
  /// clears the tentative gap and restarts it on the same frame —
  /// without satisfying the end comparison that sustains the run, so the
  /// gap age never grows and the run never closes.
  ///
  /// Mutation (drop the floor, or set it below `default_end_threshold`'s
  /// own `0.01`): every start threshold under `0.01` derives an end
  /// threshold above it. Red.
  #[test]
  fn the_hysteresis_window_is_never_inverted_across_the_permitted_range() {
    for step in 0..=1_000u32 {
      let raw = step as f32 / 1_000.0;
      let derived = SpeechOptions::default().with_start_threshold(raw);
      assert!(
        derived.end_threshold() > 0.0,
        "start {raw}: derived end threshold must be strictly positive"
      );
      assert!(
        derived.end_threshold() <= derived.start_threshold(),
        "start {raw}: derived end {} exceeds start {}",
        derived.end_threshold(),
        derived.start_threshold()
      );

      for end_step in 0..=20u32 {
        let raw_end = end_step as f32 / 20.0;
        let explicit = derived.clone().with_end_threshold(raw_end);
        assert!(
          explicit.end_threshold() > 0.0,
          "start {raw} / end {raw_end}: end threshold must be strictly positive"
        );
        assert!(
          explicit.end_threshold() <= explicit.start_threshold(),
          "start {raw} / end {raw_end}: end {} exceeds start {}",
          explicit.end_threshold(),
          explicit.start_threshold()
        );
      }
    }
  }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
  use std::time::Duration;

  use super::RunOptions;

  #[test]
  fn options_round_trip_under_the_neutral_field_names() {
    let options = RunOptions::default()
      .with_min_run_duration(Duration::from_millis(300))
      .with_min_gap_duration(Duration::from_millis(150))
      .with_min_gap_at_max_run(Duration::from_millis(90))
      .with_max_run_duration(Duration::from_millis(5_000))
      .with_pad(Duration::from_millis(40));

    let json = serde_json::to_string(&options).expect("serialize");
    assert!(json.contains("min_run_duration"), "{json}");
    assert!(json.contains("min_gap_duration"), "{json}");
    assert!(json.contains("min_gap_at_max_run"), "{json}");
    assert!(json.contains("max_run_duration"), "{json}");
    assert!(json.contains("\"pad\""), "{json}");

    let restored: RunOptions = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.min_run_duration(), Duration::from_millis(300));
    assert_eq!(restored.min_gap_duration(), Duration::from_millis(150));
    assert_eq!(restored.min_gap_at_max_run(), Duration::from_millis(90));
    assert_eq!(
      restored.max_run_duration(),
      Some(Duration::from_millis(5_000))
    );
    assert_eq!(restored.pad(), Duration::from_millis(40));
  }

  /// Deserialization must agree with the setters. `set_start_threshold` /
  /// `set_end_threshold` sanitize into `[0, 1]`, so a persisted profile
  /// carrying an out-of-range or non-finite threshold must not produce an
  /// options value the setter path could not have produced.
  ///
  /// The re-serialized form is compared, not just the accessors:
  /// `end_threshold()` sanitizes on read, so a raw stored value hides
  /// behind the accessor and only surfaces on the next round-trip.
  ///
  /// Mutation: drop the `deserialize_with` sanitizers -> `-1.0` lands in
  /// `start_threshold` verbatim, and `end_threshold` re-serializes as
  /// `-1.0` (or as `null`, for a non-finite value JSON cannot carry).
  /// Red.
  #[test]
  fn out_of_range_thresholds_are_sanitized_on_deserialize() {
    for (literal, raw) in [
      ("-1.0", -1.0_f32),
      ("5.0", 5.0),
      ("1e39", f32::INFINITY),
      ("-1e39", f32::NEG_INFINITY),
    ] {
      let json = format!(r#"{{"start_threshold": {literal}, "end_threshold": {literal}}}"#);
      let restored: RunOptions = serde_json::from_str(&json).expect("deserialize");
      let built = RunOptions::default()
        .with_start_threshold(raw)
        .with_end_threshold(raw);

      assert_eq!(
        restored.start_threshold(),
        built.start_threshold(),
        "{literal}: start threshold"
      );
      assert_eq!(
        serde_json::to_string(&restored).expect("re-serialize"),
        serde_json::to_string(&built).expect("serialize"),
        "{literal}: the stored fields must match the setter path"
      );
    }
  }

  /// Every field the serializer omits has to be optional on the way back
  /// in. `end_threshold` and `max_run_duration` are both
  /// `skip_serializing_if = "Option::is_none"` and both carry a
  /// `deserialize_with` (the threshold sanitizer and
  /// `humantime_serde::option`), which suppresses serde's implicit
  /// "a missing `Option` field is `None`" rule — so without an explicit
  /// `default` the DEFAULT options value does not survive its own
  /// round-trip.
  ///
  /// Mutation: drop either `default` -> `missing field` on deserialize.
  /// Red.
  #[test]
  fn default_options_survive_their_own_round_trip() {
    let defaults = RunOptions::default();
    let json = serde_json::to_string(&defaults).expect("serialize");
    assert!(!json.contains("end_threshold"), "{json}");
    assert!(!json.contains("max_run_duration"), "{json}");

    let restored: RunOptions = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.max_run_duration(), None);
    assert_eq!(restored.start_threshold(), defaults.start_threshold());
    assert_eq!(restored.end_threshold(), defaults.end_threshold());
    assert_eq!(restored.min_run_duration(), defaults.min_run_duration());
    assert_eq!(restored.min_gap_duration(), defaults.min_gap_duration());
    assert_eq!(restored.min_gap_at_max_run(), defaults.min_gap_at_max_run());
    assert_eq!(restored.pad(), defaults.pad());
  }

  #[test]
  fn legacy_speech_field_names_still_deserialize() {
    // A 0.1-era profile: every duration under its speech-flavoured name.
    // Mutation: drop the `alias` attributes -> `unknown field` -> red.
    let legacy = r#"{
      "sample_rate": "16k",
      "start_threshold": 0.5,
      "min_speech_duration": "300ms",
      "min_silence_duration": "150ms",
      "min_silence_at_max_speech": "90ms",
      "max_speech_duration": "5s",
      "speech_pad": "40ms"
    }"#;

    let restored: RunOptions = serde_json::from_str(legacy).expect("deserialize 0.1 profile");
    assert_eq!(restored.min_run_duration(), Duration::from_millis(300));
    assert_eq!(restored.min_gap_duration(), Duration::from_millis(150));
    assert_eq!(restored.min_gap_at_max_run(), Duration::from_millis(90));
    assert_eq!(
      restored.max_run_duration(),
      Some(Duration::from_millis(5_000))
    );
    assert_eq!(restored.pad(), Duration::from_millis(40));
  }

  /// The strictly-positive floor is a property of the stored value, not
  /// of the setter, so it has to hold on the deserialize path too — path
  /// parity is what the sanitizers exist for.
  ///
  /// Mutation (floor only in the setters): a persisted `0.0` installs a
  /// zero start threshold the setters cannot produce. Red.
  #[test]
  fn the_threshold_floor_applies_on_the_serde_path_too() {
    for (literal, raw) in [
      ("0.0", 0.0_f32),
      ("0.005", 0.005),
      ("-1.0", -1.0),
      ("-1e39", f32::NEG_INFINITY),
    ] {
      let json = format!(r#"{{"start_threshold": {literal}, "end_threshold": {literal}}}"#);
      let restored: RunOptions = serde_json::from_str(&json).expect("deserialize");
      let built = RunOptions::default()
        .with_start_threshold(raw)
        .with_end_threshold(raw);

      assert_eq!(
        restored.start_threshold(),
        RunOptions::MIN_THRESHOLD,
        "{literal}: start threshold is lifted to the floor"
      );
      assert_eq!(
        restored.start_threshold(),
        built.start_threshold(),
        "{literal}: start threshold path parity"
      );
      assert_eq!(
        serde_json::to_string(&restored).expect("re-serialize"),
        serde_json::to_string(&built).expect("serialize"),
        "{literal}: the stored fields must match the setter path"
      );
    }
  }
}
