use crate::options::SampleRate;

/// A voice-activity-detection model backend: the push-based contract the
/// detector drives to turn audio into speech probabilities.
///
/// A backend owns its own recurrent state, rolling context, and input
/// windowing. It accepts PCM through [`push`](VadBackend::push) and emits
/// zero or more speech probabilities per call by invoking a `sink`
/// callback — one probability per model frame the backend completes. The
/// backend-agnostic detection logic — [`SpeechSegmenter`], the
/// [`detect_speech_with`] one-shot helper — drives a backend through this
/// trait and never touches the underlying model, so the same segmentation
/// semantics work over any backend implementation: an ONNX Silero backend
/// whose analysis window equals its hop (512 samples at 16 kHz), a backend
/// with overlapping analysis windows (a 400-sample window advanced by a
/// 160-sample hop), or a purely synthetic one.
///
/// # Geometry
///
/// Input windowing is the backend's private business: one
/// [`push`](VadBackend::push) may complete zero, one, or many frames, and
/// successive analysis windows may overlap. The detector only needs
/// [`frame_hop`](VadBackend::frame_hop) — the number of samples one emitted
/// probability advances the timeline. For a backend whose window equals its
/// hop this is just the frame size (`512` for Silero at 16 kHz); for an
/// overlapping-window backend it is the hop (`160`), not the window
/// (`400`). It is decoupled from [`SampleRate::chunk_samples`]: the Silero
/// geometry declares `512` at 16 kHz (identical to `chunk_samples`), while
/// another artifact declares whatever hop its frames advance by and reuses
/// the same detector unchanged.
///
/// # End of stream
///
/// [`finish`](VadBackend::finish) marks end-of-stream and lets the backend
/// apply its own trailing-frame policy: a backend that zero-pads its last
/// partial frame emits one final probability; a backend that drops an
/// incomplete trailing window (`snip_edges`) emits nothing. The detector
/// imposes neither policy — it drives [`push`](VadBackend::push) over the
/// input, then [`finish`](VadBackend::finish), and segments whatever
/// probabilities the backend chose to emit.
///
/// [`SpeechSegmenter`]: crate::SpeechSegmenter
/// [`detect_speech_with`]: crate::detect_speech_with
pub trait VadBackend {
  /// The backend's own error type, bridged into [`crate::Error`].
  ///
  /// The detector converts a backend error into [`crate::Error`] via this
  /// bound. A backend either sets this to [`crate::Error`] itself
  /// (constructing [`crate::Error`] values directly — an identity
  /// conversion) or defines its own error type and provides
  /// `impl From<TheirError> for crate::Error`, wrapping it in the
  /// transparent [`crate::Error::Backend`] variant.
  type Error: Into<crate::Error>;

  /// The number of PCM samples one emitted probability advances the
  /// detector's timeline: the frame hop.
  ///
  /// Must be non-zero. The detector advances its sample timeline by this
  /// amount per probability the backend emits and converts the segmenter's
  /// time-based options against it. For a backend whose analysis window
  /// equals its hop this is the frame size; for an overlapping-window
  /// backend it is the hop, not the window.
  fn frame_hop(&self) -> usize;

  /// The sample rate the backend expects its input PCM to be sampled at.
  ///
  /// Used to convert the segmenter's time-based options (durations) into
  /// sample counts and to stamp emitted [`SpeechSegment`]s.
  ///
  /// [`SpeechSegment`]: crate::SpeechSegment
  fn sample_rate(&self) -> SampleRate;

  /// Feed PCM into the stream, invoking `sink` once per completed model
  /// frame with that frame's speech probability in `[0, 1]`.
  ///
  /// `[0, 1]` is the backend's obligation, and the segmenter enforces it
  /// rather than trusting it: a probability outside the range is
  /// canonicalized on the way in — `NaN` to `0.0`, everything else clamped
  /// — so a malformed frame degrades to silence (or to full confidence)
  /// instead of poisoning a [`SpeechSegment`](crate::SpeechSegment)'s
  /// aggregates. See
  /// [`RunSegmenter::push_probability`](crate::RunSegmenter::push_probability)
  /// for the exact policy. That repair is a backstop, not a licence: emit
  /// in-range values.
  ///
  /// The backend buffers whatever trailing PCM does not yet complete a
  /// frame and consumes it on later calls, advancing its own recurrent
  /// state and rolling context, so successive `push` calls form a single
  /// logical stream until [`reset`](VadBackend::reset). A single call may
  /// invoke `sink` zero, one, or many times depending on how many frames
  /// the newly available PCM completes — this is what lets a backend delay
  /// its first output until a full window exists and emit several
  /// probabilities from one overlapping-window buffer.
  ///
  /// # Errors
  ///
  /// Returns [`Self::Error`] if the underlying model fails to run. Any
  /// probabilities emitted earlier in the same call have already been
  /// passed to `sink`.
  fn push(&mut self, samples: &[f32], sink: &mut dyn FnMut(f32)) -> Result<(), Self::Error>;

  /// Mark end-of-stream, applying the backend's own trailing-frame policy.
  ///
  /// A backend that zero-pads its last partial frame invokes `sink` once
  /// more with the padded frame's probability; a backend that drops an
  /// incomplete trailing window (`snip_edges=true`) does not invoke `sink`
  /// at all. After `finish`, call [`reset`](VadBackend::reset) before
  /// reusing the backend for a new stream.
  ///
  /// # Errors
  ///
  /// Returns [`Self::Error`] if flushing the trailing frame runs the model
  /// and that inference fails.
  fn finish(&mut self, sink: &mut dyn FnMut(f32)) -> Result<(), Self::Error>;

  /// Clear the backend's recurrent state, rolling context, and any
  /// buffered partial-frame PCM so the next [`push`](VadBackend::push)
  /// starts a fresh logical stream.
  fn reset(&mut self);
}
