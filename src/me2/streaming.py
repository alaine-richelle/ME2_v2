"""Deterministic continuous wake-to-command state machine and microphone runner."""

from __future__ import annotations

import enum
import logging
import queue
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

import numpy as np

from me2.audio import AudioDurationError
from me2.contracts import finite, integer
from me2.logging_utils import LogWriteError, log_event
from me2.vcm.ctc import CommandResult, DecoderBudgetError, InferenceInputError

LOGGER = logging.getLogger(__name__)


class PipelineState(str, enum.Enum):
    LISTENING_FOR_WAKE = "LISTENING_FOR_WAKE"
    CAPTURING_COMMAND = "CAPTURING_COMMAND"
    DECODING = "DECODING"
    ACCEPTED_OR_REJECTED = "ACCEPTED_OR_REJECTED"
    COOLDOWN = "COOLDOWN"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class StreamingConfig:
    sample_rate: int = 16_000
    chunk_samples: int = 1_600
    wake_window_samples: int = 24_000
    wake_hop_samples: int = 4_000
    pre_roll_samples: int = 4_000
    onset_timeout_samples: int = 16_000
    minimum_command_samples: int = 4_000
    endpoint_silence_samples: int = 8_000
    maximum_command_samples: int = 80_000
    cooldown_samples: int = 16_000
    speech_rms_threshold: float = 0.008
    queue_chunks: int = 32

    def __post_init__(self) -> None:
        positive = (
            self.sample_rate,
            self.chunk_samples,
            self.wake_window_samples,
            self.wake_hop_samples,
            self.onset_timeout_samples,
            self.minimum_command_samples,
            self.endpoint_silence_samples,
            self.maximum_command_samples,
            self.cooldown_samples,
            self.queue_chunks,
        )
        for value in positive:
            integer("stream sample count", value, 1, 12_000_000)
        integer("queue_chunks", self.queue_chunks, 1, 128)
        integer("pre_roll_samples", self.pre_roll_samples, 0,
                min(self.wake_window_samples, self.maximum_command_samples - 1))
        if self.minimum_command_samples >= self.maximum_command_samples:
            raise ValueError("minimum command duration must be below maximum duration")
        finite("speech_rms_threshold", self.speech_rms_threshold, 1e-6, 1.0)


@dataclass(frozen=True)
class PipelineEvent:
    state: PipelineState
    sample_index: int
    message: str
    result: CommandResult | None = None
    utterance_id: str | None = None
    diagnostics: dict = field(default_factory=dict)


WakePredictor = Callable[[np.ndarray], bool]
CommandPredictor = Callable[[np.ndarray], CommandResult]


class StreamingPipeline:
    """Consume timestamped chunks; all model work occurs outside audio callbacks."""

    def __init__(
        self,
        config: StreamingConfig,
        wake_predictor: WakePredictor,
        command_predictor: CommandPredictor,
        *,
        on_event: Callable[[PipelineEvent], None] | None = None,
    ) -> None:
        self.config = config
        self.wake_predictor = wake_predictor
        self.command_predictor = command_predictor
        self.on_event = on_event
        self.state = PipelineState.LISTENING_FOR_WAKE
        self.sample_index = 0
        self._wake_buffer: deque[float] = deque(maxlen=config.wake_window_samples)
        self._since_wake_check = 0
        self._command_chunks: list[np.ndarray] = []
        self._command_samples = 0
        self._speech_seen = False
        self._trailing_silence = 0
        self._cooldown_remaining = 0
        self._onset_wait_samples = 0
        self._speech_samples = 0
        self._speech_span_samples = 0
        self._wake_sample_index = 0
        self._speech_start_index: int | None = None
        self._pre_onset_dropped_samples = 0
        self._pre_onset_dropped_energy = 0.0
        self.utterance_id: str | None = None

    def feed(self, chunk: np.ndarray) -> list[PipelineEvent]:
        if self.state == PipelineState.STOPPED:
            return []
        samples = np.asarray(chunk, dtype=np.float32)
        if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
            raise ValueError("Streaming chunk must be finite non-empty mono audio")
        events: list[PipelineEvent] = []
        self.sample_index += samples.size

        if self.state == PipelineState.COOLDOWN:
            self._cooldown_remaining -= samples.size
            if self._cooldown_remaining <= 0:
                self.state = PipelineState.LISTENING_FOR_WAKE
                self._wake_buffer.clear()
                self._since_wake_check = 0
                events.append(
                    self._event(
                        self.state, self.sample_index, "cooldown_complete"
                    )
                )
            return events

        if self.state == PipelineState.LISTENING_FOR_WAKE:
            self._wake_buffer.extend(samples.tolist())
            self._since_wake_check += samples.size
            if (
                len(self._wake_buffer) == self.config.wake_window_samples
                and self._since_wake_check >= self.config.wake_hop_samples
            ):
                self._since_wake_check = 0
                window = np.fromiter(self._wake_buffer, dtype=np.float32)
                if self.wake_predictor(window):
                    self.utterance_id = uuid.uuid4().hex
                    pre_roll = window[-self.config.pre_roll_samples :] if self.config.pre_roll_samples else np.empty(0, dtype=np.float32)
                    self.state = PipelineState.CAPTURING_COMMAND
                    self._command_chunks = [pre_roll] if pre_roll.size else []
                    self._command_samples = pre_roll.size
                    self._speech_seen = False
                    self._trailing_silence = 0
                    self._onset_wait_samples = 0
                    self._speech_samples = 0
                    self._speech_span_samples = 0
                    self._wake_sample_index = self.sample_index
                    self._speech_start_index = None
                    self._pre_onset_dropped_samples = 0
                    self._pre_onset_dropped_energy = 0.0
                    events.append(
                        self._event(
                            self.state, self.sample_index, "wake_detected"
                        )
                    )
            return events

        if self.state == PipelineState.CAPTURING_COMMAND:
            rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
            active = rms >= self.config.speech_rms_threshold
            log_event(
                LOGGER, "capture_activity", utterance_id=self.utterance_id,
                sample_index=self.sample_index, sample_rate=self.config.sample_rate,
                diagnostics={
                    "chunk_rms": rms, "chunk_peak": float(np.max(np.abs(samples))),
                    "clipped_samples": int(np.count_nonzero(np.abs(samples) >= .999)),
                    "speech_active": active,
                },
            )
            if not self._speech_seen:
                # The onset clock starts at the wake event, never at the start
                # of pre-roll. Waiting keeps only a bounded rolling pre-roll.
                self._onset_wait_samples += samples.size
                if not active:
                    keep = self.config.pre_roll_samples
                    history = np.concatenate([*self._command_chunks, samples])
                    # Count dropped post-wake context separately from known
                    # pre-wake samples. Energy gating is not a word boundary.
                    post_wake_start = max(0, self._wake_sample_index - (self.sample_index - history.size))
                    dropped_end = max(0, history.size - keep)
                    dropped = history[post_wake_start:dropped_end]
                    self._pre_onset_dropped_samples += dropped.size
                    self._pre_onset_dropped_energy += float(np.sum(dropped.astype(np.float64) ** 2))
                    self._command_chunks = [history[-keep:].copy()] if keep else []
                    self._command_samples = min(history.size, keep)
                    if self._onset_wait_samples >= self.config.onset_timeout_samples:
                        events.extend(self._reject_capture("command_onset_timeout"))
                    return events
                self._speech_seen = True
                self._onset_wait_samples -= samples.size
                self._speech_start_index = self.sample_index - samples.size

            self._command_chunks.append(samples.copy())
            self._command_samples += samples.size
            if active:
                self._speech_samples += samples.size
                self._speech_span_samples = self.sample_index - self._speech_start_index
                self._trailing_silence = 0
            else:
                self._trailing_silence += samples.size

            # Duration includes internal pauses, but excludes pre-onset waiting
            # and the final silence used to confirm an endpoint. Keep all
            # samples, including pre-roll and quiet endings, for preprocessing.
            if self._speech_span_samples > self.config.maximum_command_samples:
                events.extend(self._reject_capture("command_too_long"))
            elif self._trailing_silence >= self.config.endpoint_silence_samples:
                if self._speech_samples < self.config.minimum_command_samples:
                    events.extend(self._reject_capture("command_too_short"))
                else:
                    events.extend(self._decode_capture())
        return events

    def stop(self) -> PipelineEvent:
        self.state = PipelineState.STOPPED
        self._command_chunks.clear()
        self._wake_buffer.clear()
        return self._event(self.state, self.sample_index, "stopped")

    def _event(
        self,
        state: PipelineState,
        sample_index: int,
        message: str,
        result: CommandResult | None = None,
    ) -> PipelineEvent:
        diagnostics = self.capture_diagnostics()
        diagnostics["capture_end_reason"] = message
        if message in {"stopped", "stream_recovered"}:
            diagnostics["capture_discarded"] = True
        event = PipelineEvent(state, sample_index, message, result, self.utterance_id, diagnostics)
        log_event(
            LOGGER, "pipeline_state",
            level=logging.WARNING if message in {
                "command_onset_timeout", "command_too_long", "command_too_short",
                "stream_recovered",
            } else logging.INFO,
            state=state.value, reason=message,
            sample_index=sample_index, utterance_id=self.utterance_id,
            sample_rate=self.config.sample_rate,
            system_error=result.system_error if result else None,
            diagnostics=diagnostics,
        )
        if self.on_event is not None:
            self.on_event(event)
        return event

    def capture_diagnostics(self) -> dict:
        scale = 1000 / self.config.sample_rate
        return {
            "wake_sample_index": self._wake_sample_index,
            "speech_start_sample_index": self._speech_start_index,
            "onset_timeout_ms": self.config.onset_timeout_samples * scale,
            "onset_wait_ms": self._onset_wait_samples * scale,
            "speech_observed": self._speech_seen,
            "speech_ms": self._speech_samples * scale,
            "speech_span_ms": self._speech_span_samples * scale,
            "captured_samples": self._command_samples,
            "pre_roll_ms": self.config.pre_roll_samples * scale,
            "endpoint_silence_ms": self.config.endpoint_silence_samples * scale,
            "trailing_silence_ms": self._trailing_silence * scale,
            "maximum_command_ms": self.config.maximum_command_samples * scale,
            "minimum_speech_ms": self.config.minimum_command_samples * scale,
            "capture_truncated": False,
            "onset_boundary_verified": False,
            "pre_onset_dropped_samples": self._pre_onset_dropped_samples,
            "pre_onset_dropped_rms": (
                self._pre_onset_dropped_energy / max(1, self._pre_onset_dropped_samples)
            ) ** .5,
        }

    def recover(self) -> PipelineEvent:
        """Invalidate partial audio and return to a clean wake-listening state."""

        self.state = PipelineState.LISTENING_FOR_WAKE
        self._wake_buffer.clear()
        self._since_wake_check = 0
        self._command_chunks.clear()
        self._command_samples = 0
        self._speech_seen = False
        self._trailing_silence = 0
        self._cooldown_remaining = 0
        self._onset_wait_samples = 0
        self._speech_samples = 0
        self._speech_span_samples = 0
        self._speech_start_index = None
        self._pre_onset_dropped_samples = 0
        self._pre_onset_dropped_energy = 0.0
        return self._event(self.state, self.sample_index, "stream_recovered")

    def _decode_capture(self) -> list[PipelineEvent]:
        self.state = PipelineState.DECODING
        decoding = self._event(self.state, self.sample_index, "endpoint_detected")
        waveform = np.concatenate(self._command_chunks)
        decision_started = time.perf_counter()
        try:
            result = self.command_predictor(waveform)
            decision_ms = (time.perf_counter() - decision_started) * 1000.0
        except AudioDurationError as error:
            return [decoding, *self._reject_capture("command_too_long", error.diagnostics)]
        except Exception as error:
            decision_ms = (time.perf_counter() - decision_started) * 1000.0
            LOGGER.exception("Command decoding failed; rejecting the captured utterance")
            result = CommandResult(
                decoded_text="",
                intent=None,
                slots={},
                variation="",
                raw_log_score=float("-inf"),
                normalized_log_score=float("-inf"),
                candidate_posterior=0.0,
                margin=float("-inf"),
                accepted=False,
                rejection_reason="command_decoder_error",
                system_error=type(error).__name__,
                diagnostics={
                    **getattr(error, "diagnostics", {}),
                    "command_decision_ms": decision_ms,
                    "diagnostic_category": (
                        "beam_pruning_exhaustion" if isinstance(error, DecoderBudgetError)
                        else error.category if isinstance(error, InferenceInputError)
                        else "unknown_system_fault"
                    ),
                    "next_action": "inspect_frame_counts_and_contracts",
                },
            )
        result = replace(result, diagnostics={
            **self.capture_diagnostics(), **result.diagnostics,
            "command_decision_ms": decision_ms,
            "capture_end_reason": "endpoint_detected",
        })
        self.state = PipelineState.ACCEPTED_OR_REJECTED
        outcome = self._event(
            self.state,
            self.sample_index,
            (
                "accepted"
                if result.accepted
                else result.rejection_reason or "rejected"
            ),
            result,
        )
        cooldown = self._begin_cooldown()
        return [decoding, outcome, cooldown]

    def _reject_capture(self, reason: str, diagnostics=None) -> list[PipelineEvent]:
        result = CommandResult(
            decoded_text="",
            intent=None,
            slots={},
            variation="",
            raw_log_score=float("-inf"),
            normalized_log_score=float("-inf"),
            candidate_posterior=0.0,
            margin=float("-inf"),
            accepted=False,
            rejection_reason=reason,
            diagnostics={
                **self.capture_diagnostics(),
                "diagnostic_category": reason,
                "capture_discarded": True,
                "capture_end_reason": reason,
                "next_action": {
                    "command_onset_timeout": "inspect_onset_wait_and_speech_level",
                    "command_too_long": "use_shorter_supported_phrase_or_validated_longer_artifact",
                    "command_too_short": "speak_a_complete_command",
                }.get(reason, "inspect_capture_events"),
                **(diagnostics or {}),
            },
        )
        self.state = PipelineState.ACCEPTED_OR_REJECTED
        outcome = self._event(self.state, self.sample_index, reason, result)
        return [outcome, self._begin_cooldown()]

    def _begin_cooldown(self) -> PipelineEvent:
        self.state = PipelineState.COOLDOWN
        self._cooldown_remaining = self.config.cooldown_samples
        self._command_chunks.clear()
        return self._event(self.state, self.sample_index, "cooldown_started")


def replay_chunks(
    pipeline: StreamingPipeline, chunks: Iterable[np.ndarray]
) -> list[PipelineEvent]:
    events: list[PipelineEvent] = []
    for chunk in chunks:
        events.extend(pipeline.feed(chunk))
    return events


def run_microphone_stream(
    pipeline: StreamingPipeline,
    *,
    input_device: str | int | None = None,
    audio_backend: str = "sounddevice",
    alsa_device: str = "default",
    on_event: Callable[[PipelineEvent], None] | None = None,
    stop_event: threading.Event | None = None,
    on_audio: Callable[[np.ndarray], None] | None = None,
) -> None:
    """Run one bounded continuous InputStream until interrupted."""
    try:
        _run_microphone_stream(
            pipeline, input_device=input_device, audio_backend=audio_backend,
            alsa_device=alsa_device, on_event=on_event,
            stop_event=stop_event, on_audio=on_audio,
        )
    finally:
        if pipeline.state != PipelineState.STOPPED:
            stopped = pipeline.stop()
            if on_event is not None:
                on_event(stopped)


class _ArecordInputStream:
    """Minimal bounded PCM source compatible with the stream callback contract."""

    def __init__(
        self,
        *,
        samplerate: int,
        channels: int,
        dtype: str,
        blocksize: int,
        device: str,
        callback,
    ) -> None:
        if channels != 1 or dtype != "float32":
            raise ValueError("arecord backend supports mono float32 callback audio only")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("ALSA device must be a non-empty string")
        integer("arecord sample rate", samplerate, 8_000, 192_000)
        integer("arecord block size", blocksize, 1, samplerate)
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.callback = callback
        self.latency = blocksize / samplerate
        self.command = [
            shutil.which("arecord") or "arecord",
            "-q", "-D", device, "-t", "raw", "-f", "S16_LE",
            "-r", str(samplerate), "-c", "1", "--period-size", str(blocksize), "-",
        ]
        self._stopping = threading.Event()
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self.failure_reason: str | None = None

    @property
    def active(self) -> bool:
        return bool(
            self._process is not None
            and self._process.poll() is None
            and self._reader is not None
            and self._reader.is_alive()
        )

    def __enter__(self):
        self._process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._reader = threading.Thread(
            target=self._read_chunks,
            name="me2-arecord-reader",
            daemon=True,
        )
        self._reader.start()
        return self

    def _read_exact(self, expected_bytes: int) -> bytes:
        if self._process is None or self._process.stdout is None:
            return b""
        data = bytearray()
        while len(data) < expected_bytes and not self._stopping.is_set():
            block = self._process.stdout.read(expected_bytes - len(data))
            if not block:
                break
            data.extend(block)
        return bytes(data)

    def _read_chunks(self) -> None:
        expected_bytes = self.blocksize * 2
        try:
            while not self._stopping.is_set():
                raw = self._read_exact(expected_bytes)
                if not raw:
                    if not self._stopping.is_set():
                        status = self._process.poll() if self._process is not None else None
                        self.failure_reason = f"ALSA capture ended (exit status {status})"
                    return
                if len(raw) != expected_bytes:
                    if not self._stopping.is_set():
                        self.failure_reason = "ALSA capture ended with an incomplete PCM chunk"
                    return
                samples = (
                    np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                )
                self.callback(samples.reshape(-1, 1), samples.size, None, False)
        except Exception as error:  # noqa: BLE001 -- reader-thread errors cross via stream status
            if not self._stopping.is_set():
                self.failure_reason = f"ALSA reader failed ({type(error).__name__})"

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_type, exc_value, traceback
        self._stopping.set()
        process = self._process
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
            if process.stdout is not None:
                process.stdout.close()
        if self._reader is not None:
            self._reader.join(timeout=2.0)
        return False


def _run_microphone_stream(
    pipeline: StreamingPipeline, *, input_device=None, audio_backend="sounddevice",
    alsa_device="default", on_event=None,
    stop_event=None, on_audio=None,
) -> None:
    if audio_backend == "arecord":
        stream_class = _ArecordInputStream
    elif audio_backend == "sounddevice":
        try:
            import sounddevice as sd
        except ImportError as error:
            raise RuntimeError(
                "Continuous microphone input requires the optional sounddevice dependency"
            ) from error
        stream_class = None
    else:
        raise ValueError("audio_backend must be 'sounddevice' or 'arecord'")

    chunks: queue.Queue = queue.Queue(
        maxsize=pipeline.config.queue_chunks
    )
    overflowed = False
    callback_status_count = 0
    queue_drop_count = 0
    callback_frames_total = 0
    callback_count = 0
    health_started = time.monotonic()
    health_last_published = health_started
    health_queue_delays: list[float] = []
    health_feed_times: list[float] = []
    health_queue_depths: list[int] = []
    health_chunks = 0

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(np.ceil((len(ordered) - 1) * fraction)))
        return float(ordered[index])

    def publish_stream_health(*, final: bool = False) -> None:
        nonlocal health_last_published, health_queue_delays
        nonlocal health_feed_times, health_queue_depths, health_chunks
        if health_chunks == 0 and not final:
            return
        log_event(
            LOGGER, "stream_health",
            diagnostics={
                "queue_delay_p95_ms": percentile(health_queue_delays, .95),
                "feed_p95_ms": percentile(health_feed_times, .95),
                "queue_depth_mean": (
                    float(np.mean(health_queue_depths)) if health_queue_depths else 0.0
                ),
                "queue_depth_max": max(health_queue_depths, default=0),
                "queue_depth_start": health_queue_depths[0] if health_queue_depths else 0,
                "queue_depth_end": health_queue_depths[-1] if health_queue_depths else 0,
                "chunks_consumed": health_chunks,
                "callback_frames_total": callback_frames_total,
                "callback_count": callback_count,
                "callback_status_count": callback_status_count,
                "queue_drop_count": queue_drop_count,
                "queue_capacity_chunks": pipeline.config.queue_chunks,
                "stream_age_seconds": time.monotonic() - health_started,
                "flush": final,
            },
        )
        health_last_published = time.monotonic()
        health_queue_delays = []
        health_feed_times = []
        health_queue_depths = []
        health_chunks = 0

    def callback(indata, frames, timing, status) -> None:
        del frames
        nonlocal overflowed, callback_status_count, queue_drop_count
        nonlocal callback_frames_total, callback_count
        callback_frames_total += int(indata.shape[0])
        callback_count += 1
        if status:
            overflowed = True
            callback_status_count += 1
        try:
            chunks.put_nowait((
                np.asarray(indata[:, 0], dtype=np.float32).copy(),
                time.monotonic(),
                getattr(timing, "inputBufferAdcTime", None),
            ))
        except queue.Full:
            overflowed = True
            queue_drop_count += 1

    stop_event = stop_event or threading.Event()
    retries = 0
    while not stop_event.is_set():
        try:
            if audio_backend == "arecord":
                stream_context = stream_class(
                    samplerate=pipeline.config.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=pipeline.config.chunk_samples,
                    device=alsa_device,
                    callback=callback,
                )
            else:
                stream_context = sd.InputStream(
                    samplerate=pipeline.config.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=pipeline.config.chunk_samples,
                    device=input_device,
                    callback=callback,
                )
            with stream_context as stream:
                log_event(
                    LOGGER, "microphone_stream_opened", sample_rate=pipeline.config.sample_rate,
                    input_device_index=input_device if isinstance(input_device, int) else None,
                    diagnostics={"channels": 1, "sample_format": "float32",
                                 "audio_backend": audio_backend,
                                 "alsa_device": alsa_device if audio_backend == "arecord" else None,
                                 "stream_latency_seconds": getattr(stream, "latency", None)},
                )
                while pipeline.state != PipelineState.STOPPED and not stop_event.is_set():
                    try:
                        item = chunks.get(timeout=1.0)
                    except queue.Empty:
                        if not stream.active:
                            raise RuntimeError(
                                getattr(stream, "failure_reason", None)
                                or "Microphone stream became inactive"
                            )
                        continue
                    if item is None:
                        publish_stream_health(final=True)
                        return
                    chunk, enqueued_at, adc_time = item
                    queue_depth = chunks.qsize()
                    queue_delay_ms = (time.monotonic() - enqueued_at) * 1000
                    log_event(
                        LOGGER, "stream_chunk",
                        level=logging.INFO if pipeline.state == PipelineState.CAPTURING_COMMAND else logging.DEBUG,
                        utterance_id=pipeline.utterance_id,
                        diagnostics={
                            "queue_delay_ms": queue_delay_ms,
                            "adc_time_seconds": adc_time,
                            "callback_status_count": callback_status_count,
                            "queue_drop_count": queue_drop_count,
                            "queue_depth_chunks": queue_depth,
                            "queue_capacity_chunks": pipeline.config.queue_chunks,
                            "callback_frames_total": callback_frames_total,
                            "callback_count": callback_count,
                        },
                    )
                    if overflowed:
                        overflowed = False
                        LOGGER.error(
                            "Microphone samples were lost; invalidating current utterance"
                        )
                        raise RuntimeError(
                            "Microphone overflow/disconnection invalidated the utterance"
                        )
                    if on_audio is not None:
                        on_audio(chunk)
                    feed_started = time.perf_counter()
                    for event in pipeline.feed(chunk):
                        if on_event is not None:
                            on_event(event)
                    health_queue_delays.append(queue_delay_ms)
                    health_feed_times.append((time.perf_counter() - feed_started) * 1000)
                    health_queue_depths.append(queue_depth)
                    health_chunks += 1
                    if time.monotonic() - health_last_published >= 5.0:
                        publish_stream_health()
            publish_stream_health(final=True)
            return
        except KeyboardInterrupt:
            pipeline.stop()
            publish_stream_health(final=True)
            return
        except LogWriteError:
            # Disk/diagnostic failure is not a recoverable microphone disconnect.
            raise
        except Exception as error:
            if stop_event.is_set():
                publish_stream_health(final=True)
                return
            publish_stream_health(final=True)
            retries += 1
            if retries > 3:
                publish_stream_health(final=True)
                pipeline.stop()
                raise RuntimeError(
                    f"Microphone recovery exhausted after {retries - 1} retries: {error}"
                ) from error
            # Discard old audio before reopening: it cannot belong to a new utterance.
            while True:
                try:
                    chunks.get_nowait()
                except queue.Empty:
                    break
            overflowed = False
            recovery = pipeline.recover()
            if on_event is not None:
                on_event(recovery)
            delay = min(2 ** (retries - 1), 4)
            log_event(
                LOGGER, "microphone_retry", level=logging.WARNING,
                retry=retries, backoff_seconds=delay, error_type=type(error).__name__,
                utterance_id=pipeline.utterance_id,
            )
            stop_event.wait(delay)
