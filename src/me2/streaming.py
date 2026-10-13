"""Deterministic continuous wake-to-command state machine and microphone runner."""

from __future__ import annotations

import enum
import logging
import queue
import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from me2.vcm.ctc import CommandResult

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
        if any(value <= 0 for value in positive):
            raise ValueError("Streaming sample counts and queue size must be positive")
        if self.pre_roll_samples < 0:
            raise ValueError("pre_roll_samples cannot be negative")
        if self.minimum_command_samples >= self.maximum_command_samples:
            raise ValueError("minimum command duration must be below maximum duration")
        if self.speech_rms_threshold < 0:
            raise ValueError("speech_rms_threshold cannot be negative")


@dataclass(frozen=True)
class PipelineEvent:
    state: PipelineState
    sample_index: int
    message: str
    result: CommandResult | None = None


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

    def feed(self, chunk: np.ndarray) -> list[PipelineEvent]:
        samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if samples.size == 0 or not np.isfinite(samples).all():
            raise ValueError("Streaming chunk is empty or non-finite")
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
                    pre_roll = window[-self.config.pre_roll_samples :] if self.config.pre_roll_samples else np.empty(0, dtype=np.float32)
                    self.state = PipelineState.CAPTURING_COMMAND
                    self._command_chunks = [pre_roll] if pre_roll.size else []
                    self._command_samples = pre_roll.size
                    self._speech_seen = False
                    self._trailing_silence = 0
                    events.append(
                        self._event(
                            self.state, self.sample_index, "wake_detected"
                        )
                    )
            return events

        if self.state == PipelineState.CAPTURING_COMMAND:
            self._command_chunks.append(samples.copy())
            self._command_samples += samples.size
            rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
            if rms >= self.config.speech_rms_threshold:
                self._speech_seen = True
                self._trailing_silence = 0
            elif self._speech_seen:
                self._trailing_silence += samples.size

            if (
                not self._speech_seen
                and self._command_samples >= self.config.onset_timeout_samples
            ):
                events.extend(self._reject_capture("command_onset_timeout"))
            elif self._command_samples >= self.config.maximum_command_samples:
                events.extend(self._reject_capture("command_too_long"))
            elif (
                self._speech_seen
                and self._command_samples >= self.config.minimum_command_samples
                and self._trailing_silence >= self.config.endpoint_silence_samples
            ):
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
        event = PipelineEvent(state, sample_index, message, result)
        if self.on_event is not None:
            self.on_event(event)
        return event

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
        return self._event(self.state, self.sample_index, "stream_recovered")

    def _decode_capture(self) -> list[PipelineEvent]:
        self.state = PipelineState.DECODING
        decoding = self._event(self.state, self.sample_index, "endpoint_detected")
        waveform = np.concatenate(self._command_chunks)
        try:
            result = self.command_predictor(waveform)
        except Exception:
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
            )
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

    def _reject_capture(self, reason: str) -> list[PipelineEvent]:
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
    on_event: Callable[[PipelineEvent], None] | None = None,
    stop_event: threading.Event | None = None,
    on_audio: Callable[[np.ndarray], None] | None = None,
) -> None:
    """Run one bounded continuous InputStream until interrupted."""

    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError(
            "Continuous microphone input requires the optional sounddevice dependency"
        ) from error

    chunks: queue.Queue[np.ndarray | None] = queue.Queue(
        maxsize=pipeline.config.queue_chunks
    )
    overflowed = False

    def callback(indata, frames, timing, status) -> None:
        del frames, timing
        nonlocal overflowed
        if status:
            overflowed = True
        try:
            chunks.put_nowait(np.asarray(indata[:, 0], dtype=np.float32).copy())
        except queue.Full:
            overflowed = True

    stop_event = stop_event or threading.Event()
    retries = 0
    while not stop_event.is_set():
        try:
            with sd.InputStream(
                samplerate=pipeline.config.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=pipeline.config.chunk_samples,
                device=input_device,
                callback=callback,
            ) as stream:
                while pipeline.state != PipelineState.STOPPED and not stop_event.is_set():
                    try:
                        chunk = chunks.get(timeout=1.0)
                    except queue.Empty:
                        if not stream.active:
                            raise RuntimeError("Microphone stream became inactive")
                        continue
                    if chunk is None:
                        return
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
                    for event in pipeline.feed(chunk):
                        if on_event is not None:
                            on_event(event)
            return
        except KeyboardInterrupt:
            pipeline.stop()
            return
        except Exception as error:
            if stop_event.is_set():
                return
            retries += 1
            if retries > 3:
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
            LOGGER.warning(
                "Recoverable microphone fault (%s); retrying in %ss", error, delay
            )
            stop_event.wait(delay)
