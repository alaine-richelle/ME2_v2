import numpy as np
import pytest

from me2.streaming import (
    PipelineState,
    StreamingConfig,
    StreamingPipeline,
    replay_chunks,
)
from me2.vcm.ctc import CommandResult
from me2.wake_word.detector import WakeEventDetector, WakeSmoothingPolicy


def _config() -> StreamingConfig:
    return StreamingConfig(
        sample_rate=1_000,
        chunk_samples=100,
        wake_window_samples=400,
        wake_hop_samples=100,
        pre_roll_samples=100,
        onset_timeout_samples=300,
        minimum_command_samples=200,
        endpoint_silence_samples=200,
        maximum_command_samples=700,
        cooldown_samples=200,
        speech_rms_threshold=0.1,
        queue_chunks=4,
    )


def _accepted() -> CommandResult:
    return CommandResult(
        decoded_text="turn on the light",
        intent="light_on",
        slots={},
        variation="",
        raw_log_score=-1.0,
        normalized_log_score=-0.1,
        candidate_posterior=0.9,
        margin=0.4,
        accepted=True,
        rejection_reason=None,
    )


def test_continuous_replay_handles_chunk_boundary_wake_endpoint_and_cooldown():
    captured: list[np.ndarray] = []

    def command_predictor(waveform: np.ndarray) -> CommandResult:
        captured.append(waveform)
        return _accepted()

    pipeline = StreamingPipeline(
        _config(),
        lambda window: bool(np.max(window) > 0.5),
        command_predictor,
    )
    chunks = [
        np.zeros(100, dtype=np.float32),
        np.zeros(100, dtype=np.float32),
        np.zeros(100, dtype=np.float32),
        np.ones(100, dtype=np.float32),
        np.ones(100, dtype=np.float32) * 0.5,
        np.ones(100, dtype=np.float32) * 0.5,
        np.zeros(100, dtype=np.float32),
        np.zeros(100, dtype=np.float32),
        np.ones(100, dtype=np.float32),
        np.ones(100, dtype=np.float32),
    ]
    events = replay_chunks(pipeline, chunks)
    assert [event.state for event in events[:4]] == [
        PipelineState.CAPTURING_COMMAND,
        PipelineState.DECODING,
        PipelineState.ACCEPTED_OR_REJECTED,
        PipelineState.COOLDOWN,
    ]
    assert events[2].result and events[2].result.actionable_intent == "light_on"
    assert len(captured) == 1
    assert np.max(captured[0][:100]) == 1.0
    assert events[-1].state == PipelineState.LISTENING_FOR_WAKE


def test_continuous_replay_rejects_command_onset_timeout():
    pipeline = StreamingPipeline(
        _config(),
        lambda window: bool(np.max(window) > 0.5),
        lambda waveform: _accepted(),
    )
    events = replay_chunks(
        pipeline,
        [
            np.zeros(100, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
            np.ones(100, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
        ],
    )
    outcomes = [
        event
        for event in events
        if event.state == PipelineState.ACCEPTED_OR_REJECTED
    ]
    assert outcomes[0].result
    assert outcomes[0].result.rejection_reason == "command_onset_timeout"


def test_continuous_replay_rejects_overlong_command_without_decoding():
    decoded = False

    def command_predictor(waveform: np.ndarray) -> CommandResult:
        nonlocal decoded
        decoded = True
        return _accepted()

    pipeline = StreamingPipeline(
        _config(),
        lambda window: bool(np.max(window) > 0.5),
        command_predictor,
    )
    events = replay_chunks(
        pipeline,
        [
            np.zeros(100, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
            np.ones(100, dtype=np.float32) * .6,
            *[np.ones(100, dtype=np.float32) * .5 for _ in range(8)],
        ],
    )
    outcomes = [
        event
        for event in events
        if event.state == PipelineState.ACCEPTED_OR_REJECTED
    ]
    assert outcomes[0].result
    assert outcomes[0].result.rejection_reason == "command_too_long"
    assert not decoded


def test_recovery_invalidates_partial_capture_and_returns_to_wake():
    pipeline = StreamingPipeline(
        _config(),
        lambda window: True,
        lambda waveform: _accepted(),
    )
    pipeline.feed(np.ones(400, dtype=np.float32))
    assert pipeline.state == PipelineState.CAPTURING_COMMAND
    pipeline.feed(np.ones(100, dtype=np.float32))

    event = pipeline.recover()

    assert event.message == "stream_recovered"
    assert pipeline.state == PipelineState.LISTENING_FOR_WAKE


def test_decoder_failure_becomes_non_actionable_rejection_and_cooldown():
    def failed_decoder(waveform):
        raise RuntimeError("synthetic decoder failure")

    pipeline = StreamingPipeline(
        _config(),
        lambda waveform: True,
        failed_decoder,
    )
    pipeline.feed(np.ones(400, dtype=np.float32))
    pipeline.feed(np.ones(100, dtype=np.float32) * 0.5)
    pipeline.feed(np.ones(100, dtype=np.float32) * 0.5)
    pipeline.feed(np.zeros(100, dtype=np.float32))
    events = pipeline.feed(np.zeros(100, dtype=np.float32))
    result = next(event.result for event in events if event.result is not None)
    assert result.accepted is False
    assert result.actionable_intent is None
    assert result.rejection_reason == "command_decoder_error"
    assert pipeline.state == PipelineState.COOLDOWN
    assert pipeline.feed(np.zeros(100, dtype=np.float32)) == []


def test_wake_persistence_diagnostics_explain_non_trigger_then_trigger():
    detector = WakeEventDetector(WakeSmoothingPolicy(.7, smoothing_window=3, required_hits=2))
    assert not detector.update("ateko", .93)
    pending = detector.diagnostics()
    assert pending["wake_nontrigger_reason"] == "wake_persistence_pending"
    assert pending["wake_effective_threshold"] == .7
    assert pending["wake_observed_hits"] == 1
    assert detector.update("ateko", .8)
    detected = detector.diagnostics()
    assert detected["wake_nontrigger_reason"] == "wake_persistence_satisfied"
    assert detected["wake_persistence_rule"] == "n_of_m"


def test_capture_rejects_excessive_clipping_with_aggregate_diagnostics():
    decoded = False

    def predict(_waveform):
        nonlocal decoded
        decoded = True
        return _accepted()

    config = StreamingConfig(**{
        **_config().__dict__, "clipping_warning_fraction": .02,
        "clipping_reject_fraction": .05,
    })
    pipeline = StreamingPipeline(config, lambda _window: True, predict)
    pipeline.feed(np.ones(400, dtype=np.float32))
    events = pipeline.feed(np.ones(100, dtype=np.float32))
    rejected = next(event.result for event in events if event.result is not None)
    assert rejected.rejection_reason == "command_clipping_excessive"
    assert rejected.diagnostics["capture_clipped_samples"] == 100
    assert rejected.diagnostics["clipping_fraction"] == 1.0
    assert rejected.diagnostics["capture_quality_warning"]
    assert not decoded


def test_adaptive_noise_floor_hysteresis_ends_clean_short_command():
    captured = []
    config = StreamingConfig(**{
        **_config().__dict__, "noise_floor_calibration_samples": 100,
        "noise_floor_multiplier": 2.5, "speech_hysteresis_ratio": .7,
    })
    pipeline = StreamingPipeline(
        config, lambda _window: True,
        lambda waveform: captured.append(waveform) or _accepted(),
    )
    pipeline.feed(np.ones(400, dtype=np.float32))
    events = replay_chunks(pipeline, [
        np.full(100, .03, dtype=np.float32),
        np.full(100, .2, dtype=np.float32),
        np.full(100, .2, dtype=np.float32),
        np.full(100, .06, dtype=np.float32),
        np.full(100, .06, dtype=np.float32),
    ])
    accepted = next(event.result for event in events if event.result is not None)
    assert accepted.accepted
    assert accepted.diagnostics["noise_floor_rms"] == pytest.approx(.03)
    assert accepted.diagnostics["speech_hysteresis_ratio"] == .7
    assert len(captured) == 1
