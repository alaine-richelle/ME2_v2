import numpy as np

from me2.streaming import (
    PipelineState,
    StreamingConfig,
    StreamingPipeline,
    replay_chunks,
)
from me2.vcm.ctc import CommandResult


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
            np.ones(100, dtype=np.float32),
            *[np.ones(100, dtype=np.float32) for _ in range(8)],
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
