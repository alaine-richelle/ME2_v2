from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from me2 import app
from me2.streaming import PipelineState
from me2.vcm.ctc import CommandResult


def _wake_bundle(digest: str) -> dict:
    return {
        "audio": {
            "sample_rate": 1_000,
            "duration_seconds": 0.4,
            "silence_threshold_db": -48.0,
            "normalize_peak": True,
        },
        "labels": ["background", "ateko"],
        "threshold": 0.7,
        "streaming_policy": {"smoothing_window": 1, "required_hits": 1},
        "runtime_artifact_sha256": digest,
    }


def _vcm_bundle() -> dict:
    return {
        "audio": {
            "sample_rate": 1_000,
            "duration_seconds": 0.7,
            "silence_threshold_db": -48.0,
            "normalize_peak": True,
        },
    }


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        stream_chunk_ms=100,
        wake_hop_ms=100,
        pre_roll_ms=100,
        onset_timeout_ms=300,
        minimum_speech_ms=200,
        endpoint_silence_ms=200,
        maximum_command_ms=0,
        cooldown_ms=200,
        speech_rms_threshold=0.1,
        queue_chunks=4,
    )


def _accepted_command() -> CommandResult:
    return CommandResult(
        decoded_text="play music",
        intent="play_music",
        slots={},
        variation="",
        raw_log_score=-1.0,
        normalized_log_score=-0.1,
        candidate_posterior=0.9,
        margin=0.4,
        accepted=True,
        rejection_reason=None,
    )


def _pipeline(monkeypatch, primary, fallback, outcomes, commands):
    def fake_wake_result(model, bundle, waveform):
        del bundle, waveform
        outcome = outcomes[model]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(app, "_wake_result", fake_wake_result)
    return app.build_streaming_pipeline(
        _args(),
        primary,
        _wake_bundle("a" * 64) if primary is not None else None,
        object(),
        _vcm_bundle(),
        fallback_model=fallback,
        fallback_bundle=_wake_bundle("b" * 64) if fallback is not None else None,
        primary_load_error="OSError" if primary is None else None,
        on_wake=lambda label, score, elapsed, detected, status: commands.append(
            ("wake", label, score, elapsed, detected, status.copy())
        ),
    )


def test_v5_primary_controls_wake_when_v3_disagrees(monkeypatch):
    callbacks = []
    decoded = []
    pipeline = _pipeline(
        monkeypatch,
        "v5",
        "v3",
        {
            "v5": (False, "background", 0.1, 1.0),
            "v3": (True, "ateko", 0.9, 1.0),
        },
        callbacks,
    )
    pipeline.command_predictor = lambda waveform: decoded.append(waveform) or _accepted_command()

    events = pipeline.feed(np.ones(400, dtype=np.float32))

    assert events == []
    assert pipeline.state == PipelineState.LISTENING_FOR_WAKE
    assert decoded == []
    assert callbacks[-1][4] is False
    assert callbacks[-1][5]["mode"] == "primary"


def test_v5_primary_wake_is_used_with_v3_diagnostic_score(monkeypatch):
    callbacks = []
    pipeline = _pipeline(
        monkeypatch,
        "v5",
        "v3",
        {
            "v5": (True, "ateko", 0.9, 1.0),
            "v3": (False, "background", 0.1, 1.0),
        },
        callbacks,
    )

    events = pipeline.feed(np.ones(400, dtype=np.float32))

    assert [event.message for event in events] == ["wake_detected"]
    assert pipeline.state == PipelineState.CAPTURING_COMMAND
    assert callbacks[-1][1] == "ateko"
    assert callbacks[-1][5]["mode"] == "primary"


def test_v3_fallback_is_active_only_after_v5_inference_fault(monkeypatch):
    callbacks = []
    pipeline = _pipeline(
        monkeypatch,
        "v5",
        "v3",
        {
            "v5": RuntimeError("synthetic v5 fault"),
            "v3": (True, "ateko", 0.9, 1.0),
        },
        callbacks,
    )

    events = pipeline.feed(np.ones(400, dtype=np.float32))

    assert [event.message for event in events] == ["wake_detected"]
    assert callbacks[-1][5]["mode"] == "fallback_active"
    assert callbacks[-1][5]["primary_failure"] == "RuntimeError"


def test_both_wake_faults_fail_closed_without_command_capture(monkeypatch):
    callbacks = []
    decoded = []
    pipeline = _pipeline(
        monkeypatch,
        "v5",
        "v3",
        {
            "v5": RuntimeError("synthetic v5 fault"),
            "v3": RuntimeError("synthetic v3 fault"),
        },
        callbacks,
    )
    pipeline.command_predictor = lambda waveform: decoded.append(waveform) or _accepted_command()

    events = pipeline.feed(np.ones(400, dtype=np.float32))

    assert events == []
    assert pipeline.state == PipelineState.LISTENING_FOR_WAKE
    assert decoded == []
    assert callbacks[-1][5]["mode"] == "both_failed"


def test_v3_fallback_handles_v5_load_fault(monkeypatch, tmp_path):
    primary = tmp_path / "v5.me2"
    fallback = tmp_path / "v3.me2"
    fallback_bundle = _wake_bundle("b" * 64)

    def loader(path, task):
        assert task == "wake_word"
        if path == primary:
            raise OSError("missing primary")
        return "v3", fallback_bundle

    pair = app.load_wake_pair(primary, fallback, loader=loader)

    assert pair.primary_model is None
    assert pair.primary_load_error == "OSError"
    assert pair.fallback_model == "v3"
    assert pair.active_bundle is fallback_bundle
