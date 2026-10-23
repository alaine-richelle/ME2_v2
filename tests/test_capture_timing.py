"""Capture clocks and decoder diagnostics; no devices or retained recordings."""

import io
import json
import logging
import subprocess
import sys
import time
from dataclasses import replace

import numpy as np
import pytest
import torch

from me2.app import build_streaming_pipeline, parse_args
from me2.audio import AudioConfig, AudioDurationError, prepare_audio_sequence
from me2.bundle import predict_command
from me2.features import FeatureConfig, LogMel
from me2.logging_utils import _JsonLogFormatter, log_event
from me2.streaming import PipelineState, StreamingConfig, StreamingPipeline, replay_chunks
from me2.vcm.ctc import (
    CHARSET,
    CommandResult,
    CompactCTCModel,
    DecoderBudgetError,
    DecoderLimits,
    GrammarEntry,
    InferenceInputError,
    decode_command,
    inspect_phrase_support,
)


def result():
    return CommandResult("play music", "play_music", {}, "", 0, 0, 1, 0, True, None)


def fixture(**changes):
    config = replace(StreamingConfig(
        sample_rate=1000, chunk_samples=100, wake_window_samples=400,
        wake_hop_samples=100, pre_roll_samples=250, onset_timeout_samples=3000,
        minimum_command_samples=250, endpoint_silence_samples=800,
        maximum_command_samples=5000, cooldown_samples=1000,
    ), **changes)
    captures = []
    def predict(waveform):
        captures.append(waveform.copy())
        return result()
    once = [True]
    pipeline = StreamingPipeline(config, lambda _: once.pop() if once else False, predict)
    pipeline.feed(np.zeros(400, dtype=np.float32))
    return pipeline, captures


def chunks(ms, level=0.0):
    assert ms % 100 == 0
    return [np.full(100, level, dtype=np.float32) for _ in range(ms // 100)]


@pytest.mark.parametrize("pause", [0, 1000, 1500, 2000, 2900])
def test_pause_clock_does_not_consume_speech_budget(pause):
    p, captured = fixture()
    events = replay_chunks(p, chunks(pause) + chunks(4600, .1) + chunks(800))
    accepted = next(e.result for e in events if e.result is not None)
    assert accepted.accepted
    assert accepted.diagnostics["onset_wait_ms"] == pause
    assert accepted.diagnostics["speech_span_ms"] == 4600
    assert len(captured) == 1
    assert np.count_nonzero(captured[0]) == 4600
    assert captured[0].size == 250 + 4600 + 800


def test_default_five_second_cap_accepts_exact_limit_then_endpoint_silence():
    p, captured = fixture()
    events = replay_chunks(p, chunks(5000, .1) + chunks(800))
    accepted = next(event.result for event in events if event.result is not None)
    assert accepted.accepted
    assert accepted.diagnostics["maximum_command_ms"] == 5000
    assert accepted.diagnostics["speech_span_ms"] == 5000
    assert len(captured) == 1


def test_longer_cap_requires_explicit_streaming_configuration():
    p, captured = fixture(maximum_command_samples=6000)
    events = replay_chunks(p, chunks(5100, .1) + chunks(800))
    accepted = next(event.result for event in events if event.result is not None)
    assert accepted.accepted
    assert accepted.diagnostics["maximum_command_ms"] == 6000
    assert accepted.diagnostics["speech_span_ms"] == 5100
    assert len(captured) == 1


def test_onset_timeout_exactly_after_wake_and_wait_buffer_is_bounded():
    p, captures = fixture()
    assert replay_chunks(p, chunks(2900)) == []
    assert sum(c.size for c in p._command_chunks) == 250
    events = p.feed(np.zeros(100, dtype=np.float32))
    rejected = events[0].result
    assert rejected.rejection_reason == "command_onset_timeout"
    assert rejected.diagnostics["onset_wait_ms"] == 3000
    assert not rejected.diagnostics["speech_observed"]
    assert rejected.actionable_intent is None and not captures


def test_quiet_pre_onset_context_loss_is_reported_not_claimed_verified():
    p, _ = fixture()
    replay_chunks(p, chunks(1500, .004) + chunks(300, .1))
    info = p.capture_diagnostics()
    assert info["pre_onset_dropped_samples"] == 1250
    assert info["pre_onset_dropped_rms"] == pytest.approx(.004)
    assert info["onset_boundary_verified"] is False


def test_immediate_handoff_preserves_every_preroll_sample():
    p, captures = fixture()
    # The tail can contain early command audio: do not replace it with silence.
    marker = np.linspace(.01, .1, 250, dtype=np.float32)
    p._command_chunks = [marker.copy()]
    replay_chunks(p, chunks(600, .1) + chunks(800))
    np.testing.assert_array_equal(captures[0][:250], marker)


def test_endpoint_keeps_quiet_ending_and_allows_internal_pause():
    p, captures = fixture()
    replay_chunks(p, chunks(500, .1) + chunks(600) + chunks(500, .1))
    assert not captures
    replay_chunks(p, chunks(300, .003) + chunks(500))
    assert len(captures) == 1
    # Quiet final sounds below the energy threshold are still retained.
    assert np.count_nonzero(captures[0] == np.float32(.003)) == 300


def test_minimum_speech_counts_speech_not_padding_or_silence():
    p, captures = fixture()
    events = replay_chunks(p, chunks(2000) + chunks(100, .1) + chunks(800))
    rejected = next(e.result for e in events if e.result is not None)
    assert rejected.rejection_reason == "command_too_short"
    assert rejected.actionable_intent is None
    assert not captures


def test_five_second_speech_can_finish_endpoint_but_over_cap_is_discarded():
    p, captures = fixture()
    replay_chunks(p, chunks(5000, .1) + chunks(800))
    assert len(captures) == 1 and np.count_nonzero(captures[0]) == 5000
    p, captures = fixture()
    events = replay_chunks(p, chunks(5100, .1))
    rejected = next(e.result for e in events if e.result is not None)
    assert rejected.rejection_reason == "command_too_long"
    assert rejected.diagnostics["capture_discarded"]
    assert not rejected.diagnostics["capture_truncated"]
    assert not captures


def test_preroll_or_quiet_ending_cannot_bypass_artifact_audio_limit():
    p, _ = fixture()
    def predict(_):
        raise AudioDurationError(81000, 80000)
    p.command_predictor = predict
    events = replay_chunks(p, chunks(300, .1) + chunks(800))
    rejected = next(e.result for e in events if e.result is not None)
    assert rejected.rejection_reason == "command_too_long"
    assert rejected.system_error is None
    assert rejected.diagnostics["usable_samples"] == 81000
    assert rejected.actionable_intent is None


def test_budget_error_keeps_category_and_never_dispatches():
    p, _ = fixture()
    def predict(_):
        raise DecoderBudgetError("decoder_work_budget_exhausted", {"valid_output_frames": 80})
    p.command_predictor = predict
    events = replay_chunks(p, chunks(300, .1) + chunks(800))
    rejected = next(e.result for e in events if e.result is not None)
    assert rejected.actionable_intent is None
    assert rejected.diagnostics["diagnostic_category"] == "beam_pruning_exhaustion"
    assert rejected.diagnostics["valid_output_frames"] == 80
    p.stop()
    assert p.state == PipelineState.STOPPED and not p._command_chunks


def logits(ids, symbols):
    values = np.full((len(ids), symbols), -20.0)
    values[np.arange(len(ids)), ids] = 20
    return values


def test_no_terminal_categories_are_evidence_based():
    entries = [GrammarEntry("abc", "long")]
    short = decode_command(logits([0, 1], 4), entries, charset="abc", beam_width=1)
    assert short.diagnostics["diagnostic_category"] == "insufficient_valid_frames"
    prefix = decode_command(logits([0, 1, 3, 3], 4), entries, charset="abc", beam_width=1)
    assert prefix.diagnostics["diagnostic_category"] == "legal_prefix_no_terminal"
    assert prefix.diagnostics["beam_pruning_observed"]
    assert prefix.diagnostics["phrase_support"] == "unknown_without_reference"
    blank = decode_command(logits([3] * 4, 4), entries, charset="abc", beam_width=1)
    assert blank.diagnostics["diagnostic_category"] == "no_legal_trie_path"
    assert all(not r.accepted and r.actionable_intent is None for r in [short, prefix, blank])


def test_long_repeated_terminal_consumes_all_valid_frames_and_excludes_padding():
    text = "abba" * 15
    charset = "ab"
    path = []
    previous = None
    for ch in text:
        if ch == previous:
            path.append(2)
        path.append(charset.index(ch))
        previous = ch
    values = logits(path, 3)
    padded = np.concatenate([values, np.full((10, 3), np.nan)])
    decoded = decode_command(
        padded, [GrammarEntry("abba", "short"), GrammarEntry(text, "long")],
        charset=charset, input_length=len(path), beam_width=128,
    )
    assert decoded.accepted and decoded.intent == "long"
    assert decoded.diagnostics["valid_output_frames"] == len(path)
    with pytest.raises(InferenceInputError, match="non-finite"):
        decode_command(padded, [GrammarEntry(text, "long")], charset=charset)
    with pytest.raises(DecoderBudgetError) as error:
        decode_command(values, [GrammarEntry(text, "long")], charset=charset,
                       limits=DecoderLimits(max_expansions=1))
    assert error.value.diagnostics["budget_reason"] == "decoder_work_budget_exhausted"


def test_expected_phrase_support_does_not_change_grammar_or_decode():
    entries = [GrammarEntry("play music", "play_music")]
    assert inspect_phrase_support("set a timer", entries)["reference_diagnostic"] == "unsupported_or_normalization_mismatched_phrase"
    assert inspect_phrase_support("play music", entries, output_frames=2)["expected_alignment_feasible"] is False
    assert len(entries) == 1


def test_wake_window_telemetry_includes_frontend_and_preserves_model_timing(monkeypatch):
    from me2 import app

    args = parse_args([
        "--interactive", "--stream-chunk-ms", "5", "--wake-hop-ms", "5",
        "--pre-roll-ms", "0",
    ])
    audio = {"sample_rate": 16_000, "duration_seconds": .02}
    wake_bundle = {
        "audio": audio, "threshold": .7,
        "streaming_policy": {"smoothing_window": 1, "required_hits": 1},
    }
    vcm_bundle = {"audio": {**audio, "duration_seconds": 5.0}}
    logged = []
    callback = []

    def slow_frontend(samples, sample_rate, config):
        time.sleep(.002)
        return samples

    monkeypatch.setattr(app, "prepare_audio", slow_frontend)
    monkeypatch.setattr(app, "_wake_result", lambda *args: (False, "non_wake", .1, .25))
    monkeypatch.setattr(
        app, "log_event",
        lambda logger, event, **fields: logged.append((event, fields)),
    )
    pipeline = build_streaming_pipeline(
        args, object(), wake_bundle, object(), vcm_bundle,
        on_wake=lambda *values: callback.append(values),
    )
    for _ in range(4):
        pipeline.feed(np.zeros(80, dtype=np.float32))

    wake_event = next(fields for event, fields in logged if event == "wake_window")
    diagnostics = wake_event["diagnostics"]
    assert diagnostics["frontend_ms"] >= 1.0
    assert diagnostics["wake_model_ms"] == .25
    assert diagnostics["wake_pipeline_ms"] == wake_event["total_ms"]
    assert wake_event["total_ms"] >= diagnostics["frontend_ms"]
    assert diagnostics["wake_effective_threshold"] == .7
    assert diagnostics["wake_persistence_rule"] == "n_of_m"
    assert diagnostics["wake_nontrigger_reason"] == "wake_predicted_non_ateko"
    assert diagnostics["wake_debounce_state"] == "listening"
    assert callback[0][2] == .25  # Dashboard's historical field remains model inference time.


def test_runtime_and_session_config_validation():
    args = parse_args(["--interactive", "--session-seconds", "20"])
    assert args.maximum_command_ms == 0 and args.onset_timeout_ms == 1000
    assert args.session_seconds == 20
    for value in ["nan", "-1", "3601"]:
        with pytest.raises(SystemExit):
            parse_args(["--interactive", "--session-seconds", value])
    args = parse_args(["--interactive", "--maximum-command-ms", "6000"])
    audio = {"sample_rate": 16000, "duration_seconds": 5}
    with pytest.raises(ValueError, match="artifact duration"):
        build_streaming_pipeline(args, None, {"audio": audio}, None, {"audio": audio})


def test_bundle_reports_real_feature_and_output_lengths():
    torch.manual_seed(42)
    model = CompactCTCModel(width=4, projection_size=8, hidden_size=8).eval()
    config = FeatureConfig()
    bundle = {
        "architecture": "CompactCTCModel", "audio": {"sample_rate": 16000, "duration_seconds": .1},
        "features": config.__dict__, "charset": CHARSET, "blank_id": len(CHARSET),
        "grammar": [["a", "a"]], "acceptance_policy": {},
    }
    waveform, usable = prepare_audio_sequence(np.ones(800, np.float32)*.1, 16000, AudioConfig(duration_seconds=.1))
    output = predict_command(model, bundle, waveform, sample_length=usable)
    features = int(LogMel(config).frame_lengths(torch.tensor([usable]))[0])
    assert output.diagnostics["valid_feature_frames"] == features
    assert output.diagnostics["valid_output_frames"] == int(model.output_lengths(torch.tensor([features]))[0])
    assert output.diagnostics["padded_samples"] == 1600


def test_default_diagnostic_log_excludes_transcripts_and_arbitrary_keys():
    sink = io.StringIO()
    handler = logging.StreamHandler(sink)
    handler.setFormatter(_JsonLogFormatter())
    logger = logging.getLogger("capture-diagnostic-test")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        log_event(logger, "command_result", diagnostics={
            "decoded_text": "secret", "contact": "private",
            "valid_output_frames": 126, "diagnostic_category": "legal_prefix_no_terminal",
        }, accepted_intent="play_music")
    finally:
        logger.removeHandler(handler)
        handler.close()
    record = json.loads(sink.getvalue())
    fields = record["diagnostics"]
    assert fields == {"valid_output_frames": 126, "diagnostic_category": "legal_prefix_no_terminal"}
    assert record["accepted_intent"] == "play_music"
    assert "decoded_text" not in sink.getvalue()


def test_continuous_cli_installs_and_cancels_stop_timer(monkeypatch):
    from me2 import app

    calls = []
    class Timer:
        def __init__(self, seconds, callback):
            assert seconds == 20
            self.callback = callback
        def start(self):
            calls.append("start")
            self.callback()
        def cancel(self):
            calls.append("cancel")
    monkeypatch.setattr(app.threading, "Timer", Timer)
    monkeypatch.setattr(app, "build_streaming_pipeline", lambda *a, **k: object())
    def select(args, sample_rate):
        assert sample_rate == 16_000
        args.audio_backend = "sounddevice"
        args.alsa_device = "default"
    monkeypatch.setattr(app, "_select_capture_for_args", select)
    def run(pipeline, *, input_device, audio_backend, alsa_device, stop_event):
        assert audio_backend == "sounddevice" and alsa_device == "default"
        assert stop_event.is_set()
        calls.append("run")
    monkeypatch.setattr(app, "run_microphone_stream", run)
    app._run_continuous_interactive(parse_args(["--interactive", "--session-seconds", "20"]),
                                   None, {"audio": {"sample_rate": 16_000}}, None, {})
    assert calls == ["start", "run", "cancel"]


def test_live_trial_requires_explicit_capture_approval_and_fresh_log(monkeypatch, tmp_path):
    from scripts import laptop_capture_trial as trial

    args = ["trial", "--config", "unused.json", "--device", "unused", "--trial-id", "test",
            "--log", str(tmp_path / "trial.jsonl")]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        trial.main()
    assert error.value.code == 2
    path = tmp_path / "trial.jsonl"
    path.write_text("preserve\n")
    monkeypatch.setattr(sys, "argv", [*args, "--microphone-approved"])
    with pytest.raises(SystemExit) as error:
        trial.main()
    assert error.value.code == 2 and path.read_text() == "preserve\n"


def test_live_trial_watchdog_stops_child_and_records_safe_failure(monkeypatch, tmp_path):
    from scripts import laptop_capture_trial as trial

    class Process:
        def wait(self, *, timeout):
            raise subprocess.TimeoutExpired("simulated", timeout)
    monkeypatch.setattr(trial.subprocess, "Popen", lambda *a, **k: Process())
    stopped = []
    monkeypatch.setattr(trial, "stop_child", lambda process: stopped.append(True))
    log = tmp_path / "trial.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "trial", "--config", "unused.json", "--device", "unused", "--trial-id", "test",
        "--log", str(log), "--seconds", "5", "--microphone-approved",
    ])
    with pytest.raises(SystemExit) as error:
        trial.main()
    assert error.value.code == 1 and stopped == [True]
    record = json.loads(log.read_text())
    assert record["system_error"] == "CaptureWatchdogTimeout"
    assert record["raw_audio_saved"] is False
