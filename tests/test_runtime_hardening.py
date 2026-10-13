import io
import json
import logging
import sys
import threading
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from me2.app import (
    _capture,
    _prepare_vcm_audio,
    _require_live_wake_readiness,
    parse_args,
)
from me2.audio import AudioConfig, prepare_audio, prepare_audio_sequence
from me2.bundle import load_bundle, predict_command, predict_waveform, save_bundle
from me2.features import FeatureConfig
from me2.health import inspect_artifact
from me2.io_utils import atomic_json, file_sha256
from me2.logging_utils import (
    LogWriteError,
    _JsonLogFormatter,
    _PrivateRotatingFileHandler,
    log_event,
)
from me2.model import TinyAudioCNN
from me2.runtime_config import RuntimeSettings
from me2.streaming import PipelineState, StreamingConfig, StreamingPipeline, run_microphone_stream
from me2.vcm.ctc import (
    AcceptancePolicy,
    DecoderBudgetError,
    DecoderLimits,
    GrammarEntry,
    decode_command,
    prefix_beam_search,
)


def _bundle():
    model = TinyAudioCNN(num_classes=2, width=4)
    return {
        "task": "wake_word", "architecture": "TinyAudioCNN", "model_version": "test",
        "model_width": 4, "labels": ["background", "ateko"],
        "audio": AudioConfig(duration_seconds=1).__dict__,
        "features": FeatureConfig().__dict__, "state_dict": model.state_dict(),
    }


@pytest.mark.parametrize("mutate", [
    lambda m: m.pop("task"),
    lambda m: m["audio"].update(duration_seconds=float("nan")),
    lambda m: m["features"].update(sample_rate=8000),
    lambda m: m.update(model_width=1_000_000),
    lambda m: m.update(labels=["background", "background"]),
    lambda m: m.update(threshold=float("nan")),
    lambda m: m.update(deployment_ready="false"),
    lambda m: m["audio"].update(duration_seconds=60),
])
def test_invalid_metadata_fails_before_artifact_activation(tmp_path, mutate):
    metadata = _bundle()
    mutate(metadata)
    with pytest.raises((ValueError, TypeError)):
        save_bundle(tmp_path / "invalid.me2", metadata)
    assert not list(tmp_path.iterdir())


def test_archive_duplicate_members_and_oversize_are_rejected(tmp_path, monkeypatch):
    import me2.bundle as module

    original = tmp_path / "original.me2"
    save_bundle(original, _bundle())
    duplicate = tmp_path / "duplicate.me2"
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(duplicate, "w") as target:
        for name in source.namelist():
            target.writestr(name, source.read(name))
        with pytest.warns(UserWarning):
            target.writestr("metadata.json", source.read("metadata.json"))
    with pytest.raises(ValueError, match="duplicate"):
        load_bundle(duplicate)
    monkeypatch.setattr(module, "MAX_METADATA_BYTES", 1)
    with pytest.raises(ValueError, match="budget"):
        load_bundle(original)


def test_nonfinite_weights_never_replace_known_good(tmp_path):
    path = tmp_path / "known-good.me2"
    metadata = _bundle()
    save_bundle(path, metadata)
    digest = file_sha256(path)
    next(iter(metadata["state_dict"].values())).flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        save_bundle(path, metadata)
    assert file_sha256(path) == digest
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize("name", ["min_normalized_log_score", "min_margin", "min_candidate_posterior"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.5", True])
def test_acceptance_thresholds_cannot_bypass_comparison(name, value):
    with pytest.raises(ValueError):
        AcceptancePolicy.from_mapping({name: value})


def test_classifier_nan_logits_never_become_command_or_wake():
    class Invalid(torch.nn.Module):
        def forward(self, features):
            return torch.tensor([[float("nan"), 100.0]])

    bundle = _bundle()
    wave = np.zeros(16000, dtype=np.float32)
    for prediction in (predict_command, predict_waveform):
        with pytest.raises(ValueError, match="logits"):
            prediction(Invalid(), bundle, wave)
    with pytest.raises(ValueError, match="sample_length"):
        predict_command(Invalid(), bundle, wave, sample_length=16001)


def test_impossible_repeated_terminal_is_not_an_infinite_scored_candidate():
    assert prefix_beam_search(
        np.zeros((2, 2)), [GrammarEntry("aa", "repeat")], charset="a"
    ) == []
    result = decode_command(np.zeros((2, 2)), [GrammarEntry("aa", "repeat")], charset="a")
    assert not result.accepted and result.rejection_reason == "no_terminal_candidate"


def test_deterministic_decoder_limits_fail_closed_and_exclude_padding():
    entries = [GrammarEntry("a", "one")]
    logits = np.zeros((10, 2))
    with pytest.raises(DecoderBudgetError, match="frame"):
        decode_command(logits, entries, charset="a", limits=DecoderLimits(max_frames=2))
    with pytest.raises(DecoderBudgetError, match="work"):
        decode_command(logits, entries, charset="a", limits=DecoderLimits(max_expansions=1))
    padded = logits.copy()
    padded[2:] = np.nan
    result = decode_command(
        padded, entries, charset="a", input_length=2, limits=DecoderLimits(max_frames=2)
    )
    assert result.accepted and result.candidate_count == 1
    with pytest.raises(ValueError, match="beam_width"):
        decode_command(logits, entries, charset="a", beam_width=100000)


def test_runtime_config_precedence_and_path_resolution(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({
        "schema_version": 1, "runtime": {"threads": 1, "pre_roll_ms": 300},
        "paths": {"vcm_model": "candidate.me2"},
    }))
    args = parse_args(["--vcm", "--config", str(path), "--threads", "3"])
    assert args.threads == 3 and args.pre_roll_ms == 300
    assert args.vcm_model == tmp_path / "candidate.me2"
    with pytest.raises(ValueError):
        RuntimeSettings(speech_rms_threshold=float("nan"))
    path.write_text('{"schema_version":1,"runtime":{"threadz":2}}')
    with pytest.raises(SystemExit):
        parse_args(["--vcm", "--config", str(path)])


def test_microphone_and_file_preparation_share_ctc_and_classifier_contract():
    audio = AudioConfig(duration_seconds=1)
    wave = np.concatenate([np.zeros(1000), np.ones(4000) * 0.3, np.zeros(1000)])
    for architecture in ("TinyAudioCNN", "CharacterAudioCNN"):
        bundle = {"architecture": architecture, "audio": audio.__dict__}
        observed, length = _prepare_vcm_audio(bundle, wave, 16000)
        if architecture == "CharacterAudioCNN":
            expected, expected_length = prepare_audio_sequence(wave, 16000, audio)
            assert length == expected_length
        else:
            expected = prepare_audio(wave, 16000, audio)
            assert length is None
        np.testing.assert_array_equal(observed, expected)


def test_missing_wake_approval_does_not_bypass_live_gate():
    with pytest.raises(ValueError, match="approval"):
        _require_live_wake_readiness({}, allow_unverified=False)
    _require_live_wake_readiness({"deployment_ready": True}, allow_unverified=False)


def test_logs_default_does_not_retain_private_message_trace_or_fields():
    logger = logging.getLogger("test.privacy")
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(_JsonLogFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        try:
            raise ValueError("/Users/private/secret.wav contact Alice")
        except ValueError:
            logger.exception("transcript: call Alice secret-token")
        log_event(logger, "command_result", accepted=False, utterance_id="abc123",
                  reason="no_terminal_candidate", transcript="call Alice",
                  model_version="/Users/private/model.me2")
    finally:
        logger.removeHandler(handler)
    rendered = output.getvalue()
    assert all(text not in rendered for text in ("Alice", "/Users", "secret-token", "secret.wav"))
    rows = [json.loads(line) for line in rendered.splitlines()]
    assert rows[0]["error_type"] == "ValueError"
    assert rows[1]["reason"] == "no_terminal_candidate"
    assert rows[1]["utterance_id"] == "abc123"
    assert rows[1]["accepted"] is False
    record = logging.LogRecord("test", logging.DEBUG, "", 1, "call Alice", (), None)
    assert "call Alice" in _JsonLogFormatter(debug_sensitive=True).format(record)


def test_stopped_capture_cleans_up_even_when_stop_was_requested_before_start(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace())
    pipeline = StreamingPipeline(StreamingConfig(), lambda _: False, lambda _: None)
    pipeline._wake_buffer.extend([0.1] * 10)
    stop = threading.Event()
    stop.set()
    run_microphone_stream(pipeline, stop_event=stop)
    assert pipeline.state == PipelineState.STOPPED
    assert len(pipeline._wake_buffer) == 0
    assert pipeline.feed(np.ones(1600)) == []


def test_health_verifies_checksum_and_never_marks_experimental_wake_ready(tmp_path):
    path = tmp_path / "wake.me2"
    save_bundle(path, _bundle())
    with pytest.raises(ValueError, match="checksum"):
        inspect_artifact(path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="approval"):
        inspect_artifact(path, require_approved_wake=True)
    report = inspect_artifact(path, expected_sha256=file_sha256(path))
    assert report["synthetic_operator_smoke_only"]
    assert "wake_not_approved" in report["warnings"]


def test_atomic_report_failure_preserves_previous_output(tmp_path, monkeypatch):
    from pathlib import Path

    path = tmp_path / "report.json"
    path.write_text('{"known_good": true}\n')
    monkeypatch.setattr(Path, "replace", lambda *_: (_ for _ in ()).throw(OSError("disk fault")))
    with pytest.raises(OSError):
        atomic_json(path, {"new": True})
    assert json.loads(path.read_text()) == {"known_good": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_log_startup_failure_is_nonzero_without_private_traceback(monkeypatch, capsys):
    from me2 import app

    monkeypatch.setattr(sys, "argv", ["me2", "--vcm"])
    def fail_logging(*args, **kwargs):
        raise PermissionError("/Users/private/secret/log.jsonl")
    monkeypatch.setattr(app, "configure_logging", fail_logging)
    with pytest.raises(SystemExit) as raised:
        app.main()
    assert raised.value.code == 1
    stderr = capsys.readouterr().err
    assert "PermissionError" in stderr
    assert "/Users/private" not in stderr


def test_interrupted_bundle_write_cleans_temp_and_keeps_previous_model(tmp_path, monkeypatch):
    from pathlib import Path

    path = tmp_path / "model.me2"
    bundle = _bundle()
    save_bundle(path, bundle)
    digest = file_sha256(path)
    def interrupted_replace(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(Path, "replace", interrupted_replace)
    with pytest.raises(KeyboardInterrupt):
        save_bundle(path, bundle)
    assert file_sha256(path) == digest
    assert not list(tmp_path.glob("*.tmp"))


def test_single_component_capture_stops_on_interrupt(monkeypatch):
    stopped = []
    def interrupted_wait():
        raise KeyboardInterrupt
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(
        query_devices=lambda **kwargs: {"name": "mock"},
        rec=lambda *args, **kwargs: np.zeros((16000, 1)),
        wait=interrupted_wait,
        stop=lambda: stopped.append(True),
    ))
    with pytest.raises(KeyboardInterrupt):
        _capture(1, 16000)
    assert stopped == [True]


def test_disk_full_logging_failure_never_prints_original_private_message(tmp_path, capsys):
    class FullDisk(io.StringIO):
        def write(self, text):
            raise OSError("disk full at /Users/private/path")

    handler = _PrivateRotatingFileHandler(tmp_path / "runtime.jsonl", maxBytes=100)
    handler.stream.close()
    handler.stream = FullDisk()
    handler.setFormatter(_JsonLogFormatter())
    record = logging.LogRecord("test", logging.ERROR, "", 1,
                               "call Alice at /Users/private/audio.wav", (), None)
    with pytest.raises(LogWriteError, match="structured_log_write_failed"):
        handler.handle(record)
    assert capsys.readouterr().err == ""


def test_log_sink_failure_is_fatal_not_retried_as_microphone_fault(monkeypatch):
    closed = []
    waits = []
    class InputStream:
        active = True
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]
        def __enter__(self):
            self.callback(np.zeros((1600, 1)), 1600, None, False)
            return self
        def __exit__(self, *args):
            closed.append(True)
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=InputStream))
    stop = SimpleNamespace(is_set=lambda: False, wait=lambda delay: waits.append(delay))
    pipeline = StreamingPipeline(StreamingConfig(), lambda _: False, lambda _: None)
    def fail_sink(_):
        raise LogWriteError("structured_log_write_failed")
    with pytest.raises(LogWriteError):
        run_microphone_stream(pipeline, stop_event=stop, on_audio=fail_sink)
    assert closed == [True] and waits == []
    assert pipeline.state == PipelineState.STOPPED
