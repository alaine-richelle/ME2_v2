import dataclasses
import json
import sys
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from me2.dashboard import Dashboard, load_settings, make_server
from me2.streaming import (
    PipelineEvent,
    PipelineState,
    StreamingConfig,
    StreamingPipeline,
    run_microphone_stream,
)
from me2.vcm.ctc import CommandResult


@pytest.fixture
def dashboard(tmp_path):
    return Dashboard(
        {"mappings_file": tmp_path / "mappings.json", "input_device": None},
        demo=True,
    )


def result(**changes):
    values = {
        "decoded_text": "turn on the light", "intent": "light_on", "slots": {},
        "variation": "", "raw_log_score": -1.0, "normalized_log_score": -0.1,
        "candidate_posterior": 0.9, "margin": 0.5, "accepted": True,
        "rejection_reason": None,
    }
    return CommandResult(**(values | changes))


def test_rejected_diagnostic_intent_never_dispatches(dashboard):
    rejected = result(accepted=False, rejection_reason="low_score", margin=float("-inf"))
    dashboard.on_event(PipelineEvent(PipelineState.ACCEPTED_OR_REJECTED, 100, "rejected", rejected))
    data = dashboard.snapshot()
    assert data["devices"]["light"] is False
    assert data["last_result"]["actionable_intent"] is None
    assert data["last_result"]["margin"] is None
    assert data["counts"]["rejected"] == 1
    json.dumps(data, allow_nan=False)


def test_mapping_changes_persist_and_disabled_mapping_does_not_act(dashboard):
    mappings = dataclasses.asdict(result())  # Unrelated payload must not replace mappings.
    with pytest.raises(ValueError):
        dashboard.save_mappings(mappings)
    mappings = json.loads(json.dumps(dashboard.mappings))
    mappings["light_on"] = {"enabled": False, "action": "light_on"}
    dashboard.save_mappings(mappings)
    assert "disabled" in dashboard.dispatch(result())
    assert dashboard.data["devices"]["light"] is False
    restored = Dashboard(dashboard.settings, demo=True)
    assert restored.mappings == mappings


@pytest.mark.parametrize("bad", [
    {"enabled": True, "action": "run_shell"},
    {"enabled": "false", "action": "light_on"},
    {"enabled": True, "action": "light_on", "command": "anything"},
])
def test_mappings_reject_unlisted_actions_and_bad_types(dashboard, bad):
    mappings = json.loads(json.dumps(dashboard.mappings))
    mappings["light_on"] = bad
    with pytest.raises(ValueError):
        dashboard.save_mappings(mappings)
    assert not dashboard.settings["mappings_file"].exists()


def test_stop_suppresses_a_command_finishing_during_shutdown(dashboard):
    dashboard.stop()
    assert "stopping" in dashboard.dispatch(result())
    assert dashboard.data["devices"]["light"] is False


def test_decoding_event_arrives_before_predictor_runs():
    observed = []

    def predict(waveform):
        assert observed[-1].state == PipelineState.DECODING
        return result()

    pipeline = StreamingPipeline(
        StreamingConfig(
            sample_rate=1000, chunk_samples=100, wake_window_samples=100,
            wake_hop_samples=100, pre_roll_samples=0, onset_timeout_samples=500,
            minimum_command_samples=100, endpoint_silence_samples=100,
            maximum_command_samples=1000,
        ),
        lambda waveform: True, predict, on_event=observed.append,
    )
    pipeline.feed(np.ones(100, dtype=np.float32))
    pipeline.feed(np.ones(100, dtype=np.float32))
    returned = pipeline.feed(np.zeros(100, dtype=np.float32))
    assert [e.state for e in returned] == [
        PipelineState.DECODING, PipelineState.ACCEPTED_OR_REJECTED, PipelineState.COOLDOWN,
    ]
    assert len(observed) == 4


def await_condition(check, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        threading.Event().wait(0.02)
    raise AssertionError("Dashboard condition did not become true")


def test_demo_runs_real_pipeline_and_can_stop_restart(dashboard):
    try:
        dashboard.start()
        with pytest.raises(ValueError, match="already"):
            dashboard.start()
        await_condition(lambda: dashboard.snapshot()["state"] == "LISTENING_FOR_WAKE")
        dashboard.demo_command("light_on", True)
        await_condition(lambda: dashboard.snapshot()["counts"]["accepted"] == 1)
        assert dashboard.snapshot()["devices"]["light"] is True
        await_condition(lambda: dashboard.snapshot()["state"] == "LISTENING_FOR_WAKE")
        dashboard.demo_command("light_off", False)
        await_condition(lambda: dashboard.snapshot()["counts"]["rejected"] == 1)
        assert dashboard.snapshot()["devices"]["light"] is True
        dashboard.close()
        assert not dashboard.worker.is_alive()
        assert dashboard.snapshot()["state"] == "STOPPED"
        dashboard.start()
        await_condition(lambda: dashboard.snapshot()["state"] == "LISTENING_FOR_WAKE")
    finally:
        dashboard.close()


def test_live_readiness_error_is_visible_without_opening_microphone(dashboard, monkeypatch):
    import me2.dashboard as module

    dashboard.demo = False
    # Paths are converted as they would be by load_settings.
    from pathlib import Path
    dashboard.settings.update(wake_model=Path(__file__), vcm_model=Path(__file__))
    bundle = {
        "task": "wake_word", "deployment_ready": False,
        "audio": {"sample_rate": 16000, "duration_seconds": 2},
        "labels": ["background", "ateko"],
    }
    monkeypatch.setattr(module, "_model", lambda path, task: (None, bundle))
    dashboard.start()
    await_condition(lambda: not dashboard.snapshot()["running"])
    assert dashboard.snapshot()["state"] == "ERROR"
    assert "deployment_ready=false" in dashboard.snapshot()["error"]


def test_microphone_recovery_discards_queued_audio_and_stop_releases_stream(monkeypatch):
    stop = threading.Event()
    received = []
    events = []
    opened = []
    closed = []

    class InputStream:
        active = True

        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def __enter__(self):
            opened.append(self)
            if len(opened) == 1:
                # Simulate lost audio with two stale blocks waiting behind it.
                for index in range(3):
                    self.callback(np.ones((1600, 1)), 1600, None, index == 0)
            else:
                self.callback(np.zeros((1600, 1)), 1600, None, False)
            return self

        def __exit__(self, *args):
            closed.append(self)

    class StopWithImmediateRetry:
        def is_set(self):
            return stop.is_set()

        def wait(self, timeout):
            return stop.is_set()

    def on_audio(chunk):
        received.append(chunk)
        stop.set()

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=InputStream))
    pipeline = StreamingPipeline(
        StreamingConfig(), lambda waveform: False, lambda waveform: result(),
        on_event=events.append,
    )
    run_microphone_stream(pipeline, stop_event=StopWithImmediateRetry(), on_audio=on_audio)
    assert len(opened) == len(closed) == 2
    assert len(received) == 1
    assert np.count_nonzero(received[0]) == 0
    assert [event.message for event in events] == ["stream_recovered", "stopped"]
    assert pipeline.state == PipelineState.STOPPED


def test_microphone_retry_exhaustion_stops_with_actionable_error(monkeypatch):
    attempts = []

    class BrokenInputStream:
        def __init__(self, **kwargs):
            attempts.append(kwargs)

        def __enter__(self):
            raise OSError("device disconnected")

        def __exit__(self, *args):
            return None

    class ImmediateRetry:
        def is_set(self):
            return False

        def wait(self, timeout):
            return False

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(InputStream=BrokenInputStream),
    )
    pipeline = StreamingPipeline(
        StreamingConfig(),
        lambda waveform: False,
        lambda waveform: result(),
    )
    with pytest.raises(RuntimeError, match="recovery exhausted after 3 retries"):
        run_microphone_stream(pipeline, stop_event=ImmediateRetry())
    assert len(attempts) == 4
    assert pipeline.state == PipelineState.STOPPED


def test_config_paths_are_relative_to_config_file(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "wake_model": "wake.me2", "vcm_model": "command.me2",
        "mappings_file": "state/mappings.json", "input_device": "2",
    }))
    settings = load_settings(config)
    assert settings["wake_model"] == tmp_path / "wake.me2"
    assert settings["input_device"] == 2


def test_http_assets_and_control_require_local_host_and_token(dashboard):
    server = make_server(dashboard, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/") as response:
            assert b"Voice pipeline" in response.read()
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        with urlopen(base + "/api/status") as response:
            token = json.load(response)["token"]
        for headers in (
            {},
            {"X-ME2-Token": token, "Origin": "https://untrusted.example"},
            {"X-ME2-Token": token, "Host": f"untrusted.example:{server.server_port}"},
        ):
            request = Request(base + "/api/start", data=b"{}", headers=headers)
            with pytest.raises(HTTPError) as error:
                urlopen(request)
            assert error.value.code == 403
        request = Request(
            base + "/api/mappings", data=json.dumps(dashboard.mappings).encode(),
            headers={"X-ME2-Token": token, "Content-Type": "application/json"},
        )
        with urlopen(request) as response:
            assert json.load(response) == {"ok": True}
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/../dashboard.py")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
        dashboard.close()
