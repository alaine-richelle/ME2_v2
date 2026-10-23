import dataclasses
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from me2.app import _load_vcm_wav, _model, _predict_vcm
from me2.dashboard import (
    ACTIONS,
    DEFAULT_ACTION_BY_INTENT,
    DEMO_SLOTS,
    INTENTS,
    Dashboard,
    load_settings,
    make_server,
)
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


@pytest.fixture
def live_dashboard(tmp_path):
    return Dashboard(
        {"mappings_file": tmp_path / "live-mappings.json", "input_device": None},
        demo=False,
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


@pytest.mark.parametrize("intent", INTENTS)
def test_every_canonical_intent_traverses_result_event_dispatch_to_visible_state(
    dashboard, intent, monkeypatch
):
    assert len(INTENTS) == 19
    assert dashboard.mappings[intent] == {
        "enabled": True, "action": DEFAULT_ACTION_BY_INTENT[intent]
    }

    def complete_weather():
        dashboard.data["devices"]["weather"].update(
            status="ready", city="Manila",
            current={"condition": "Clear", "observation_time": "2026-10-15T12:00"},
            cached_at="2026-10-15T12:00:00+00:00",
        )
        dashboard.data["last_action_status"] = "success"

    monkeypatch.setattr(dashboard, "_start_weather_lookup", complete_weather)
    command = result(intent=intent, slots=DEMO_SLOTS.get(intent, {}))
    dashboard.on_event(PipelineEvent(
        PipelineState.ACCEPTED_OR_REJECTED, 100, "accepted", command,
        utterance_id=f"accepted-{intent}",
    ))
    snapshot = dashboard.snapshot()
    assert snapshot["last_result"]["actionable_intent"] == intent
    assert snapshot["last_action"] == ACTIONS[intent]
    assert snapshot["last_action_status"] == "success"
    assert snapshot["counts"]["accepted"] == 1
    devices = snapshot["devices"]
    if intent in {"play_music", "pause", "stop"}:
        assert devices["player"] == {
            "play_music": "playing", "pause": "paused", "stop": "stopped",
        }[intent]
    elif intent == "next":
        assert devices["track"] == 2
    elif intent == "volume_up":
        assert devices["volume"] == 60
    elif intent == "volume_down":
        assert devices["volume"] == 40
    elif intent == "light_on":
        assert devices["light"] is True
    elif intent == "light_off":
        assert devices["light"] is False
    elif intent == "brightness":
        assert devices["brightness"] == 60
    elif intent == "color":
        assert devices["color"] == "Blue"
    elif intent == "temperature":
        assert devices["thermostat_c"] == 22
    elif intent == "time":
        assert devices["time"]
    elif intent == "weather":
        assert devices["weather"]["status"] == "ready"
    elif intent == "call":
        assert "no call was placed" in devices["call_demo"]["message"]
    elif intent == "message":
        assert "not sent" in devices["message_demo"]["message"]
    elif intent == "list_reminders":
        assert devices["reminders_viewed_at"]
    elif intent == "create_reminder":
        assert devices["reminders"][-1]["text"] == "Drink water"
    elif intent == "timer":
        assert devices["timers"][-1]["seconds"] == 10
    elif intent == "alarm":
        assert devices["alarms"][-1]["time"] == "08:00"


def test_file_based_vcm_result_reaches_dashboard_without_opening_speaker(tmp_path):
    root = Path(__file__).resolve().parents[1]
    model_path = root / "vcm/models/vcm_ctc_model.me2"
    audio_path = root / "vcm/data/test_sample.wav"
    if not model_path.is_file() or not audio_path.is_file():
        pytest.skip("local selected VCM bundle or file fixture is not present")

    model, bundle = _model(model_path, "vcm")
    waveform, sample_length = _load_vcm_wav(bundle, audio_path)
    command, _ = _predict_vcm(model, bundle, waveform, sample_length=sample_length)
    assert command.accepted
    assert command.actionable_intent == "play_music"

    dashboard = Dashboard(
        {
            "mappings_file": tmp_path / "mappings.json",
            "music": {"playback_enabled": False},
        },
        demo=False,
    )
    try:
        dashboard.on_event(PipelineEvent(
            PipelineState.ACCEPTED_OR_REJECTED, 100, "accepted", command,
            utterance_id="file-inference-play-music",
        ))
        snapshot = dashboard.snapshot()
        assert snapshot["devices"]["player"] == "playing"
        assert snapshot["last_result"]["actionable_intent"] == "play_music"
        assert snapshot["last_action_status"] == "success"
    finally:
        dashboard.close()


def test_slot_dependent_action_rejects_missing_or_unsupported_value_without_mutation(dashboard):
    before = dict(dashboard.data["devices"])
    missing = dashboard.dispatch(result(intent="color", slots={}))
    assert "required supported slot is missing" in missing
    assert dashboard.data["devices"] == before
    unsupported = dashboard.dispatch(result(intent="color", slots={"value": "Purple"}))
    assert "unsupported or unavailable color slot" in unsupported
    assert dashboard.data["devices"] == before
    assert dashboard.data["last_action_status"] == "failed"
    unexpected = dashboard.dispatch(result(intent="light_on", slots={"value": "Red"}))
    assert "unsupported slot" in unexpected
    assert dashboard.data["devices"] == before


def test_legacy_display_mappings_upgrade_in_memory_but_do_not_rewrite_user_file(tmp_path):
    path = tmp_path / "mappings.json"
    mappings = {
        intent: {"enabled": intent != "weather", "action": "display"}
        for intent in INTENTS
    }
    path.write_text(json.dumps(mappings), encoding="utf-8")
    dashboard = Dashboard({"mappings_file": path}, demo=True)
    assert dashboard.mappings["weather"] == {"enabled": False, "action": "weather"}
    assert json.loads(path.read_text()) == mappings


def test_unknown_accepted_intent_fails_closed(dashboard):
    action = dashboard.dispatch(result(intent="unknown_future_intent"))
    assert action == "Failed · unknown intent"
    assert dashboard.data["devices"]["light"] is False
    assert dashboard.data["last_action_status"] == "failed"


def test_system_error_result_never_mutates_devices(dashboard):
    before = json.loads(json.dumps(dashboard.data["devices"]))
    dashboard.on_event(PipelineEvent(
        PipelineState.ACCEPTED_OR_REJECTED, 10, "system_error",
        result(intent="light_on", system_error="RuntimeError"),
        utterance_id="system-error-no-dispatch",
    ))
    assert dashboard.snapshot()["devices"] == before
    assert dashboard.snapshot()["last_result"]["actionable_intent"] is None


def test_player_action_failure_does_not_advance_visible_track(tmp_path):
    dashboard = Dashboard(
        {
            "mappings_file": tmp_path / "mappings.json",
            "music": {"playback_enabled": True},
        },
        demo=False,
    )
    dashboard.data["devices"]["player"] = "playing"
    before = json.loads(json.dumps(dashboard.data["devices"]))
    dashboard.audio_player.next_track = lambda: (_ for _ in ()).throw(OSError("unavailable"))
    try:
        assert "action unavailable" in dashboard.dispatch(result(intent="next"))
        assert dashboard.data["devices"] == before
    finally:
        dashboard.close()


def test_live_music_device_failure_does_not_change_dashboard_state(tmp_path):
    dashboard = Dashboard(
        {
            "mappings_file": tmp_path / "mappings.json",
            "music": {"playback_enabled": True, "sample_rate": 22_050},
        },
        demo=False,
    )
    dashboard.audio_player.play = lambda: (_ for _ in ()).throw(OSError("device unavailable"))
    before = dict(dashboard.data["devices"])
    try:
        outcome = dashboard.dispatch(result(intent="play_music"))
        assert "action unavailable" in outcome
        assert dashboard.data["devices"] == before
        assert dashboard.data["last_action_status"] == "failed"
    finally:
        dashboard.close()


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


def test_mapping_cannot_redirect_intent_to_another_allowlisted_action(dashboard):
    mappings = json.loads(json.dumps(dashboard.mappings))
    mappings["light_on"] = {"enabled": True, "action": "light_off"}
    with pytest.raises(ValueError, match="canonical dashboard action"):
        dashboard.save_mappings(mappings)


def test_duplicate_utterance_id_is_dispatched_once(dashboard):
    first = PipelineEvent(
        PipelineState.ACCEPTED_OR_REJECTED, 10, "accepted", result(intent="next"),
        utterance_id="same-interaction",
    )
    dashboard.on_event(first)
    dashboard.on_event(first)
    snapshot = dashboard.snapshot()
    assert snapshot["devices"]["track"] == 2
    assert snapshot["counts"]["accepted"] == 1
    assert snapshot["counts"]["duplicates"] == 1
    assert snapshot["last_action_status"] == "duplicate"


def test_weather_success_and_failure_are_visible_and_never_break_dispatch(dashboard, monkeypatch):
    import me2.dashboard as module

    expected = {
        "city": "Manila", "condition": "Cloudy", "temperature": 29,
        "temperature_unit": "°C", "apparent_temperature": 34,
        "relative_humidity_percent": 78, "observation_time": "2026-10-14T12:00",
        "timezone": "Asia/Manila", "precipitation": 0.0,
        "precipitation_unit": "mm", "wind_speed": 4.0, "wind_speed_unit": "km/h",
        "weather_code": 2, "region": None, "country": "Philippines",
    }
    monkeypatch.setattr(module, "fetch_current_weather", lambda city, timeout_seconds: expected)
    assert "weather" in dashboard.dispatch(result(intent="weather"))
    await_condition(lambda: dashboard.snapshot()["devices"]["weather"]["status"] == "ready")
    assert dashboard.snapshot()["devices"]["weather"]["current"] == expected
    previous = dashboard.snapshot()["devices"]["weather"]["current"]

    dashboard.weather_settings["cache_ttl_seconds"] = 0
    monkeypatch.setattr(
        module, "fetch_current_weather",
        lambda city, timeout_seconds: (_ for _ in ()).throw(RuntimeError("private detail")),
    )
    assert "weather" in dashboard.dispatch(result(intent="weather"))
    await_condition(lambda: dashboard.snapshot()["devices"]["weather"]["status"] == "stale")
    weather = dashboard.snapshot()["devices"]["weather"]
    assert weather["current"] == previous
    assert "cached observation" in weather["error"]
    assert "private detail" not in json.dumps(weather)


def test_weather_failure_without_cache_is_structured_and_nonfatal(dashboard, monkeypatch):
    import me2.dashboard as module

    monkeypatch.setattr(
        module, "fetch_current_weather",
        lambda city, timeout_seconds: (_ for _ in ()).throw(TimeoutError("raw detail")),
    )
    dashboard.dispatch(result(intent="weather"))
    await_condition(lambda: dashboard.snapshot()["devices"]["weather"]["status"] == "error")
    snapshot = dashboard.snapshot()
    assert snapshot["last_action_status"] == "failed"
    assert "check network and city settings" in snapshot["last_action"]
    assert "raw detail" not in json.dumps(snapshot)


def test_local_dashboard_actions_apply_supported_slot_values(dashboard, monkeypatch):
    monkeypatch.setattr(dashboard, "_start_weather_lookup", lambda: None)
    dashboard.dispatch(result(intent="brightness", slots={"value": "60"}))
    dashboard.dispatch(result(intent="color", slots={"value": "Blue"}))
    dashboard.dispatch(result(intent="temperature", slots={"value": "22"}))
    dashboard.dispatch(result(intent="timer", slots={"value": "10 seconds"}))
    dashboard.dispatch(result(intent="alarm", slots={"value": "8:00 AM"}))
    dashboard.dispatch(result(intent="create_reminder", slots={"value": "Drink water"}))
    devices = dashboard.snapshot()["devices"]
    assert devices["brightness"] == 60
    assert devices["color"] == "Blue"
    assert devices["thermostat_c"] == 22
    assert devices["timers"][0]["seconds"] == 10
    assert devices["alarms"][0]["time"] == "08:00"
    assert devices["reminders"][0]["text"] == "Drink water"


def test_stop_suppresses_a_command_finishing_during_shutdown(dashboard):
    dashboard.stop()
    assert "stopping" in dashboard.dispatch(result())
    assert dashboard.data["devices"]["light"] is False


def _wake_policy(mode="primary"):
    return {
        "primary_role": "v5_primary",
        "fallback_role": "v3_fault_fallback",
        "mode": mode,
        "primary_status": "ready" if mode == "primary" else "inference_failed",
        "fallback_status": "ready",
        "primary_failure": None if mode == "primary" else "RuntimeError",
        "fallback_failure": None,
    }


def _wake_event(utterance_id="wake-1"):
    return PipelineEvent(
        PipelineState.CAPTURING_COMMAND, 10, "wake_detected",
        utterance_id=utterance_id,
    )


def test_wake_acknowledgement_lifecycle_and_primary_chime_request(live_dashboard):
    requests = []
    live_dashboard.wake_chime.enabled = True
    live_dashboard.wake_chime.request = lambda: requests.append("requested") or True
    live_dashboard.on_wake("ateko", 0.91, 4.0, True, _wake_policy())
    live_dashboard.on_event(_wake_event())
    acknowledgement = live_dashboard.snapshot()["wake_acknowledgement"]
    assert acknowledgement["visible"] is True
    assert acknowledgement["message"] == "Yes? Listening for your command."
    assert acknowledgement["model_role"] == "v5_primary"
    assert acknowledgement["audio_status"] == "queued"
    assert requests == ["requested"]

    live_dashboard.on_event(PipelineEvent(
        PipelineState.ACCEPTED_OR_REJECTED, 20, "accepted", result(),
        utterance_id="wake-1",
    ))
    acknowledgement = live_dashboard.snapshot()["wake_acknowledgement"]
    assert acknowledgement["visible"] is False
    assert acknowledgement["timestamp"] is None


def test_disabled_chime_keeps_the_visual_acknowledgement(live_dashboard):
    live_dashboard.on_wake("ateko", 0.91, 4.0, True, _wake_policy())
    live_dashboard.on_event(_wake_event("no-speaker-yet"))
    acknowledgement = live_dashboard.snapshot()["wake_acknowledgement"]
    assert acknowledgement["visible"] is True
    assert acknowledgement["audio_status"] == "disabled"


@pytest.mark.parametrize("policy", [
    {
        "primary_role": "v5_primary", "fallback_role": "v3_fault_fallback",
        "mode": "both_failed", "primary_status": "inference_failed",
        "fallback_status": "inference_failed", "primary_failure": "RuntimeError",
        "fallback_failure": "RuntimeError",
    },
])
def test_wake_acknowledgement_rejects_non_active_wake_paths(live_dashboard, policy):
    requests = []
    live_dashboard.wake_chime.enabled = True
    live_dashboard.wake_chime.request = lambda: requests.append("requested") or True
    live_dashboard.on_wake("unavailable", 0.0, 0.0, False, policy)
    live_dashboard.on_event(_wake_event("not-an-accepted-wake"))
    assert live_dashboard.snapshot()["wake_acknowledgement"]["visible"] is False
    assert requests == []


def test_healthy_non_wake_never_chimes(dashboard):
    requests = []
    dashboard.wake_chime.enabled = True
    dashboard.wake_chime.request = lambda: requests.append("requested") or True
    dashboard.on_wake("background", 0.1, 2.0, False, _wake_policy())
    assert dashboard.snapshot()["wake_acknowledgement"]["visible"] is False
    assert requests == []


def test_duplicate_wake_and_chime_failure_do_not_mutate_dashboard_action_state(live_dashboard):
    requests = []
    live_dashboard.wake_chime.enabled = True
    live_dashboard.wake_chime.request = lambda: requests.append("requested") or True
    live_dashboard.on_wake("ateko", 0.91, 4.0, True, _wake_policy())
    event = _wake_event("same-wake")
    live_dashboard.on_event(event)
    live_dashboard.on_event(event)
    before = json.loads(json.dumps(live_dashboard.data["devices"]))
    live_dashboard._on_chime_complete("failed", "CalledProcessError")
    acknowledgement = live_dashboard.snapshot()["wake_acknowledgement"]
    assert requests == ["requested"]
    assert acknowledgement["visible"] is True
    assert acknowledgement["audio_status"] == "failed"
    assert live_dashboard.data["devices"] == before
    live_dashboard.stop()
    assert live_dashboard.snapshot()["wake_acknowledgement"]["visible"] is False


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


@pytest.mark.parametrize("acknowledgement", [
    {"backend": "sounddevice"},
    {"alsa_device": "bad device name"},
    {"alsa_device": "plughw:0,0"},
    {"duration_ms": 501},
    {"frequency_hz": 199},
    {"volume": 0.51},
    {"unknown": True},
])
def test_wake_acknowledgement_config_rejects_unsupported_or_unbounded_values(
    tmp_path, acknowledgement
):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "schema_version": 1,
        "wake_model": "wake.me2",
        "vcm_model": "command.me2",
        "mappings_file": "state/mappings.json",
        "wake_acknowledgement": acknowledgement,
    }))
    with pytest.raises((TypeError, ValueError)):
        load_settings(config)


def test_http_assets_and_control_require_local_host_and_token(dashboard):
    server = make_server(dashboard, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/") as response:
            page = response.read()
            assert b"Voice pipeline" in page
            assert b"Atecco" in page
            assert b"Yes? Listening for your command." in page
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        with urlopen(base + "/dashboard-actions.css") as response:
            assert b".action-grid" in response.read()
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
