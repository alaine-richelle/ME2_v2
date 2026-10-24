import base64
import time

import numpy as np

from me2.vcm_dashboard import INTENTS, VoiceCommandDashboard, load_settings


def test_command_only_demo_dispatches_without_wake_state(tmp_path):
    dashboard = VoiceCommandDashboard({"mappings_file": tmp_path / "mappings.json"}, demo=True)
    try:
        dashboard.start()
        for _ in range(100):
            if dashboard.snapshot()["state"] == "LISTENING":
                break
            time.sleep(.01)
        dashboard.demo_command("light_on", True)
        for _ in range(100):
            if dashboard.snapshot()["counts"]["accepted"] == 1:
                break
            time.sleep(.01)
        snapshot = dashboard.snapshot()
        assert snapshot["last_result"]["actionable_intent"] == "light_on"
        assert snapshot["last_action"] == "light on"
        assert "wake" not in snapshot
    finally:
        dashboard.close()


def test_command_only_mappings_fail_closed(tmp_path):
    dashboard = VoiceCommandDashboard({"mappings_file": tmp_path / "mappings.json"}, demo=True)
    mappings = dashboard.mappings | {"light_on": {"enabled": False, "action": "light_on"}}
    dashboard.save_mappings(mappings)
    assert dashboard.mappings["light_on"]["enabled"] is False
    assert set(dashboard.mappings) == set(INTENTS)


def test_command_only_config_resolves_only_vcm_paths(tmp_path):
    path = tmp_path / "dashboard.json"
    path.write_text('{"vcm_model":"model.me2","mappings_file":"state/mappings.json"}')
    settings = load_settings(path)
    assert settings["vcm_model"] == tmp_path / "model.me2"
    assert "wake_model" not in settings


def test_live_browser_pcm_is_bounded_and_queued_after_start(tmp_path):
    dashboard = VoiceCommandDashboard({"mappings_file": tmp_path / "mappings.json"}, demo=False)
    dashboard.data["running"] = True
    samples = np.array([0.1, -0.1, 0.0], dtype="<f4")
    dashboard.ingest_audio(base64.b64encode(samples.tobytes()).decode(), 48_000)
    queued, sample_rate = dashboard.audio_requests.get_nowait()
    assert sample_rate == 48_000
    assert np.array_equal(queued, samples)
