import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/pi5_collect_telemetry.py"
SPEC = importlib.util.spec_from_file_location("pi5_collect_telemetry", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _record(event, timestamp, **fields):
    return {"event": event, "timestamp": timestamp, **fields}


def test_summarizer_reports_stream_health_without_transcripts_or_audio(tmp_path):
    rows = [
        _record("stream_health", "2026-10-15T10:00:00+00:00", diagnostics={
            "queue_depth_mean": 1.0, "queue_depth_max": 2, "queue_capacity_chunks": 32,
            "queue_drop_count": 0, "callback_status_count": 0,
            "callback_frames_total": 16000, "callback_count": 10,
            "queue_delay_p95_ms": 4.0, "feed_p95_ms": 12.0,
        }),
        _record("stream_health", "2026-10-15T10:00:05+00:00", diagnostics={
            "queue_depth_mean": 1.5, "queue_depth_max": 3, "queue_capacity_chunks": 32,
            "queue_drop_count": 0, "callback_status_count": 0,
            "callback_frames_total": 80000, "callback_count": 50,
            "queue_delay_p95_ms": 6.0, "feed_p95_ms": 18.0,
        }),
        _record("stream_health", "2026-10-15T10:00:10+00:00", diagnostics={
            "queue_depth_mean": 1.2, "queue_depth_max": 3, "queue_capacity_chunks": 32,
            "queue_drop_count": 0, "callback_status_count": 0,
            "callback_frames_total": 160000, "callback_count": 100,
            "queue_delay_p95_ms": 5.0, "feed_p95_ms": 16.0,
        }),
        _record("wake_window", "2026-10-15T10:00:02+00:00", total_ms=42.0),
        _record("wake_window", "2026-10-15T10:00:07+00:00", total_ms=66.0),
        _record("pipeline_state", "2026-10-15T10:00:01+00:00", state="CAPTURING_COMMAND",
                reason="wake_detected", utterance_id="u1"),
        _record("pipeline_state", "2026-10-15T10:00:08+00:00", state="DECODING",
                reason="endpoint_detected", utterance_id="u1"),
        _record("command_result", "2026-10-15T10:00:08.030+00:00", diagnostics={
            "command_decision_ms": 29.5,
        }),
        _record("pipeline_state", "2026-10-15T10:00:08.031+00:00",
                state="ACCEPTED_OR_REJECTED", reason="accepted", utterance_id="u1"),
        _record("capture_activity", "2026-10-15T10:00:08+00:00", diagnostics={
            "clipped_samples": 2,
        }),
        _record("private_input", "2026-10-15T10:00:09+00:00",
                transcript="private spoken phrase", decoded_text="private spoken phrase"),
    ]
    log = tmp_path / "safe-runtime.jsonl"
    log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    summary = MODULE.summarize_runtime_log(log, sample_rate=16_000, wake_hop_ms=250)

    assert summary["queue_drop_count"] == 0
    assert summary["callback_status_count"] == 0
    assert summary["callback_frames_total"] == 160_000
    assert summary["processed_audio_seconds"] == 10
    assert summary["queue_depth_max_observed"] == 3
    assert summary["queue_health"] == "pass"
    assert summary["queue_depth_trend"] == "stable_or_decreasing"
    assert summary["wake_window_processing_ms_p95"] == 66
    assert summary["wake_processing_keeps_pace"] == "pass"
    assert summary["queue_delay_p95_ms_max_batch"] == 6
    assert summary["feed_processing_p95_ms_max_batch"] == 18
    assert summary["chunk_processing_keeps_pace"] == "pass"
    assert summary["command_decision_ms_p95"] == 29.5
    assert summary["endpoint_to_result_ms_p95"] == 31
    assert summary["wake_events"] == 1
    assert summary["accepted_commands"] == 1
    assert summary["clipped_samples_observed"] == 2
    assert summary["false_activations_per_hour"] is None
    rendered = json.dumps(summary).casefold()
    assert "transcript" not in rendered
    assert "decoded_text" not in rendered
    assert "private spoken phrase" not in rendered
    assert "/private/" not in rendered


def test_summarizer_marks_missing_stream_evidence_and_drops_as_fail_closed(tmp_path):
    log = tmp_path / "overflow-runtime.jsonl"
    log.write_text(
        json.dumps(_record("stream_health", "2026-10-15T10:00:00+00:00", diagnostics={
            "queue_depth_mean": 4.0, "queue_depth_max": 8, "queue_capacity_chunks": 8,
            "queue_drop_count": 1, "callback_status_count": 1,
            "callback_frames_total": 32000, "callback_count": 20,
        })) + "\n",
        encoding="utf-8",
    )
    summary = MODULE.summarize_runtime_log(log, sample_rate=16_000, wake_hop_ms=250)
    assert summary["queue_drop_count"] == 1
    assert summary["callback_status_count"] == 1
    assert summary["queue_health"] == "fail"
    assert summary["chunk_processing_keeps_pace"] == "not_measured"
    assert summary["wake_processing_keeps_pace"] == "not_measured"
    assert summary["wake_events"] == 0
    assert summary["false_activations_per_hour"] is None
