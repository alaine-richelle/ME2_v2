import csv
import json

import pytest

from me2.pi_report import (
    NO_RESPONSE,
    REJECT,
    _class_metrics,
    _wilson,
    generate_report,
    score_run,
)


def schema():
    return {
        "intents": [
            "alarm", "brightness", "call", "color", "create_reminder",
            "light_off", "light_on", "list_reminders", "message", "next",
            "pause", "play_music", "stop", "temperature", "time", "timer",
            "volume_down", "volume_up", "weather",
        ],
        "variations": {
            "light_on": ["Lights on"],
            "light_off": ["Lights out"],
            "brightness": ["Brightness 20 percent", "Brightness 0 percent"],
        },
        "slots": {
            "brightness": {
                "required": ["value"], "numeric_kind": "percent", "unit": "percent",
            },
            "timer": {
                "required": ["value"], "numeric_kind": "seconds", "unit": "seconds",
            },
            "color": {"required": ["value"], "numeric_kind": None, "unit": None},
        },
    }


def metadata():
    return {
        "response_window_ms": 1000,
        "evaluation_mode": "wake-gated replay",
        "status": "completed",
    }


def test_first_decision_and_extra_wrong_fire_are_scored_separately():
    manifest = [{
        "clip_id": "one",
        "expected_intent": "light_on",
        "expected_command": "Lights on",
        "expected_slots": "{}",
        "wake_eligible": True,
    }]
    events = [
        {"clip_id": "one", "event_type": "wake_detected", "relative_ms": -200},
        {
            "clip_id": "one", "event_type": "command", "relative_ms": 100,
            "intent": "light_on", "variation": "Lights on", "slots": {},
        },
        {
            "clip_id": "one", "event_type": "command", "relative_ms": 200,
            "intent": "light_off", "variation": "Lights out", "slots": {},
        },
    ]
    metrics, predictions, _ = score_run(manifest, events, [], metadata(), schema())
    assert metrics["intent"]["accuracy"] == 1.0
    assert metrics["command"]["accuracy"] == 1.0
    assert metrics["intent"]["misfires"] == 1
    assert metrics["command"]["misfires"] == 1
    assert metrics["pipeline"]["total_extra_fires"] == 1
    assert metrics["pipeline"]["wake_detection_rate"] == 1.0
    assert predictions[0]["primary_intent"] == "light_on"


def test_explicit_reject_and_silence_share_classification_but_not_pipeline_count():
    manifest = [
        {"clip_id": "explicit", "expected_intent": "light_on", "expected_command": "Lights on"},
        {"clip_id": "silent", "expected_intent": "light_off", "expected_command": "Lights out"},
    ]
    events = [{"clip_id": "explicit", "event_type": "reject", "relative_ms": 20}]
    metrics, predictions, _ = score_run(manifest, events, [], metadata(), schema())
    assert [row["classification_intent"] for row in predictions] == [REJECT, REJECT]
    assert predictions[1]["primary_intent"] == NO_RESPONSE
    assert metrics["intent"]["false_rejects"] == 2
    assert metrics["pipeline"]["response_count"] == 1
    assert metrics["pipeline"]["no_response_count"] == 1


def test_oos_unknown_command_is_false_accept_and_unknown_name():
    manifest = [{"clip_id": "oos", "out_of_scope": True}]
    events = [{
        "clip_id": "oos", "event_type": "command", "relative_ms": 50,
        "intent": "LIGHT_MAGIC", "variation": "Magic lights",
    }]
    metrics, _, tables = score_run(manifest, events, [], metadata(), schema())
    assert metrics["intent"]["false_accepts"] == 1
    assert metrics["intent"]["false_accept_rate"] == 1.0
    assert metrics["unknown_intents"]["light_magic"]["clip_ids"] == ["oos"]
    assert tables["intent_confusions"][0]["got"] == "light_magic"


def test_right_intent_wrong_variation_is_only_command_error():
    manifest = [{
        "clip_id": "wrong-variation",
        "expected_intent": "light_on",
        "expected_command": "Lights on",
    }]
    events = [{
        "clip_id": "wrong-variation", "event_type": "command", "relative_ms": 10,
        "intent": "light_on", "variation": "Power on the lights",
    }]
    metrics, _, _ = score_run(manifest, events, [], metadata(), schema())
    assert metrics["intent"]["accuracy"] == 1.0
    assert metrics["command"]["accuracy"] == 0.0


def test_missing_variation_instrumentation_marks_command_metrics_unavailable():
    manifest = [{
        "clip_id": "missing",
        "expected_intent": "light_on",
        "expected_command": "Lights on",
    }]
    events = [{
        "clip_id": "missing", "event_type": "command", "relative_ms": 10,
        "intent": "light_on",
    }]
    metrics, _, tables = score_run(manifest, events, [], metadata(), schema())
    assert metrics["command"] is None
    assert "Pi output lacks" in metrics["command_metrics_unavailable_reason"]
    assert tables["command_confusions"] == []


def test_macro_f2_keeps_represented_class_with_no_correct_predictions():
    result = _class_metrics(["a", "b"], ["a", "a"])
    assert result["per_class"]["b"]["precision"] == 0.0
    assert result["per_class"]["b"]["recall"] == 0.0
    assert result["per_class"]["b"]["f2"] == 0.0
    assert result["macro_f2"] == pytest.approx(5 / 12)
    assert _wilson(0, 0) is None
    lower, upper = _wilson(1, 1)
    assert 0 <= lower < upper <= 1


def test_slot_metrics_count_invalid_and_zero_reference_without_epsilon():
    manifest = [
        {
            "clip_id": "invalid", "expected_intent": "brightness",
            "expected_command": "Brightness 20 percent",
            "expected_slots": json.dumps({"value": "20 percent"}),
        },
        {
            "clip_id": "zero", "expected_intent": "brightness",
            "expected_command": "Brightness 0 percent",
            "expected_slots": json.dumps({"value": "0 percent"}),
        },
    ]
    events = [
        {
            "clip_id": "invalid", "event_type": "command", "relative_ms": 1,
            "intent": "brightness", "variation": "Brightness 20 percent",
            "slots": {"value": "loud"},
        },
        {
            "clip_id": "zero", "event_type": "command", "relative_ms": 1,
            "intent": "brightness", "variation": "Brightness 0 percent",
            "slots": {"value": "10 percent"},
        },
    ]
    _, _, tables = score_run(manifest, events, [], metadata(), schema())
    brightness = next(row for row in tables["slots"] if row["intent"] == "brightness")
    assert brightness["invalid_numeric"] == 1
    assert brightness["valid_numeric"] == 1
    assert brightness["zero_reference_count"] == 1
    assert brightness["mean_relative_error"] is None
    assert next(row for row in tables["slots"] if row["intent"] == "all")["mae_unit"] == (
        "N/A — mixed units"
    )


def test_pending_report_has_exact_sections_and_regenerable_empty_tables(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    source_schema = json.loads(
        (
            __import__("pathlib").Path(__file__).parents[1]
            / "config/command_schema_v1.json"
        ).read_text()
    )
    schema_path = project / "schema.json"
    schema_path.write_text(json.dumps(source_schema))
    metadata_path = project / "metadata.json"
    metadata_path.write_text(json.dumps({
        "run_id": "pending", "status": "not_run",
        "unavailable_reason": "Pi was unavailable",
    }))
    output = project / "reports/pending"
    metrics = generate_report(
        output_dir=output,
        metadata_path=metadata_path,
        schema_path=schema_path,
    )
    report = (output / "report.md").read_text()
    headings = [line for line in report.splitlines() if line.startswith("# ")]
    assert headings == [
        "# 1. Header",
        "# 2. Classification",
        "# 3. Slot values",
        "# 4. Raspberry Pi",
        "# 5. Most frequent confusions",
        "# 6. Per-intent scores",
    ]
    assert "N/A — Pi was unavailable" in report
    assert metrics["counts"]["n_all"] == 0
    with (output / "predictions.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == []
    assert json.loads((output / "metrics.json").read_text()) == metrics


def test_completed_report_regenerates_identical_metrics_from_saved_inputs(tmp_path):
    root = __import__("pathlib").Path(__file__).parents[1]
    schema_path = root / "config/command_schema_v1.json"
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "run_id": "fixture",
        "status": "completed",
        "response_window_ms": 1000,
        "evaluation_mode": "command-only",
        "test_duration_seconds": 5,
        "assistant_cpu_seconds": 1,
        "annotated_audio_seconds": 2,
        "pi": {"model": "fixture-pi"},
    }))
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "clip_id", "expected_intent", "expected_command", "expected_slots",
            "speaker_id", "source",
        ])
        writer.writeheader()
        writer.writerow({
            "clip_id": "one", "expected_intent": "light_on",
            "expected_command": "Lights on", "expected_slots": "{}",
            "speaker_id": "speaker-a", "source": "real",
        })
        writer.writerow({
            "clip_id": "two", "expected_intent": "light_off",
            "expected_command": "Lights out", "expected_slots": "{}",
            "speaker_id": "speaker-b", "source": "synthetic",
        })
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("\n".join([
        json.dumps({
            "clip_id": "one", "event_type": "command", "relative_ms": 100,
            "intent": "light_on", "variation": "Lights on", "slots": {},
            "infer_ms": 20, "audio_ms": 1000,
        }),
        json.dumps({"clip_id": "two", "event_type": "reject", "relative_ms": 200}),
    ]) + "\n")
    telemetry_path = tmp_path / "telemetry.csv"
    telemetry_path.write_text(
        "timestamp,cpu_temperature_c,throttling_flags\n0,45,0x0\n1,46,0x1\n"
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_metrics = generate_report(
        output_dir=first,
        metadata_path=metadata_path,
        schema_path=schema_path,
        manifest_path=manifest_path,
        events_path=events_path,
        telemetry_path=telemetry_path,
    )
    second_metrics = generate_report(
        output_dir=second,
        metadata_path=metadata_path,
        schema_path=schema_path,
        manifest_path=manifest_path,
        events_path=events_path,
        telemetry_path=telemetry_path,
    )
    assert first_metrics == second_metrics
    assert first_metrics["intent"]["correct"] == 1
    assert first_metrics["intent"]["speaker_grouped_bootstrap_95"] is not None
    assert first_metrics["command"]["false_rejects"] == 1
    assert (first / "input_references.json").exists()
    assert (first / "telemetry_metrics.csv").exists()
