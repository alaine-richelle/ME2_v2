"""Score structured Raspberry Pi trials and render the required six-section report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

INTENTS = (
    "alarm",
    "brightness",
    "call",
    "color",
    "create_reminder",
    "light_off",
    "light_on",
    "list_reminders",
    "message",
    "next",
    "pause",
    "play_music",
    "stop",
    "temperature",
    "time",
    "timer",
    "volume_down",
    "volume_up",
    "weather",
)
REJECT = "REJECT"
NO_RESPONSE = "NO_RESPONSE"
DECISION_EVENTS = {"command", "accepted", "reject", "rejected", "decision"}
COMMAND_EVENTS = {"command", "accepted"}


def _read_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.casefold() == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _as_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_mapping(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        return {str(key): str(item) for key, item in value.items()}
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, Mapping):
        return {}
    return {str(key): str(item) for key, item in parsed.items()}


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _canonical_intent(value: Any, aliases: Mapping[str, str]) -> str:
    name = _normalized_text(value).replace(" ", "_")
    if name in {"", "none", "null", "out_of_scope", "oos", "reject", "rejected"}:
        return REJECT
    return aliases.get(name, name)


def _wilson(successes: int, total: int) -> tuple[float, float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    spread = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, center - spread), min(1.0, center + spread)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f_score(precision: float, recall: float, beta: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_squared = beta * beta
    return (1 + beta_squared) * precision * recall / (
        beta_squared * precision + recall
    )


def _class_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
    *,
    all_classes: Sequence[str] | None = None,
) -> dict[str, Any]:
    if len(expected) != len(predicted):
        raise ValueError("Expected and predicted arrays must have equal length")
    represented = sorted(set(expected))
    prediction_counts = Counter(predicted)
    confusion = Counter(zip(expected, predicted))
    per_class: dict[str, dict[str, Any]] = {}
    for name in all_classes or represented:
        support = expected.count(name)
        predicted_count = prediction_counts[name]
        true_positive = confusion[(name, name)]
        if support == 0:
            per_class[name] = {
                "support": 0,
                "predicted": predicted_count,
                "precision": None,
                "recall": None,
                "f1": None,
                "f2": None,
            }
            continue
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support
        per_class[name] = {
            "support": support,
            "predicted": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1": _f_score(precision, recall, 1.0),
            "f2": _f_score(precision, recall, 2.0),
        }
    represented_rows = [per_class[name] for name in represented]
    correct = sum(left == right for left, right in zip(expected, predicted))
    return {
        "total": len(expected),
        "correct": correct,
        "accuracy": _rate(correct, len(expected)),
        "accuracy_wilson_95": _wilson(correct, len(expected)),
        "represented_classes": represented,
        "balanced_accuracy": (
            statistics.fmean(row["recall"] for row in represented_rows)
            if represented_rows
            else None
        ),
        "macro_precision": (
            statistics.fmean(row["precision"] for row in represented_rows)
            if represented_rows
            else None
        ),
        "macro_recall": (
            statistics.fmean(row["recall"] for row in represented_rows)
            if represented_rows
            else None
        ),
        "macro_f1": (
            statistics.fmean(row["f1"] for row in represented_rows)
            if represented_rows
            else None
        ),
        "macro_f2": (
            statistics.fmean(row["f2"] for row in represented_rows)
            if represented_rows
            else None
        ),
        "per_class": per_class,
        "confusion": {
            f"{left}\u2192{right}": count
            for (left, right), count in sorted(confusion.items())
        },
    }


def _event_kind(row: Mapping[str, Any]) -> str:
    kind = _normalized_text(row.get("event_type") or row.get("event"))
    if kind == "decision":
        accepted = _as_bool(row.get("accepted"), default=bool(row.get("intent")))
        return "command" if accepted else "reject"
    if kind == "rejected":
        return "reject"
    return kind


def _event_offset_ms(event: Mapping[str, Any], trial: Mapping[str, Any]) -> float | None:
    direct_value = (
        event.get("relative_ms")
        if event.get("relative_ms") not in (None, "")
        else event.get("latency_ms")
    )
    direct = _as_float(direct_value)
    if direct is not None:
        return direct
    timestamp = _as_float(event.get("timestamp_ms"))
    command_end = _as_float(trial.get("command_end_ms"))
    if timestamp is None or command_end is None:
        return None
    return timestamp - command_end


def _slot_key(slots: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), _normalized_text(value)) for key, value in slots.items()))


def associate_trials(
    manifest_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    *,
    response_window_ms: float,
    aliases: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Associate ordered events to trials without choosing a later matching answer."""

    if response_window_ms <= 0:
        raise ValueError("response_window_ms must be positive")
    aliases = aliases or {}
    events_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sequence, raw_event in enumerate(event_rows):
        event = dict(raw_event)
        clip_id = str(event.get("clip_id") or "").strip()
        if not clip_id:
            raise ValueError("Every event requires clip_id")
        event["_sequence"] = sequence
        events_by_clip[clip_id].append(event)

    predictions: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for raw_trial in manifest_rows:
        trial = dict(raw_trial)
        clip_id = str(trial.get("clip_id") or "").strip()
        if not clip_id:
            raise ValueError("Every manifest row requires clip_id")
        if clip_id in observed_ids:
            raise ValueError(f"Duplicate manifest clip_id: {clip_id}")
        observed_ids.add(clip_id)
        expected_intent = _canonical_intent(
            REJECT if _as_bool(trial.get("out_of_scope")) else trial.get("expected_intent"),
            aliases,
        )
        if expected_intent not in {*INTENTS, REJECT}:
            raise ValueError(f"Unknown ground-truth intent for {clip_id}: {expected_intent}")

        events = events_by_clip.get(clip_id, [])
        ordered = sorted(
            events,
            key=lambda item: (
                _event_offset_ms(item, trial)
                if _event_offset_ms(item, trial) is not None
                else math.inf,
                int(item["_sequence"]),
            ),
        )
        decisions = [event for event in ordered if _event_kind(event) in DECISION_EVENTS]
        command_fires = [event for event in ordered if _event_kind(event) in COMMAND_EVENTS]
        in_window = [
            event
            for event in decisions
            if (offset := _event_offset_ms(event, trial)) is not None
            and 0.0 <= offset <= response_window_ms
        ]
        primary = in_window[0] if in_window else None
        primary_kind = _event_kind(primary) if primary else ""
        if primary is None:
            observed_intent = NO_RESPONSE
            observed_command = ""
            observed_slots: dict[str, str] = {}
        elif primary_kind == "reject":
            observed_intent = REJECT
            observed_command = REJECT
            observed_slots = {}
        else:
            observed_intent = _canonical_intent(primary.get("intent"), aliases)
            observed_command = str(
                primary.get("command_id")
                or primary.get("variation_id")
                or primary.get("variation")
                or ""
            ).strip()
            observed_slots = _as_mapping(primary.get("slots"))

        expected_command = str(
            trial.get("expected_command")
            or trial.get("expected_variation")
            or trial.get("variation")
            or ""
        ).strip()
        expected_slots = _as_mapping(trial.get("expected_slots") or trial.get("slots"))
        prediction = {
            **trial,
            "clip_id": clip_id,
            "expected_intent": expected_intent,
            "expected_command": expected_command,
            "expected_slots": expected_slots,
            "primary_intent": observed_intent,
            "classification_intent": (
                REJECT if observed_intent == NO_RESPONSE else observed_intent
            ),
            "primary_command": observed_command,
            "primary_slots": observed_slots,
            "primary_event_type": primary_kind or NO_RESPONSE,
            "primary_latency_ms": _event_offset_ms(primary, trial) if primary else None,
            "command_stage_responded": primary is not None,
            "wake_detected": any(_event_kind(event) == "wake_detected" for event in ordered),
            "command_fire_count": len(command_fires),
            "extra_fire_count": max(0, len(command_fires) - 1),
            "premature_fire_count": sum(
                (_event_offset_ms(event, trial) or 0.0) < 0 for event in command_fires
            ),
            "late_fire_count": sum(
                (_event_offset_ms(event, trial) or 0.0) > response_window_ms
                for event in command_fires
            ),
            "all_emitted_intents": [
                _canonical_intent(event.get("intent"), aliases) for event in command_fires
            ],
            "all_emitted_commands": [
                str(
                    event.get("command_id")
                    or event.get("variation_id")
                    or event.get("variation")
                    or ""
                ).strip()
                for event in command_fires
            ],
            "all_emitted_slots": [_as_mapping(event.get("slots")) for event in command_fires],
            "infer_ms": _as_float(primary.get("infer_ms")) if primary else None,
            "audio_ms": _as_float(primary.get("audio_ms")) if primary else None,
        }
        predictions.append(prediction)
    unassociated = sorted(set(events_by_clip) - observed_ids)
    if unassociated:
        raise ValueError(f"Events reference unknown clip IDs: {unassociated[:5]}")
    return predictions


def _command_available(predictions: Sequence[Mapping[str, Any]]) -> tuple[bool, str]:
    missing_truth = [
        row["clip_id"]
        for row in predictions
        if row["expected_intent"] != REJECT and not str(row["expected_command"])
    ]
    if missing_truth:
        return False, f"ground truth lacks command variation IDs for {len(missing_truth)} clips"
    missing_output = [
        row["clip_id"]
        for row in predictions
        if row["primary_intent"] in INTENTS
        and not str(row["primary_command"])
    ]
    if missing_output:
        return False, f"Pi output lacks command variation IDs for {len(missing_output)} clips"
    return True, ""


def _joint_command_correct(row: Mapping[str, Any]) -> bool:
    if row["expected_intent"] == REJECT:
        return row["classification_intent"] == REJECT
    return (
        row["primary_command"] == row["expected_command"]
        and _slot_key(row["primary_slots"]) == _slot_key(row["expected_slots"])
    )


def _resolution_metrics(
    predictions: Sequence[Mapping[str, Any]], *, resolution: str
) -> dict[str, Any]:
    expected_intents = [str(row["expected_intent"]) for row in predictions]
    if resolution == "intent":
        observed = [str(row["classification_intent"]) for row in predictions]
        metrics = _class_metrics(
            expected_intents,
            observed,
            all_classes=(*INTENTS, REJECT),
        )
        primary_correct = [
            row["expected_intent"] == row["classification_intent"] for row in predictions
        ]
        misfire = [
            row["expected_intent"] != REJECT
            and any(
                intent != row["expected_intent"]
                for intent in row["all_emitted_intents"]
            )
            for row in predictions
        ]
    else:
        expected = [
            REJECT if row["expected_intent"] == REJECT else str(row["expected_command"])
            for row in predictions
        ]
        observed = []
        primary_correct = []
        for row in predictions:
            if row["classification_intent"] == REJECT:
                outcome = REJECT
            else:
                outcome = str(row["primary_command"]) or (
                    f"UNKNOWN_INTENT:{row['classification_intent']}"
                )
                if _slot_key(row["primary_slots"]) != _slot_key(row["expected_slots"]):
                    outcome = f"{outcome} [wrong slots]"
            observed.append(outcome)
            primary_correct.append(_joint_command_correct(row))
        metrics = _class_metrics(expected, observed)
        misfire = []
        for row in predictions:
            wrong_fire = False
            if row["expected_intent"] != REJECT:
                for command, slots in zip(
                    row["all_emitted_commands"], row["all_emitted_slots"]
                ):
                    if command != row["expected_command"] or _slot_key(slots) != _slot_key(
                        row["expected_slots"]
                    ):
                        wrong_fire = True
                        break
            misfire.append(wrong_fire)
    total = len(predictions)
    oos = [row for row in predictions if row["expected_intent"] == REJECT]
    in_scope = [row for row in predictions if row["expected_intent"] != REJECT]
    false_accepts = sum(row["command_fire_count"] > 0 for row in oos)
    false_rejects = sum(
        row["classification_intent"] == REJECT for row in in_scope
    )
    misfires = sum(misfire)
    in_scope_correct = sum(
        correct
        for correct, row in zip(primary_correct, predictions)
        if row["expected_intent"] != REJECT
    )
    metrics.update(
        {
            "n_all": total,
            "n_in_scope": len(in_scope),
            "n_oos": len(oos),
            "false_accepts": false_accepts,
            "false_accept_rate": _rate(false_accepts, len(oos)),
            "false_accept_wilson_95": _wilson(false_accepts, len(oos)),
            "false_rejects": false_rejects,
            "false_reject_rate": _rate(false_rejects, len(in_scope)),
            "misfires": misfires,
            "misfire_rate": _rate(misfires, len(in_scope)),
            "in_scope_correct": in_scope_correct,
            "in_scope_accuracy": _rate(in_scope_correct, len(in_scope)),
        }
    )
    grouped: dict[str, list[bool]] = defaultdict(list)
    for correct, row in zip(primary_correct, predictions):
        speaker = str(row.get("speaker_id") or "").strip()
        if speaker:
            grouped[speaker].append(bool(correct))
    if len(grouped) >= 2:
        generator = random.Random(0)
        speaker_ids = sorted(grouped)
        bootstrap = []
        for _ in range(2000):
            sampled = [generator.choice(speaker_ids) for _ in speaker_ids]
            values = [item for speaker in sampled for item in grouped[speaker]]
            bootstrap.append(statistics.fmean(values))
        bootstrap.sort()
        metrics["speaker_grouped_bootstrap_95"] = (
            bootstrap[49],
            bootstrap[1949],
        )
        metrics["speaker_groups"] = len(speaker_ids)
    else:
        metrics["speaker_grouped_bootstrap_95"] = None
        metrics["speaker_groups"] = len(grouped)
    return metrics


def _edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, left in enumerate(reference, start=1):
        current = [index]
        for other_index, right in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[other_index] + 1,
                    previous[other_index - 1] + int(left != right),
                )
            )
        previous = current
    return previous[-1]


def _numeric_slot(value: str, kind: str) -> tuple[float, str] | None:
    text = _normalized_text(value)
    numbers = [
        float(token)
        for token in text.replace(":", " ").split()
        if token.replace(".", "", 1).isdigit()
    ]
    if not numbers:
        return None
    if kind == "seconds":
        amount = numbers[0] * (60.0 if "minute" in text else 1.0)
        return amount, "seconds"
    if kind == "percent":
        return numbers[0], "percent"
    if kind == "degrees":
        return numbers[0], "degrees"
    if kind == "clock_minutes":
        hour = numbers[0] % 12
        minute = numbers[1] if len(numbers) > 1 else 0.0
        if "pm" in text:
            hour += 12
        return hour * 60 + minute, "minutes"
    return None


def _slot_metrics(
    predictions: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    slot_schema = schema.get("slots", {})
    rows: list[dict[str, Any]] = []
    joint_eligible = [
        row
        for row in predictions
        if row["expected_intent"] != REJECT and row["expected_slots"]
    ]
    joint_correct = sum(
        row["classification_intent"] == row["expected_intent"]
        and _slot_key(row["primary_slots"]) == _slot_key(row["expected_slots"])
        for row in joint_eligible
    )
    pooled_exact: list[bool] = []
    pooled_distances: list[float] = []
    numeric_by_unit: dict[str, list[float]] = defaultdict(list)
    for intent, specification in slot_schema.items():
        eligible = [
            row
            for row in predictions
            if row["expected_intent"] == intent
            and row["classification_intent"] == intent
            and row["expected_slots"]
        ]
        valid = 0
        invalid = 0
        exact = 0
        absolute_errors: list[float] = []
        relative_errors: list[float] = []
        zero_references = 0
        character_distances: list[float] = []
        kind = str(specification.get("numeric_kind") or "")
        for row in eligible:
            expected = _normalized_text(row["expected_slots"].get("value", ""))
            predicted = _normalized_text(row["primary_slots"].get("value", ""))
            matched = expected == predicted and bool(expected)
            exact += int(matched)
            pooled_exact.append(matched)
            distance = _edit_distance(expected, predicted)
            normalized_distance = distance / max(len(expected), len(predicted), 1)
            character_distances.append(normalized_distance)
            pooled_distances.append(normalized_distance)
            if kind:
                expected_number = _numeric_slot(expected, kind)
                predicted_number = _numeric_slot(predicted, kind)
                if expected_number is None or predicted_number is None:
                    invalid += 1
                    continue
                valid += 1
                error = abs(predicted_number[0] - expected_number[0])
                if kind == "clock_minutes":
                    error = min(error, 1440.0 - error)
                absolute_errors.append(error)
                numeric_by_unit[expected_number[1]].append(error)
                if expected_number[0] == 0:
                    zero_references += 1
                else:
                    relative_errors.append(error / abs(expected_number[0]))
        rows.append(
            {
                "intent": intent,
                "eligible_correct_intent": len(eligible),
                "valid_numeric": valid,
                "invalid_numeric": invalid,
                "exact_match_rate": _rate(exact, len(eligible)),
                "mae": statistics.fmean(absolute_errors) if absolute_errors else None,
                "mae_unit": (
                    specification.get("unit")
                    if kind
                    else "N/A — categorical slot"
                ),
                "mean_relative_error": (
                    statistics.fmean(relative_errors) if relative_errors else None
                ),
                "zero_reference_count": zero_references,
                "phonetic_distance": "N/A — no versioned phonemizer configured",
                "mean_normalized_character_distance": (
                    statistics.fmean(character_distances)
                    if character_distances
                    else None
                ),
            }
        )
    rows.append(
        {
            "intent": "all",
            "eligible_correct_intent": len(pooled_exact),
            "valid_numeric": sum(row["valid_numeric"] for row in rows),
            "invalid_numeric": sum(row["invalid_numeric"] for row in rows),
            "exact_match_rate": (
                statistics.fmean(pooled_exact) if pooled_exact else None
            ),
            "mae": None,
            "mae_unit": "N/A — mixed units",
            "mean_relative_error": None,
            "zero_reference_count": sum(row["zero_reference_count"] for row in rows),
            "phonetic_distance": "N/A — no versioned phonemizer configured",
            "mean_normalized_character_distance": (
                statistics.fmean(pooled_distances) if pooled_distances else None
            ),
            "mae_by_unit": {
                unit: statistics.fmean(values)
                for unit, values in sorted(numeric_by_unit.items())
            },
        }
    )
    return rows, {
        "joint_intent_slot_correct": joint_correct,
        "joint_intent_slot_total": len(joint_eligible),
        "joint_intent_slot_accuracy": _rate(joint_correct, len(joint_eligible)),
    }


def _confusions(
    predictions: Sequence[Mapping[str, Any]], *, resolution: str
) -> list[dict[str, Any]]:
    pairs: Counter[tuple[str, str]] = Counter()
    supports: Counter[str] = Counter()
    representatives: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in predictions:
        if resolution == "intent":
            expected = str(row["expected_intent"])
            observed = str(row["primary_intent"])
        else:
            expected = (
                REJECT
                if row["expected_intent"] == REJECT
                else str(row["expected_command"])
            )
            observed = (
                str(row["primary_intent"])
                if row["primary_intent"] in {REJECT, NO_RESPONSE}
                else (
                    str(row["primary_command"])
                    or f"UNKNOWN_INTENT:{row['primary_intent']}"
                )
            )
            if (
                observed not in {REJECT, NO_RESPONSE}
                and _slot_key(row["primary_slots"]) != _slot_key(row["expected_slots"])
            ):
                observed += " [wrong slots]"
        supports[expected] += 1
        if expected != observed:
            pairs[(expected, observed)] += 1
            if len(representatives[(expected, observed)]) < 5:
                representatives[(expected, observed)].append(str(row["clip_id"]))
    return [
        {
            "expected": expected,
            "got": observed,
            "count": count,
            "expected_support": supports[expected],
            "error_rate": count / supports[expected],
            "representative_clip_ids": representatives[(expected, observed)],
        }
        for (expected, observed), count in sorted(
            pairs.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )[:10]
    ]


def _summary(values: Sequence[float], *, latency: bool = False) -> dict[str, Any]:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return {
            "n": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    def percentile(fraction: float) -> float:
        position = (len(clean) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return clean[lower]
        return clean[lower] + (clean[upper] - clean[lower]) * (position - lower)

    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "p50": percentile(0.50) if latency else None,
        "p95": percentile(0.95),
        "p99": percentile(0.99) if latency else None,
        "max": max(clean),
    }


def _telemetry_metrics(
    predictions: Sequence[Mapping[str, Any]],
    telemetry_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    definitions = [
        ("Response latency", "ms", [row["primary_latency_ms"] for row in predictions], True),
        ("Inference time", "ms", [row["infer_ms"] for row in predictions], True),
        (
            "Real-time factor",
            "ratio",
            [
                row["infer_ms"] / row["audio_ms"]
                for row in predictions
                if row["infer_ms"] is not None
                and row["audio_ms"] is not None
                and row["audio_ms"] > 0
            ],
            True,
        ),
        (
            "CPU temperature",
            "°C",
            [_as_float(row.get("cpu_temperature_c")) for row in telemetry_rows],
            False,
        ),
        (
            "Observed CPU clock",
            "MHz",
            [_as_float(row.get("cpu_clock_mhz")) for row in telemetry_rows],
            False,
        ),
        (
            "Load average 1 min",
            "load",
            [_as_float(row.get("load_1m")) for row in telemetry_rows],
            False,
        ),
        (
            "Load average 5 min",
            "load",
            [_as_float(row.get("load_5m")) for row in telemetry_rows],
            False,
        ),
        (
            "Load average 15 min",
            "load",
            [_as_float(row.get("load_15m")) for row in telemetry_rows],
            False,
        ),
        (
            "Whole-Pi CPU utilization",
            "%",
            [_as_float(row.get("system_cpu_percent")) for row in telemetry_rows],
            False,
        ),
        (
            "Assistant process-tree CPU utilization",
            "% single-core equivalents",
            [_as_float(row.get("assistant_cpu_percent")) for row in telemetry_rows],
            False,
        ),
        (
            "Whole-Pi RAM used",
            "MiB",
            [_as_float(row.get("system_ram_used_mib")) for row in telemetry_rows],
            False,
        ),
        (
            "Assistant process-tree RSS",
            "MiB",
            [_as_float(row.get("assistant_rss_mib")) for row in telemetry_rows],
            False,
        ),
    ]
    rows = []
    for name, unit, raw_values, latency in definitions:
        values = [value for value in raw_values if value is not None]
        row = {"metric": name, "unit": unit, **_summary(values, latency=latency)}
        if not values:
            row["unavailable_reason"] = (
                "Pi telemetry not emitted"
                if name != "Response latency"
                else "no structured command-stage decisions with common-clock timestamps"
            )
        rows.append(row)
    for name, key, unit in (
        ("Assistant CPU-seconds per audio second", "cpu_seconds_per_audio_second", "ratio"),
        ("Assistant CPU share of test wall time", "assistant_cpu_share_percent", "%"),
        ("Test duration", "test_duration_seconds", "s"),
    ):
        value = _as_float(metadata.get(key))
        if value is None and key == "cpu_seconds_per_audio_second":
            cpu_seconds = _as_float(metadata.get("assistant_cpu_seconds"))
            audio_seconds = _as_float(metadata.get("annotated_audio_seconds"))
            if cpu_seconds is not None and audio_seconds is not None and audio_seconds > 0:
                value = cpu_seconds / audio_seconds
        if value is None and key == "assistant_cpu_share_percent":
            cpu_seconds = _as_float(metadata.get("assistant_cpu_seconds"))
            wall_seconds = _as_float(metadata.get("test_duration_seconds"))
            if cpu_seconds is not None and wall_seconds is not None and wall_seconds > 0:
                value = 100.0 * cpu_seconds / wall_seconds
        rows.append(
            {
                "metric": name,
                "unit": unit,
                "n": 1 if value is not None else 0,
                "mean": value,
                "p50": None,
                "p95": None,
                "p99": None,
                "max": value,
                **(
                    {}
                    if value is not None
                    else {"unavailable_reason": "not supplied by Pi run metadata"}
                ),
            }
        )
    throttles = [
        str(row.get("throttling_flags") or "").strip()
        for row in telemetry_rows
        if str(row.get("throttling_flags") or "").strip()
    ]
    rows.append(
        {
            "metric": "Throttling flags",
            "unit": "bitmask",
            "n": len(throttles),
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "observed_values": sorted(set(throttles)),
            **(
                {}
                if throttles
                else {"unavailable_reason": "Pi throttling telemetry not emitted"}
            ),
        }
    )
    parsed_throttles = []
    for value in throttles:
        try:
            parsed_throttles.append(int(value, 0))
        except ValueError:
            parsed_throttles = []
            break
    if parsed_throttles:
        combined = 0
        for value in parsed_throttles:
            combined |= value
        for bit in range(max(1, combined.bit_length())):
            mask = 1 << bit
            if not combined & mask:
                continue
            active = [100.0 if value & mask else 0.0 for value in parsed_throttles]
            rows.append(
                {
                    "metric": f"Throttling flag 0x{mask:x} active fraction",
                    "unit": "%",
                    **_summary(active),
                }
            )
        rows[-(combined.bit_count() + 1)]["combined_bitmask"] = f"0x{combined:x}"
    return rows


def _source_breakdown(
    predictions: Sequence[Mapping[str, Any]], command_available: bool
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        source = _normalized_text(
            row.get("source_type") or row.get("source") or "unknown"
        )
        if "synthetic" in source:
            source = "synthetic"
        elif source in {"real", "real_voice", "human"}:
            source = "real"
        elif not source:
            source = "unknown"
        groups[source].append(row)
    result = []
    for source, rows in sorted(groups.items()):
        correct = sum(
            row["expected_intent"] == row["classification_intent"] for row in rows
        )
        command_correct = sum(_joint_command_correct(row) for row in rows)
        result.append(
            {
                "source": source,
                "intent_correct": correct,
                "total": len(rows),
                "intent_accuracy": _rate(correct, len(rows)),
                "intent_wilson_95": _wilson(correct, len(rows)),
                "command_correct": command_correct if command_available else None,
                "command_accuracy": (
                    _rate(command_correct, len(rows)) if command_available else None
                ),
            }
        )
    return result


def score_run(
    manifest_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    telemetry_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    aliases = {
        _normalized_text(key).replace(" ", "_"): str(value)
        for key, value in dict(schema.get("intent_aliases", {})).items()
    }
    predictions = associate_trials(
        manifest_rows,
        event_rows,
        response_window_ms=float(metadata.get("response_window_ms", 3000)),
        aliases=aliases,
    )
    variation_to_intent = {
        str(variation): str(intent)
        for intent, variations in schema.get("variations", {}).items()
        for variation in variations
    }
    invalid_truth = [
        str(row["clip_id"])
        for row in predictions
        if row["expected_intent"] != REJECT
        and variation_to_intent.get(str(row["expected_command"]))
        != row["expected_intent"]
    ]
    if invalid_truth:
        raise ValueError(
            "Ground-truth variation is missing from the versioned schema or mapped "
            f"to another intent for clips: {invalid_truth[:5]}"
        )
    command_available, command_reason = _command_available(predictions)
    intent_metrics = _resolution_metrics(predictions, resolution="intent")
    command_metrics = (
        _resolution_metrics(predictions, resolution="command")
        if command_available
        else None
    )
    slot_rows, slot_summary = _slot_metrics(predictions, schema)
    unknown: dict[str, dict[str, Any]] = {}
    for row in predictions:
        for intent in row["all_emitted_intents"]:
            if intent not in {*INTENTS, REJECT}:
                item = unknown.setdefault(intent, {"count": 0, "clip_ids": []})
                item["count"] += 1
                if row["clip_id"] not in item["clip_ids"]:
                    item["clip_ids"].append(row["clip_id"])
    wake_eligible = [row for row in predictions if _as_bool(row.get("wake_eligible"))]
    wake_detected = sum(row["wake_detected"] for row in wake_eligible)
    pipeline = {
        "response_count": sum(row["command_stage_responded"] for row in predictions),
        "response_rate": _rate(
            sum(row["command_stage_responded"] for row in predictions), len(predictions)
        ),
        "no_response_count": sum(
            row["primary_intent"] == NO_RESPONSE for row in predictions
        ),
        "total_extra_fires": sum(row["extra_fire_count"] for row in predictions),
        "clips_with_multiple_fires": sum(
            row["command_fire_count"] > 1 for row in predictions
        ),
        "premature_fires": sum(row["premature_fire_count"] for row in predictions),
        "late_fires": sum(row["late_fire_count"] for row in predictions),
        "wake_detected": wake_detected,
        "wake_eligible": len(wake_eligible),
        "wake_detection_rate": _rate(wake_detected, len(wake_eligible)),
        "evaluation_mode": metadata.get("evaluation_mode", "not supplied"),
    }
    tables = {
        "classification": [
            {
                "metric": key,
                "intent": intent_metrics.get(key),
                "command": command_metrics.get(key) if command_metrics else None,
            }
            for key in (
                "accuracy",
                "balanced_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "macro_f2",
                "false_accept_rate",
                "false_reject_rate",
                "misfire_rate",
                "in_scope_accuracy",
            )
        ],
        "slots": slot_rows,
        "intent_confusions": _confusions(predictions, resolution="intent"),
        "command_confusions": (
            _confusions(predictions, resolution="command")
            if command_available
            else []
        ),
        "per_intent": [
            {"intent": name, **intent_metrics["per_class"][name]}
            for name in (*INTENTS, REJECT)
        ],
        "telemetry": _telemetry_metrics(predictions, telemetry_rows, metadata),
        "source_breakdown": _source_breakdown(predictions, command_available),
    }
    metrics = {
        "report_schema_version": 1,
        "status": metadata.get("status", "completed"),
        "scoring": {
            "response_window_ms": float(metadata.get("response_window_ms", 3000)),
            "primary_rule": "first structured decision at 0..response_window_ms",
            "no_response_classification_mapping": REJECT,
            "confidence_interval": "Wilson score, 95%, clip-independence assumption",
            "macro_classes": "ground-truth classes represented in this test",
            "zero_division": "represented class with no predicted positives receives precision=0",
            "extra_fire_note": (
                "A correct primary prediction can still be an operational misfire "
                "when a later emitted command is wrong."
            ),
        },
        "counts": {
            "n_all": len(predictions),
            "n_in_scope": intent_metrics["n_in_scope"],
            "n_oos": intent_metrics["n_oos"],
        },
        "intent": intent_metrics,
        "command": command_metrics,
        "command_metrics_available": command_available,
        "command_metrics_unavailable_reason": command_reason or None,
        "pipeline": pipeline,
        "unknown_intents": unknown,
        "slot_summary": slot_summary,
        "source_breakdown": tables["source_breakdown"],
        "pi": {
            "specs": dict(metadata.get("pi", {})),
            "telemetry_method": dict(metadata.get("telemetry_method", {})),
            "onnx": dict(metadata.get("onnx", {})),
        },
    }
    return metrics, predictions, tables


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str] = ()) -> None:
    materialized = list(rows)
    fieldnames = list(fields) or list(
        dict.fromkeys(key for row in materialized for key in row)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in materialized:
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else value
                        for key, value in row.items()
                    }
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _format(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value * 100:.2f}%" if percent else f"{value:.4f}"
    return str(value)


def _metric_cell(metrics: Mapping[str, Any] | None, key: str) -> str:
    if metrics is None:
        return "N/A — command variation identity unavailable"
    value = metrics.get(key)
    if key == "accuracy" and value is not None:
        interval = metrics.get("accuracy_wilson_95")
        suffix = (
            f", 95% CI {_format(interval[0], percent=True)}–"
            f"{_format(interval[1], percent=True)}"
            if interval
            else ""
        )
        grouped = metrics.get("speaker_grouped_bootstrap_95")
        grouped_suffix = (
            f"; speaker bootstrap {_format(grouped[0], percent=True)}–"
            f"{_format(grouped[1], percent=True)}"
            if grouped
            else ""
        )
        return (
            f"{_format(value, percent=True)} "
            f"({metrics['correct']}/{metrics['total']}{suffix}{grouped_suffix})"
        )
    if key == "false_accept_rate":
        interval = metrics.get("false_accept_wilson_95")
        suffix = (
            f", 95% CI {_format(interval[0], percent=True)}–"
            f"{_format(interval[1], percent=True)}"
            if interval
            else ""
        )
        return (
            "N/A — no OOS clips"
            if value is None
            else f"{_format(value, percent=True)} "
            f"({metrics['false_accepts']}/{metrics['n_oos']}{suffix})"
        )
    count_keys = {
        "false_reject_rate": ("false_rejects", "n_in_scope"),
        "misfire_rate": ("misfires", "n_in_scope"),
        "in_scope_accuracy": ("in_scope_correct", "n_in_scope"),
    }
    if key in count_keys:
        numerator, denominator = count_keys[key]
        return (
            f"{_format(value, percent=True)} "
            f"({metrics[numerator]}/{metrics[denominator]})"
            if value is not None
            else "N/A — zero denominator"
        )
    return _format(value, percent=True)


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def render_report(
    metrics: Mapping[str, Any],
    tables: Mapping[str, list[dict[str, Any]]],
    metadata: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> str:
    intent = metrics.get("intent")
    command = metrics.get("command")
    pending_reason = str(
        metadata.get("unavailable_reason")
        or "no held-out Raspberry Pi event log and telemetry were supplied"
    )
    na = f"N/A — {pending_reason}"
    if intent is None:
        intent = command = None
    lines = ["# 1. Header", ""]
    header_rows = [
        ("Student", metadata.get("student_name") or "not supplied"),
        ("Student identifier", metadata.get("student_identifier") or "not supplied"),
        ("Test status", metadata.get("status", "completed")),
        ("Test date/time", metadata.get("test_datetime") or f"N/A — {pending_reason}"),
        ("Timezone", metadata.get("timezone") or "N/A — not supplied"),
        ("Wake word", "ateko — connected “a-teh-ko”"),
        ("Supported schema", "19 intents / 93 command variations"),
        (
            "Represented in test",
            (
                f"{len(intent['represented_classes'])} intent/OOS classes / "
                f"{len(command['represented_classes']) if command else 'N/A'} command/OOS classes"
                if intent
                else na
            ),
        ),
        ("Audio/playback connection", metadata.get("audio_connection") or na),
        ("Pi control/logging connection", metadata.get("control_connection") or na),
        ("Holdout source", metadata.get("holdout_source") or na),
        ("Dataset revision", metadata.get("dataset_revision") or na),
        ("Split", metadata.get("split") or na),
        ("Manifest SHA-256", metadata.get("manifest_sha256") or na),
        ("Clips / speakers", metadata.get("clip_speaker_summary") or na),
        ("Real / synthetic", metadata.get("source_summary") or na),
        ("Run ID", metadata.get("run_id") or "not supplied"),
        ("Code revision", metadata.get("code_revision") or "not supplied"),
        ("Checkpoint SHA-256", metadata.get("checkpoint_sha256") or "not supplied"),
        ("Grammar version", metadata.get("grammar_version") or schema.get("version") or "not supplied"),
        ("Acceptance settings", json.dumps(metadata.get("acceptance_settings", {}), sort_keys=True)),
    ]
    lines.extend(_markdown_table(("Field", "Value"), header_rows))

    lines.extend(["", "# 2. Classification", ""])
    metric_names = (
        ("Accuracy", "accuracy"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("Macro precision", "macro_precision"),
        ("Macro recall", "macro_recall"),
        ("Macro F1", "macro_f1"),
        ("Macro F2", "macro_f2"),
        ("False accept rate", "false_accept_rate"),
        ("False reject rate", "false_reject_rate"),
        ("Misfire rate", "misfire_rate"),
        ("In-scope-only accuracy", "in_scope_accuracy"),
    )
    lines.extend(
        _markdown_table(
            ("Metric", "19 intents", "93 commands"),
            (
                (
                    label,
                    _metric_cell(intent, key) if intent else na,
                    _metric_cell(command, key) if command else (
                        f"N/A — {metrics.get('command_metrics_unavailable_reason') or pending_reason}"
                    ),
                )
                for label, key in metric_names
            ),
        )
    )
    lines.append(
        "\nPrimary accuracy includes all scored clips, including OOS. Macro metrics "
        "average per-class scores over represented ground-truth classes and include "
        "REJECT when represented. F2 = 5PR/(4P+R). Wilson intervals assume independent "
        "clips; speaker-grouped bootstrap is N/A unless enough speaker groups are supplied."
    )
    pipeline = metrics.get("pipeline", {})
    if pipeline:
        lines.extend(
            [
                "",
                (
                    "**Pipeline:** response "
                    f"{_format(pipeline.get('response_rate'), percent=True)} "
                    f"({pipeline.get('response_count')}/{metrics['counts']['n_all']}), "
                    f"NO_RESPONSE={pipeline.get('no_response_count')}, "
                    f"extra fires={pipeline.get('total_extra_fires')}, "
                    f"clips with multiple fires={pipeline.get('clips_with_multiple_fires')}, "
                    f"premature fires={pipeline.get('premature_fires')}, "
                    f"late fires={pipeline.get('late_fires')}; wake detection="
                    + (
                        f"{pipeline.get('wake_detected')}/{pipeline.get('wake_eligible')} "
                        f"({_format(pipeline.get('wake_detection_rate'), percent=True)})"
                        if pipeline.get("wake_eligible")
                        else "N/A — no wake-eligible trials"
                    )
                    + f". Mode: {pipeline.get('evaluation_mode')}."
                ),
            ]
        )
    else:
        lines.append(f"\n**Pipeline:** {na}.")
    unknown = metrics.get("unknown_intents", {})
    lines.append(
        "\n**Unknown intent names:** "
        + (
            "; ".join(
                f"{name}: {item['count']} ({', '.join(item['clip_ids'])})"
                for name, item in sorted(unknown.items())
            )
            if unknown
            else ("none observed" if intent else na)
        )
    )
    source_rows = tables.get("source_breakdown", [])
    lines.extend(["", "**Real versus synthetic voices**", ""])
    lines.extend(
        _markdown_table(
            ("Source", "Intent accuracy", "93-command accuracy"),
            (
                (
                    row["source"],
                    (
                        f"{_format(row['intent_accuracy'], percent=True)} "
                        f"({row['intent_correct']}/{row['total']})"
                    ),
                    (
                        f"{_format(row['command_accuracy'], percent=True)} "
                        f"({row['command_correct']}/{row['total']})"
                        if row["command_accuracy"] is not None
                        else "N/A — variation identity unavailable"
                    ),
                )
                for row in source_rows
            ),
        )
        if source_rows
        else [na]
    )

    lines.extend(["", "# 3. Slot values", ""])
    slot_rows = tables.get("slots", [])
    if slot_rows:
        lines.extend(
            _markdown_table(
                (
                    "Intent",
                    "Eligible",
                    "Valid numeric",
                    "Invalid numeric",
                    "Exact match",
                    "MAE",
                    "Mean relative error",
                    "Phonetic distance",
                    "Character distance",
                ),
                (
                    (
                        row["intent"],
                        row["eligible_correct_intent"],
                        row["valid_numeric"],
                        row["invalid_numeric"],
                        _format(row["exact_match_rate"], percent=True),
                        (
                            f"{_format(row['mae'])} {row['mae_unit']}"
                            if row["mae"] is not None
                            else row["mae_unit"]
                        ),
                        _format(row["mean_relative_error"], percent=True),
                        row["phonetic_distance"],
                        _format(row["mean_normalized_character_distance"]),
                    )
                    for row in slot_rows
                ),
            )
        )
        slot_summary = metrics["slot_summary"]
        lines.append(
            "\nThis table is conditional on the primary prediction having the correct "
            "intent. Missing/invalid slots fail exact match. Character distance is "
            "normalized Levenshtein distance on case-folded whitespace-normalized slot "
            "strings. Alarm time uses circular minutes since midnight; timer values use "
            "seconds. Overall joint intent/slot accuracy: "
            f"{_format(slot_summary['joint_intent_slot_accuracy'], percent=True)} "
            f"({slot_summary['joint_intent_slot_correct']}/"
            f"{slot_summary['joint_intent_slot_total']})."
        )
    else:
        lines.append(na)

    lines.extend(["", "# 4. Raspberry Pi", "", "## Specs", ""])
    specs = metrics.get("pi", {}).get("specs", {})
    spec_fields = (
        "model",
        "hostname",
        "logical_cores",
        "cpu_identification",
        "configured_max_clock_mhz",
        "observed_max_clock_mhz",
        "ram_mib",
        "os",
        "kernel",
        "python",
        "packages",
        "microphone",
        "runtime",
        "pids",
    )
    lines.extend(
        _markdown_table(
            ("Field", "Value"),
            (
                (
                    field,
                    json.dumps(specs[field], sort_keys=True)
                    if isinstance(specs.get(field), (dict, list))
                    else specs.get(field) or na,
                )
                for field in spec_fields
            ),
        )
    )
    lines.extend(["", "## Metrics", ""])
    telemetry = tables.get("telemetry", [])
    lines.extend(
        _markdown_table(
            ("Metric", "Unit", "N", "Mean", "P50", "P95", "P99", "Max"),
            (
                (
                    row["metric"],
                    row["unit"],
                    row["n"],
                    _format(row["mean"])
                    if row["mean"] is not None
                    else f"N/A — {row.get('unavailable_reason', 'not a numeric distribution')}",
                    _format(row["p50"]),
                    _format(row["p95"]),
                    _format(row["p99"]),
                    _format(row["max"]),
                )
                for row in telemetry
            ),
        )
    )
    method = metrics.get("pi", {}).get("telemetry_method", {})
    lines.append(
        "\nMeasurement method: "
        + (json.dumps(method, sort_keys=True) if method else na)
        + ". Latency is command-end to first structured command-stage decision; "
        "timeouts are censored, not zero. Load averages are distinct from CPU "
        "utilization. CPU share uses single-core equivalents and may exceed 100%."
    )
    onnx = metrics.get("pi", {}).get("onnx", {})
    lines.append(
        "\nONNX model metrics: "
        + (json.dumps(onnx, sort_keys=True) if onnx else "N/A — no ONNX artifact supplied")
    )

    lines.extend(["", "# 5. Most frequent confusions", "", "## Intent level", ""])
    intent_confusions = tables.get("intent_confusions", [])
    lines.extend(
        _markdown_table(
            ("Expected → got", "Count", "Expected support", "Error rate", "Clip IDs"),
            (
                (
                    f"{row['expected']} → {row['got']}",
                    row["count"],
                    row["expected_support"],
                    _format(row["error_rate"], percent=True),
                    ", ".join(row["representative_clip_ids"]),
                )
                for row in intent_confusions
            ),
        )
        if intent_confusions
        else [("No off-diagonal errors observed." if intent else na)]
    )
    lines.extend(["", "## 93-command level", ""])
    command_confusions = tables.get("command_confusions", [])
    lines.extend(
        _markdown_table(
            ("Expected → got", "Count", "Expected support", "Error rate", "Clip IDs"),
            (
                (
                    f"{row['expected']} → {row['got']}",
                    row["count"],
                    row["expected_support"],
                    _format(row["error_rate"], percent=True),
                    ", ".join(row["representative_clip_ids"]),
                )
                for row in command_confusions
            ),
        )
        if command_confusions
        else [
            (
                "No off-diagonal errors observed."
                if command
                else f"N/A — {metrics.get('command_metrics_unavailable_reason') or pending_reason}"
            )
        ]
    )
    lines.append(
        "\nThe classification matrix maps NO_RESPONSE to REJECT. These diagnostic "
        "tables retain NO_RESPONSE, unknown names, wrong slots, late outputs, and "
        "explicit rejection separately."
    )

    lines.extend(["", "# 6. Per-intent scores", ""])
    per_intent = tables.get("per_intent", [])
    lines.extend(
        _markdown_table(
            ("Intent", "Support", "Predicted", "Precision", "Recall", "F1", "F2"),
            (
                (
                    row["intent"],
                    row["support"],
                    row["predicted"],
                    _format(row["precision"]),
                    _format(row["recall"]),
                    _format(row["f1"]),
                    _format(row["f2"]),
                )
                for row in per_intent
            ),
        )
        if per_intent
        else [na]
    )
    lines.append(
        "\nAll 19 intents plus REJECT are listed. No-support rows are undefined; "
        "represented classes with no predicted positives use precision=0. Unknown "
        "emitted intents remain wrong predictions and are not converted to REJECT."
    )
    return "\n".join(lines) + "\n"


def generate_report(
    *,
    output_dir: Path,
    metadata_path: Path,
    schema_path: Path,
    manifest_path: Path | None = None,
    events_path: Path | None = None,
    telemetry_path: Path | None = None,
) -> dict[str, Any]:
    metadata = _read_json(metadata_path)
    schema = _read_json(schema_path)
    variations = [
        variation
        for values in schema.get("variations", {}).values()
        for variation in values
    ]
    if len(schema.get("intents", [])) != 19 or len(variations) != 93:
        raise ValueError("Report schema must define exactly 19 intents and 93 variations")
    if len(set(variations)) != 93:
        raise ValueError("Command variation IDs must be globally unique")

    manifest = _read_rows(manifest_path)
    events = _read_rows(events_path)
    telemetry = _read_rows(telemetry_path)
    pending = not manifest
    if pending:
        metrics: dict[str, Any] = {
            "report_schema_version": 1,
            "status": metadata.get("status", "not_run"),
            "intent": None,
            "command": None,
            "command_metrics_available": False,
            "command_metrics_unavailable_reason": metadata.get(
                "unavailable_reason",
                "no held-out Raspberry Pi event log was supplied",
            ),
            "counts": {"n_all": 0, "n_in_scope": 0, "n_oos": 0},
            "pipeline": {},
            "unknown_intents": {},
            "slot_summary": {},
            "pi": {
                "specs": dict(metadata.get("pi", {})),
                "telemetry_method": dict(metadata.get("telemetry_method", {})),
                "onnx": dict(metadata.get("onnx", {})),
            },
        }
        predictions: list[dict[str, Any]] = []
        tables = {
            "classification": [],
            "slots": [],
            "intent_confusions": [],
            "command_confusions": [],
            "per_intent": [],
            "telemetry": _telemetry_metrics([], telemetry, metadata),
            "source_breakdown": [],
        }
    else:
        if events_path is None:
            raise ValueError("An event log is required when a manifest is supplied")
        if "manifest_sha256" not in metadata and manifest_path is not None:
            metadata["manifest_sha256"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
        metrics, predictions, tables = score_run(
            manifest, events, telemetry, metadata, schema
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        output_dir / "metrics.json",
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    _write_csv(
        output_dir / "predictions.csv",
        predictions,
        fields=(
            "clip_id",
            "expected_intent",
            "expected_command",
            "expected_slots",
            "primary_intent",
            "classification_intent",
            "primary_command",
            "primary_slots",
            "primary_event_type",
            "primary_latency_ms",
            "command_fire_count",
            "extra_fire_count",
            "premature_fire_count",
            "late_fire_count",
            "wake_detected",
            "source",
            "speaker_id",
        ),
    )
    _write_csv(
        output_dir / "telemetry.csv",
        telemetry,
        fields=(
            "timestamp",
            "cpu_temperature_c",
            "cpu_clock_mhz",
            "load_1m",
            "load_5m",
            "load_15m",
            "system_cpu_percent",
            "assistant_cpu_percent",
            "system_ram_used_mib",
            "assistant_rss_mib",
            "throttling_flags",
        ),
    )
    table_outputs = {
        "classification": (
            "classification.csv",
            ("metric", "intent", "command"),
        ),
        "slots": (
            "slot_values.csv",
            (
                "intent",
                "eligible_correct_intent",
                "valid_numeric",
                "invalid_numeric",
                "exact_match_rate",
                "mae",
                "mae_unit",
                "mean_relative_error",
                "zero_reference_count",
                "phonetic_distance",
                "mean_normalized_character_distance",
            ),
        ),
        "intent_confusions": (
            "intent_confusions.csv",
            ("expected", "got", "count", "expected_support", "error_rate", "representative_clip_ids"),
        ),
        "command_confusions": (
            "command_confusions.csv",
            ("expected", "got", "count", "expected_support", "error_rate", "representative_clip_ids"),
        ),
        "per_intent": (
            "per_intent.csv",
            ("intent", "support", "predicted", "precision", "recall", "f1", "f2"),
        ),
        "telemetry": (
            "telemetry_metrics.csv",
            ("metric", "unit", "n", "mean", "p50", "p95", "p99", "max", "unavailable_reason"),
        ),
        "source_breakdown": (
            "source_breakdown.csv",
            (
                "source",
                "intent_correct",
                "total",
                "intent_accuracy",
                "intent_wilson_95",
                "command_correct",
                "command_accuracy",
            ),
        ),
    }
    for name, rows in tables.items():
        filename, fields = table_outputs[name]
        _write_csv(output_dir / filename, rows, fields=fields)
    references = {}
    for name, path in (
        ("manifest", manifest_path),
        ("events", events_path),
        ("telemetry", telemetry_path),
        ("metadata", metadata_path),
        ("schema", schema_path),
    ):
        references[name] = (
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            if path is not None
            else None
        )
    _atomic_text(
        output_dir / "input_references.json",
        json.dumps(references, indent=2) + "\n",
    )
    _atomic_text(
        output_dir / "report.md",
        render_report(metrics, tables, metadata, schema),
    )
    _atomic_text(
        output_dir / "run_metadata.json",
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config/command_schema_v1.json",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = generate_report(
        output_dir=args.output,
        metadata_path=args.metadata,
        schema_path=args.schema,
        manifest_path=args.manifest,
        events_path=args.events,
        telemetry_path=args.telemetry,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
