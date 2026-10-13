"""Generate explicit command-confusion and rejection analysis from evaluation rows."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from me2.bundle import load_bundle
from me2.vcm.ctc import coerce_grammar_entries, normalize_transcript

REJECTED = "__rejected__"
OOS = "out_of_scope"


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _as_mapping(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        return {str(key): str(item) for key, item in value.items()}
    if value in {None, ""}:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, Mapping):
        return {}
    return {str(key): str(item) for key, item in parsed.items()}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float:
    if not precision or not recall:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def analyze_predictions(
    predictions_path: Path,
    model_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Analyze one fixed prediction artifact without rerunning inference."""

    rows = _read_rows(predictions_path)
    _, bundle = load_bundle(model_path)
    labels = [str(value) for value in bundle["labels"]]
    grammar_by_text: dict[str, dict[str, Any]] = {}
    raw_grammar = bundle.get("grammar", [])
    if raw_grammar:
        for entry in coerce_grammar_entries(raw_grammar):
            grammar_by_text[entry.text] = {
                "intent": entry.intent,
                "variation": entry.variation,
                "slots": dict(entry.slots),
            }

    normalized_rows: list[dict[str, Any]] = []
    supports: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    representatives: dict[tuple[str, str], list[str]] = defaultdict(list)
    slot_rows = 0
    slot_prediction_available_rows = 0
    slot_correct = 0
    greedy_available = 0
    greedy_correct = 0
    trie_available = 0
    trie_correct = 0
    final_correct = 0

    for source_row in rows:
        row = dict(source_row)
        gold = str(row.get("gold_command") or row.get("true_command") or "")
        predicted = str(row.get("predicted_command") or OOS)
        accepted = _as_bool(row.get("accepted"), default=predicted != OOS)
        outcome = predicted if accepted else (OOS if gold == OOS else REJECTED)
        if accepted and not predicted:
            outcome = REJECTED
        supports[gold] += 1
        predicted_counts[outcome] += 1
        confusion[(gold, outcome)] += 1
        audio_id = str(row.get("input_audio") or row.get("audio_id") or "")
        if len(representatives[(gold, outcome)]) < 5:
            representatives[(gold, outcome)].append(audio_id)

        reference_text = str(
            row.get("reference_normalized_text")
            or normalize_transcript(str(row.get("actual_transcript") or ""))
        )
        greedy_text = str(row.get("greedy_text") or "")
        trie_text = str(row.get("decoded_text") or "")
        greedy_semantic = grammar_by_text.get(greedy_text, {}).get("intent")
        trie_semantic = (
            row.get("trie_candidate_intent")
            or grammar_by_text.get(trie_text, {}).get("intent")
        )
        if "greedy_text" in row:
            greedy_available += 1
            greedy_correct += int(greedy_semantic == gold)
        if "decoded_text" in row:
            trie_available += 1
            trie_correct += int(trie_semantic == gold)
        final_correct += int(outcome == gold)

        reference_slot = str(row.get("slot_value") or "").strip()
        predicted_slots = _as_mapping(
            row.get("trie_candidate_slots") or row.get("predicted_slots")
        )
        slot_is_supported = reference_slot.casefold() not in {"", "none", "nan"}
        if slot_is_supported:
            slot_rows += 1
            slot_prediction_available = (
                "trie_candidate_slots" in row or "predicted_slots" in row
            )
            slot_prediction_available_rows += int(slot_prediction_available)
            slot_correct += int(
                slot_prediction_available
                and accepted
                and predicted == gold
                and predicted_slots.get("value") == reference_slot
            )

        error_type = ""
        if outcome != gold:
            if outcome == REJECTED:
                error_type = "rejected_in_scope"
            elif gold == OOS:
                error_type = "out_of_scope_false_accept"
            else:
                error_type = "wrong_command"
        normalized_rows.append(
            {
                **row,
                "accepted": accepted,
                "analysis_outcome": outcome,
                "error_type": error_type,
                "reference_normalized_text": reference_text,
                "greedy_semantic_candidate": greedy_semantic,
                "trie_semantic_candidate": trie_semantic,
                "final_semantic_correct": outcome == gold,
                "slot_supported": slot_is_supported,
                "joint_intent_slot_correct": (
                    accepted
                    and predicted == gold
                    and (
                        not slot_is_supported
                        or predicted_slots.get("value") == reference_slot
                    )
                ),
                "model_version": row.get(
                    "model_version", bundle.get("model_version", "")
                ),
                "grammar_version": row.get(
                    "grammar_version", bundle.get("grammar_version", "")
                ),
            }
        )

    outcome_labels = [
        *labels,
        *(
            [REJECTED]
            if any(outcome == REJECTED for _, outcome in confusion)
            else []
        ),
    ]
    per_command: list[dict[str, Any]] = []
    for command in labels:
        support = supports[command]
        true_positives = confusion[(command, command)]
        predicted_count = predicted_counts[command]
        precision = _safe_rate(true_positives, predicted_count)
        recall = _safe_rate(true_positives, support)
        wrong = sum(
            count
            for (gold, outcome), count in confusion.items()
            if gold == command and outcome not in {command, REJECTED}
        )
        rejected = confusion[(command, REJECTED)] if command != OOS else 0
        per_command.append(
            {
                "command": command,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
                "correct_accepts": true_positives,
                "wrong_command_predictions": wrong,
                "wrong_command_rate": _safe_rate(wrong, support),
                "rejected_in_scope": rejected,
                "rejection_rate": _safe_rate(rejected, support),
                "total_failure_rate": _safe_rate(support - true_positives, support),
                "small_support": support < 20,
            }
        )

    raw_matrix = [
        {
            "true_command": gold,
            **{outcome: confusion[(gold, outcome)] for outcome in outcome_labels},
        }
        for gold in labels
    ]
    normalized_matrix = [
        {
            "true_command": gold,
            **{
                outcome: (
                    confusion[(gold, outcome)] / supports[gold]
                    if supports[gold]
                    else 0.0
                )
                for outcome in outcome_labels
            },
        }
        for gold in labels
    ]
    pair_rows = []
    for (gold, outcome), count in confusion.items():
        if gold == outcome:
            continue
        pair_rows.append(
            {
                "true_command": gold,
                "predicted_outcome": outcome,
                "count": count,
                "rate_within_true_command": _safe_rate(count, supports[gold]),
                "true_command_support": supports[gold],
                "small_support": supports[gold] < 20,
                "representative_audio_ids": json.dumps(
                    representatives[(gold, outcome)], ensure_ascii=False
                ),
            }
        )

    oos_count = supports[OOS]
    oos_false_accepts = sum(
        count
        for (gold, outcome), count in confusion.items()
        if gold == OOS and outcome != OOS
    )
    in_scope_count = sum(
        support for command, support in supports.items() if command != OOS
    )
    in_scope_rejects = sum(
        count
        for (gold, outcome), count in confusion.items()
        if gold != OOS and outcome == REJECTED
    )
    summary = {
        "predictions_path": str(predictions_path.resolve()),
        "model_path": str(model_path.resolve()),
        "model_version": bundle.get("model_version", ""),
        "grammar_version": bundle.get("grammar_version", ""),
        "rows": len(rows),
        "labels": labels,
        "final_accuracy": _safe_rate(final_correct, len(rows)),
        "out_of_scope_count": oos_count,
        "out_of_scope_false_accepts": oos_false_accepts,
        "out_of_scope_false_accept_rate": _safe_rate(
            oos_false_accepts, oos_count
        ),
        "in_scope_count": in_scope_count,
        "in_scope_false_rejects": in_scope_rejects,
        "in_scope_false_reject_rate": _safe_rate(
            in_scope_rejects, in_scope_count
        ),
        "supported_slot_rows": slot_rows,
        "slot_prediction_available_rows": slot_prediction_available_rows,
        "supported_slot_joint_accuracy": (
            _safe_rate(slot_correct, slot_rows)
            if slot_prediction_available_rows == slot_rows and slot_rows
            else None
        ),
        "greedy_semantic_rows": greedy_available,
        "greedy_semantic_accuracy": _safe_rate(greedy_correct, greedy_available),
        "trie_candidate_semantic_rows": trie_available,
        "trie_candidate_semantic_accuracy": _safe_rate(
            trie_correct, trie_available
        ),
        "final_semantic_rows": len(rows),
        "final_semantic_accuracy": _safe_rate(final_correct, len(rows)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "per_command.csv", per_command)
    _write_csv(output_dir / "confusion_raw.csv", raw_matrix)
    _write_csv(output_dir / "confusion_row_normalized.csv", normalized_matrix)
    _write_csv(
        output_dir / "confusion_pairs_by_count.csv",
        sorted(
            pair_rows,
            key=lambda item: (
                -int(item["count"]),
                str(item["true_command"]),
                str(item["predicted_outcome"]),
            ),
        ),
    )
    _write_csv(
        output_dir / "confusion_pairs_by_rate.csv",
        sorted(
            pair_rows,
            key=lambda item: (
                -(float(item["rate_within_true_command"]) or 0.0),
                -int(item["count"]),
                str(item["true_command"]),
            ),
        ),
    )
    error_rows = [row for row in normalized_rows if row["error_type"]]
    _write_csv(output_dir / "errors.csv", error_rows)
    with (output_dir / "errors.jsonl").open("w", encoding="utf-8") as stream:
        for row in error_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    top_pairs = sorted(pair_rows, key=lambda item: -int(item["count"]))[:15]
    report_lines = [
        "# Voice-command misunderstanding analysis",
        "",
        f"- Rows: {len(rows)}",
        f"- Final semantic accuracy: {summary['final_semantic_accuracy']}",
        (
            "- OOS false acceptance: "
            f"{oos_false_accepts}/{oos_count} "
            f"({summary['out_of_scope_false_accept_rate']})"
        ),
        (
            "- In-scope rejection: "
            f"{in_scope_rejects}/{in_scope_count} "
            f"({summary['in_scope_false_reject_rate']})"
        ),
        (
            "- Supported-slot joint accuracy: "
            f"{slot_correct}/{slot_rows} "
            f"({summary['supported_slot_joint_accuracy']})"
        ),
        "",
        "## Most frequent true-command → outcome errors",
        "",
    ]
    report_lines.extend(
        (
            f"- {item['true_command']} → {item['predicted_outcome']}: "
            f"{item['count']}/{item['true_command_support']} "
            f"({item['rate_within_true_command']:.4f})"
        )
        for item in top_pairs
    )
    if not top_pairs:
        report_lines.append("- No errors observed.")
    report_lines.extend(
        [
            "",
            (
                "Greedy and trie semantic metrics are reported only when those fields "
                "exist in the source predictions. Decoder scores are diagnostic and are "
                "not interpreted as calibrated correctness probabilities."
            ),
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze_predictions(args.predictions, args.model, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
