"""Observe an already-running ME2 process and summarize safe runtime JSONL logs.

This utility never starts inference, opens an audio device, sends network
requests, or stores audio/transcripts. Run it alongside an operator-approved
Pi session, then review the generated telemetry report.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from me2.io_utils import file_sha256
from me2.logging_utils import config_hash
from me2.runtime_config import load_runtime_defaults

ROOT = Path(__file__).resolve().parents[1]


def _percentile(values: list[float], fraction: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    index = min(len(clean) - 1, math.ceil((len(clean) - 1) * fraction))
    return float(clean[index])


def summarize_runtime_log(
    path: Path,
    *,
    sample_rate: int,
    wake_hop_ms: float,
    stream_chunk_ms: float = 100.0,
) -> dict:
    records: list[dict[str, Any]] = []
    malformed = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(value, dict):
                records.append(value)

    wake_ms: list[float] = []
    command_ms: list[float] = []
    endpoint_latency_ms: list[float] = []
    endpoint_at: dict[str, datetime] = {}
    health: list[dict] = []
    queue_delays_ms: list[float] = []
    feed_p95_ms: list[float] = []
    reasons: Counter[str] = Counter()
    accepted = rejected = wake_events = 0
    clipped_samples = 0
    queue_drop_count = callback_status_count = callback_frames_total = 0
    callback_count = 0
    for record in records:
        event = record.get("event")
        diagnostics = record.get("diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        if event == "wake_window" and _finite(record.get("total_ms")) is not None:
            wake_ms.append(float(record["total_ms"]))
        if event == "stream_health":
            health.append(diagnostics)
            queue_delay = _finite(diagnostics.get("queue_delay_p95_ms"))
            feed_p95 = _finite(diagnostics.get("feed_p95_ms"))
            if queue_delay is not None:
                queue_delays_ms.append(queue_delay)
            if feed_p95 is not None:
                feed_p95_ms.append(feed_p95)
            queue_drop_count = max(
                queue_drop_count, _integer(diagnostics.get("queue_drop_count"))
            )
            callback_status_count = max(
                callback_status_count, _integer(diagnostics.get("callback_status_count"))
            )
            callback_frames_total = max(
                callback_frames_total, _integer(diagnostics.get("callback_frames_total"))
            )
            callback_count = max(callback_count, _integer(diagnostics.get("callback_count")))
        if event == "stream_chunk":
            queue_drop_count = max(queue_drop_count, _integer(diagnostics.get("queue_drop_count")))
            callback_status_count = max(
                callback_status_count, _integer(diagnostics.get("callback_status_count"))
            )
            callback_frames_total = max(
                callback_frames_total, _integer(diagnostics.get("callback_frames_total"))
            )
            callback_count = max(callback_count, _integer(diagnostics.get("callback_count")))
        if event == "capture_activity":
            clipped_samples += _integer(diagnostics.get("clipped_samples"))
        if event == "pipeline_state":
            reason = record.get("reason")
            if isinstance(reason, str):
                reasons[reason] += 1
                if reason == "wake_detected":
                    wake_events += 1
                elif reason == "accepted":
                    accepted += 1
                elif (
                    reason not in {
                        "endpoint_detected", "cooldown_started", "cooldown_complete",
                        "stream_recovered", "stopped", "wake_detected",
                    }
                    and record.get("state") == "ACCEPTED_OR_REJECTED"
                ):
                    rejected += 1
            utterance_id = record.get("utterance_id")
            timestamp = _timestamp(record.get("timestamp"))
            if utterance_id and timestamp and reason == "endpoint_detected":
                endpoint_at[str(utterance_id)] = timestamp
            elif utterance_id and timestamp and record.get("state") == "ACCEPTED_OR_REJECTED":
                started = endpoint_at.get(str(utterance_id))
                if started:
                    endpoint_latency_ms.append(
                        max(0.0, (timestamp - started).total_seconds() * 1000)
                    )
        if event == "command_result":
            value = _finite(diagnostics.get("command_decision_ms"))
            if value is not None:
                command_ms.append(value)

    stream_health_samples = len(health)
    queue_depths = [
        _finite(item.get("queue_depth_mean"))
        for item in health
        if _finite(item.get("queue_depth_mean")) is not None
    ]
    queue_maxes = [
        _finite(item.get("queue_depth_max"))
        for item in health
        if _finite(item.get("queue_depth_max")) is not None
    ]
    queue_capacity = max(
        (_integer(item.get("queue_capacity_chunks")) for item in health), default=0
    )
    wake_p95 = _percentile(wake_ms, .95)
    queue_trend = "not_measured"
    if len(queue_depths) >= 3:
        first = statistics.mean(queue_depths[:3])
        last = statistics.mean(queue_depths[-3:])
        queue_trend = "stable_or_decreasing" if last <= first + 1 else "increasing"
    pace_status = (
        "not_measured" if wake_p95 is None
        else "pass" if wake_p95 <= wake_hop_ms
        else "fail"
    )
    queue_status = "not_measured" if not health else (
        "fail" if queue_drop_count or callback_status_count else
        "pass" if queue_capacity and max(queue_maxes, default=0) < queue_capacity
        else "fail"
    )
    chunk_feed_p95 = max(feed_p95_ms, default=None)
    feed_pace_status = (
        "not_measured" if chunk_feed_p95 is None
        else "pass" if chunk_feed_p95 <= stream_chunk_ms
        else "fail"
    )
    return {
        "log_file": path.name,
        "parsed_records": len(records),
        "malformed_lines": malformed,
        "stream_health_reports": stream_health_samples,
        "callback_frames_total": callback_frames_total,
        "callback_count": callback_count,
        "processed_audio_seconds": callback_frames_total / sample_rate,
        "callback_status_count": callback_status_count,
        "queue_drop_count": queue_drop_count,
        "queue_capacity_chunks": queue_capacity or None,
        "queue_depth_mean_max": max(queue_depths) if queue_depths else None,
        "queue_depth_max_observed": max(queue_maxes) if queue_maxes else None,
        "queue_depth_trend": queue_trend,
        "queue_delay_p95_ms_max_batch": max(queue_delays_ms, default=None),
        "feed_processing_p95_ms_max_batch": chunk_feed_p95,
        "configured_stream_chunk_ms": stream_chunk_ms,
        "chunk_processing_keeps_pace": feed_pace_status,
        "clipped_samples_observed": clipped_samples,
        "wake_window_count": len(wake_ms),
        "wake_window_processing_ms_p50": _percentile(wake_ms, .50),
        "wake_window_processing_ms_p95": wake_p95,
        "configured_wake_hop_ms": wake_hop_ms,
        "wake_processing_keeps_pace": pace_status,
        "command_decision_ms_p50": _percentile(command_ms, .50),
        "command_decision_ms_p95": _percentile(command_ms, .95),
        "endpoint_to_result_ms_p50": _percentile(endpoint_latency_ms, .50),
        "endpoint_to_result_ms_p95": _percentile(endpoint_latency_ms, .95),
        "wake_events": wake_events,
        "accepted_commands": accepted,
        "rejected_commands": rejected,
        "event_reasons": dict(sorted(reasons.items())),
        "queue_health": queue_status,
        "false_activations_per_hour": None,
        "false_activation_note": (
            "Requires a separate annotated continuous-negative trial manifest; "
            "wake windows and clip false accepts are not false activations per hour."
        ),
    }


def _finite(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _integer(value) -> int:
    return value if type(value) is int and value >= 0 else 0


def _timestamp(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _read_float(path: Path) -> float | None:
    try:
        return float(path.read_text().strip()) / 1000
    except (OSError, ValueError):
        return None


def _vcgencmd(command: str) -> str | None:
    try:
        return subprocess.check_output(
            ["vcgencmd", command], text=True, stderr=subprocess.DEVNULL, timeout=2
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect(args) -> dict:
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve()
    runtime = load_runtime_defaults(config_path)
    process = psutil.Process(args.pid)
    process.cpu_percent(None)
    started = time.monotonic()
    cpu_samples: list[float] = []
    rss_peak = process.memory_info().rss
    while time.monotonic() - started < args.duration_seconds:
        time.sleep(min(args.interval_seconds, max(
            .05, args.duration_seconds - (time.monotonic() - started)
        )))
        try:
            cpu_samples.append(process.cpu_percent(None))
            rss_peak = max(rss_peak, process.memory_info().rss)
        except psutil.NoSuchProcess:
            break
    elapsed = time.monotonic() - started
    temperature = _read_float(Path("/sys/class/thermal/thermal_zone0/temp"))
    board_model = None
    try:
        board_model = Path("/proc/device-tree/model").read_text().strip("\x00\n ")
    except OSError:
        pass
    artifacts = {}
    for key in ("wake_model", "vcm_model"):
        value = runtime.get(key)
        if value is None:
            continue
        artifact = Path(value)
        artifacts[key] = {
            "file": artifact.name,
            "sha256": file_sha256(artifact) if artifact.is_file() else None,
            "exists": artifact.is_file(),
        }
    log_path = args.log if args.log else Path(runtime.get("log_file", ""))
    if not log_path.is_absolute():
        log_path = (config_path.parent / log_path).resolve()
    stream = summarize_runtime_log(
        log_path,
        sample_rate=int(runtime.get("sample_rate", 16_000)),
        wake_hop_ms=float(runtime.get("wake_hop_ms", 250)),
        stream_chunk_ms=float(runtime.get("stream_chunk_ms", 100)),
    ) if log_path.is_file() else {
        "log_file": log_path.name, "status": "missing_log", "false_activations_per_hour": None,
    }
    return {
        "schema_version": 1,
        "status": "telemetry_only_not_deployment_approval",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "file": config_path.name,
            "sha256": config_hash(runtime),
            "wake_hop_ms": runtime.get("wake_hop_ms"),
            "stream_chunk_ms": runtime.get("stream_chunk_ms"),
            "threads": runtime.get("threads"),
            "onset_timeout_ms": runtime.get("onset_timeout_ms"),
            "endpoint_silence_ms": runtime.get("endpoint_silence_ms"),
            "maximum_command_ms": runtime.get("maximum_command_ms"),
        },
        "artifacts": artifacts,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "board_model": board_model,
            "temperature_c": temperature,
            "vcgencmd_measure_temp": _vcgencmd("measure_temp"),
            "vcgencmd_get_throttled": _vcgencmd("get_throttled"),
        },
        "process": {
            "pid": args.pid,
            "monitor_duration_seconds": elapsed,
            "cpu_percent_one_core_p50": _percentile(cpu_samples, .50),
            "cpu_percent_one_core_p95": _percentile(cpu_samples, .95),
            "cpu_percent_all_cores_p95": (
                _percentile(cpu_samples, .95) / max(1, psutil.cpu_count() or 1)
                if cpu_samples else None
            ),
            "rss_peak_bytes": rss_peak,
            "rss_peak_mb": round(rss_peak / 1024**2, 2),
            "samples": len(cpu_samples),
        },
        "stream": stream,
        "limitations": [
            "This monitor does not open a microphone, play audio, start inference, or dispatch actions.",
            "CPU/RSS describe the selected process during this observation window.",
            "False activations/hour require annotated continuous-negative audio and event matching.",
            "A passing operator/telemetry check is not model, microphone, or deployment approval.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True, help="PID of an already-running ME2 process")
    parser.add_argument("--config", type=Path, default=Path("config/runtime.pi5.natural-pause.json"))
    parser.add_argument("--log", type=Path, help="Structured JSONL log to summarize")
    parser.add_argument("--duration-seconds", type=float, default=300)
    parser.add_argument("--interval-seconds", type=float, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pid <= 0 or not 1 <= args.duration_seconds <= 86_400:
        parser.error("PID must be positive and duration must be in [1, 86400] seconds")
    if not .1 <= args.interval_seconds <= 60:
        parser.error("interval must be in [0.1, 60] seconds")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    if output.exists():
        parser.error("output already exists; choose a new run-specific path")
    report = collect(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, prefix=output.name + ".",
        suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(rendered)
        stream.flush()
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
