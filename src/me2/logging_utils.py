"""Bounded structured diagnostics; speech, paths and exception messages are opt-in."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import subprocess
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONTEXT: dict = {}
_FIELDS = {
    "utterance_id", "state", "reason", "accepted", "candidate_count", "system_error",
    "sample_index", "sample_rate", "duration_seconds", "stage_timings_ms",
    "total_ms", "retry", "backoff_seconds", "task", "architecture", "model_version",
    "grammar_version", "artifact_sha256", "config_sha256", "decoder_version",
    "parameters", "artifact_bytes", "deployment_ready", "threads", "rss_bytes",
    "error_type", "wake_detected", "score", "debug_sensitive", "skipped_count",
    "diagnostics", "trial_id", "input_device_index", "mock_dispatch",
    "implementation_sha256", "accepted_intent",
}
_TOKEN = re.compile(r"^[A-Za-z0-9_.:+-]{1,160}$")
_DIAGNOSTIC_FIELDS = {
    "wake_sample_index", "speech_start_sample_index", "onset_timeout_ms",
    "onset_wait_ms", "speech_observed", "speech_ms", "speech_span_ms",
    "captured_samples", "pre_roll_ms", "endpoint_silence_ms",
    "trailing_silence_ms", "maximum_command_ms", "minimum_speech_ms",
    "capture_truncated", "capture_discarded", "capture_end_reason",
    "diagnostic_category", "next_action", "budget_reason", "usable_samples",
    "artifact_maximum_samples", "padded_samples", "feature_frames",
    "valid_feature_frames", "output_frames", "valid_output_frames", "beam_width",
    "grammar_terminal_count", "minimum_terminal_frames", "pruned_prefix_count",
    "beam_pruning_observed", "surviving_prefix_count", "nonempty_prefix_count",
    "longest_surviving_prefix", "terminal_count", "expansions", "phrase_support",
    "wake_class_is_ateko", "chunk_rms", "chunk_peak", "clipped_samples",
    "speech_active", "queue_delay_ms", "adc_time_seconds", "channels",
    "sample_format", "stream_latency_seconds", "callback_status_count",
    "queue_drop_count", "queue_depth_chunks", "queue_capacity_chunks",
    "callback_frames_total", "callback_count", "queue_delay_p95_ms",
    "feed_p95_ms", "queue_depth_mean", "queue_depth_max",
    "queue_depth_start", "queue_depth_end", "chunks_consumed",
    "stream_age_seconds", "flush",
    "command_decision_ms", "expected_phrase_supported", "normalization_changed",
    "expected_required_frames", "expected_alignment_feasible", "reference_diagnostic",
    "onset_boundary_verified", "pre_onset_dropped_samples", "pre_onset_dropped_rms",
    "frontend_ms", "wake_model_ms", "wake_pipeline_ms",
    "wake_primary_role", "wake_fallback_role", "wake_mode",
    "wake_primary_status", "wake_fallback_status",
    "wake_primary_failure", "wake_fallback_failure",
    "wake_primary_artifact_sha256", "wake_fallback_artifact_sha256",
    "wake_primary_label", "wake_primary_score", "wake_primary_model_ms",
    "wake_primary_detected", "wake_fallback_label", "wake_fallback_score",
    "wake_fallback_model_ms", "wake_fallback_detected", "wake_fallback_disagreement",
    "wake_effective_threshold", "wake_persistence_rule", "wake_smoothing_window",
    "wake_required_hits", "wake_observed_hits", "wake_current_hit",
    "wake_nontrigger_reason", "wake_debounce_state", "wake_cooldown_remaining_ms",
    "noise_floor_rms", "noise_floor_calibration_ms", "noise_floor_multiplier",
    "speech_hysteresis_ratio", "speech_onset_threshold", "speech_release_threshold",
    "capture_total_samples", "capture_clipped_samples", "clipping_fraction",
    "clipping_warning_fraction", "clipping_reject_fraction", "capture_peak",
    "capture_rms_min", "capture_rms_max", "capture_quality_warning",
}


def _safe_field(key, value):
    if key == "diagnostics":
        return {
            k: _safe(v) for k, v in value.items() if k in _DIAGNOSTIC_FIELDS
        } if isinstance(value, dict) else {}
    return _safe(value)


class LogWriteError(OSError):
    """A mandatory diagnostic sink failed; callers must stop safely."""


class _PrivateErrorHandler:
    def handleError(self, record):
        # The stdlib fallback prints unformatted msg/args, defeating redaction.
        logging.getLogger().removeHandler(self)
        self.close()
        raise LogWriteError("structured_log_write_failed") from None


class _PrivateStreamHandler(_PrivateErrorHandler, logging.StreamHandler):
    pass


class _PrivateRotatingFileHandler(_PrivateErrorHandler, RotatingFileHandler):
    pass


def _safe(value):
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value if _TOKEN.fullmatch(value) else "[redacted]"
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items() if _TOKEN.fullmatch(str(k))}
    return "[redacted]"


def config_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def runtime_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return "unknown-installed-build"
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL, timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def runtime_source_hash() -> str:
    """Identify uncommitted or installed runtime code independently of Git HEAD."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def log_event(logger: logging.Logger, event: str, *, level=logging.INFO, **fields) -> None:
    logger.log(level, event, extra={
        "event": event, "structured_fields": {k: v for k, v in fields.items() if k in _FIELDS},
    })


class _JsonLogFormatter(logging.Formatter):
    def __init__(self, *, debug_sensitive: bool = False):
        super().__init__()
        self.debug_sensitive = debug_sensitive

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "component": _safe(record.name),
            **_CONTEXT,
            "event": _safe(getattr(record, "event", "legacy_diagnostic")),
            **{k: _safe_field(k, v) for k, v in getattr(record, "structured_fields", {}).items()
               if k in _FIELDS},
        }
        if record.exc_info:
            payload["error_type"] = record.exc_info[0].__name__
        if self.debug_sensitive:
            payload["debug_message"] = record.getMessage()[:2048]
            if record.exc_info:
                payload["debug_exception"] = self.formatException(record.exc_info)[-4096:]
        return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def configure_logging(
    log_path: str | Path | None = None,
    level: int = logging.INFO,
    *,
    debug_sensitive: bool = False,
    context: dict | None = None,
) -> None:
    global _CONTEXT
    _CONTEXT = {
        "log_schema_version": 2,
        "run_id": uuid.uuid4().hex,
        "git_commit": runtime_revision(),
        "implementation_sha256": runtime_source_hash(),
        **{k: _safe(v) for k, v in (context or {}).items() if k in _FIELDS},
    }
    formatter = _JsonLogFormatter(debug_sensitive=debug_sensitive)
    handlers: list[logging.Handler] = [_PrivateStreamHandler()]
    try:
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(_PrivateRotatingFileHandler(
                path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8",
            ))
        for handler in handlers:
            handler.setFormatter(formatter)
        logging.basicConfig(level=level, handlers=handlers, force=True)
    except OSError:
        for handler in handlers:
            handler.close()
        raise
    log_event(logging.getLogger("me2.runtime"), "run_started", debug_sensitive=debug_sensitive)
