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
}
_TOKEN = re.compile(r"^[A-Za-z0-9_.:+-]{1,160}$")


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
            **{k: _safe(v) for k, v in getattr(record, "structured_fields", {}).items()
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
        **{k: _safe(v) for k, v in (context or {}).items() if k in _FIELDS},
    }
    formatter = _JsonLogFormatter(debug_sensitive=debug_sensitive)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(RotatingFileHandler(
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
