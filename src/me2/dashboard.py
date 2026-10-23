"""Local dashboard for the shared wake/command pipeline; no web build required."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import logging
import math
import queue
import re
import secrets
import signal
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import psutil
import torch

from me2.app import (
    _model,
    _require_live_wake_readiness,
    build_streaming_pipeline,
)
from me2.app import build_parser as inference_parser
from me2.audio_devices import resolve_alsa_device, select_capture_input, validate_alsa_device
from me2.dashboard_audio import DashboardAudioPlayer, WakeChimePlayer, feedback_tone
from me2.logging_utils import configure_logging, log_event
from me2.streaming import (
    PipelineEvent,
    PipelineState,
    StreamingConfig,
    StreamingPipeline,
    run_microphone_stream,
)
from me2.vcm.ctc import CommandResult, coerce_grammar_entries
from me2.weather import fetch_current_weather

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).with_name("dashboard_static")
INTENTS = (
    "play_music", "weather", "time", "light_on", "light_off", "pause", "stop",
    "next", "volume_up", "volume_down", "call", "message", "list_reminders",
    "timer", "alarm", "temperature", "brightness", "color", "create_reminder",
)
ACTION_LABELS = {
    "play_music": "Play sample music",
    "weather": "Show configured-city weather",
    "time": "Show current local time",
    "light_on": "Set demo light on",
    "light_off": "Set demo light off",
    "pause": "Pause sample music",
    "stop": "Stop sample music",
    "next": "Advance sample track",
    "volume_up": "Raise demo volume",
    "volume_down": "Lower demo volume",
    "call": "Show simulated call ringtone",
    "message": "Show demo message (not sent)",
    "list_reminders": "Show demo reminders",
    "timer": "Create demo timer",
    "alarm": "Create demo alarm",
    "temperature": "Set demo thermostat",
    "brightness": "Set demo brightness",
    "color": "Set demo light color",
    "create_reminder": "Create demo reminder",
}
ACTIONS = {intent: ACTION_LABELS[intent] for intent in INTENTS}
DEFAULT_ACTION_BY_INTENT = {intent: intent for intent in INTENTS}

# These are display-only examples for the synthetic dashboard demo. Live
# dispatch instead requires semantic values supplied by the loaded grammar.
DEMO_SLOTS = {
    "brightness": {"value": "60"},
    "color": {"value": "Blue"},
    "temperature": {"value": "22"},
    "timer": {"value": "10 seconds"},
    "alarm": {"value": "8:00 AM"},
    "create_reminder": {"value": "Drink water"},
}
SLOT_REQUIRED = frozenset(DEMO_SLOTS)
_ALSA_DEVICE = re.compile(r"^[A-Za-z0-9_.:,=-]{1,120}$")
_NUMERIC_ALSA_SELECTOR = re.compile(r"(?:plughw|hw):\d+,\d+", re.IGNORECASE)
_ACK_MESSAGE = "Yes? Listening for your command."


def clean_json(value):
    """Browser JSON cannot represent the -inf scores used for capture rejection."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _duration_seconds(value: str | None) -> int:
    if not value:
        raise ValueError("required supported timer value is missing")
    match = re.fullmatch(
        r"\s*(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m)?\s*",
        value,
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError("timer value is not a supported duration")
    amount = int(match.group(1))
    unit = (match.group(2) or "seconds").casefold()
    seconds = amount * (60 if unit.startswith(("m",)) else 1)
    if not 1 <= seconds <= 86_400:
        raise ValueError("timer duration must be between 1 second and 24 hours")
    return seconds


def _clock_label(value: str | None) -> str:
    if not value:
        raise ValueError("required supported alarm time is missing")
    numeric = re.fullmatch(r"\d{1,4}", value.strip())
    if numeric:
        amount = int(value)
        if 24 <= amount <= 1_439:
            return f"{amount // 60:02d}:{amount % 60:02d}"
    match = re.fullmatch(
        r"\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*", value, re.IGNORECASE
    )
    if match is None:
        raise ValueError("alarm value is not a supported clock time")
    hour, minute = int(match.group(1)), int(match.group(2) or 0)
    suffix = (match.group(3) or "").upper()
    if minute > 59 or (suffix and not 1 <= hour <= 12) or (not suffix and hour > 23):
        raise ValueError("alarm time is outside the supported clock range")
    if suffix:
        hour = (hour % 12) + (12 if suffix == "PM" else 0)
    return f"{hour:02d}:{minute:02d}"


def load_settings(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    schema_version = config.get("schema_version", 1)
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("dashboard schema_version must be 1")
    config["schema_version"] = schema_version
    for key in ("wake_model", "vcm_model", "mappings_file"):
        value = Path(config[key]).expanduser()
        config[key] = value if value.is_absolute() else (path.parent / value).resolve()
    if "wake_fallback_model" in config:
        raise ValueError("wake_fallback_model is not supported; wake faults fail closed")
    config.setdefault("weather", {})
    config.setdefault("music", {})
    if not isinstance(config["weather"], dict) or not isinstance(config["music"], dict):
        raise TypeError("weather and music settings must be JSON objects")
    city = config["weather"].get("city", "Manila")
    if not isinstance(city, str) or not city.strip() or len(city) > 120:
        raise ValueError("weather.city must be a non-empty city name")
    config["weather"]["city"] = city.strip()
    timeout = config["weather"].get("timeout_seconds", 4.0)
    cache_ttl = config["weather"].get("cache_ttl_seconds", 900)
    if type(timeout) not in (int, float) or not 0.2 <= timeout <= 15:
        raise ValueError("weather.timeout_seconds must be in [0.2, 15]")
    if type(cache_ttl) is not int or not 0 <= cache_ttl <= 86_400:
        raise ValueError("weather.cache_ttl_seconds must be in [0, 86400]")
    config["weather"]["timeout_seconds"] = float(timeout)
    config["weather"]["cache_ttl_seconds"] = cache_ttl
    if "playback_enabled" in config["music"] and type(
        config["music"]["playback_enabled"]
    ) is not bool:
        raise ValueError("music.playback_enabled must be boolean")
    if "sample_rate" in config["music"] and (
        type(config["music"]["sample_rate"]) is not int
        or not 8_000 <= config["music"]["sample_rate"] <= 48_000
    ):
        raise ValueError("music.sample_rate must be an integer in [8000, 48000]")
    output_device = config["music"].get("output_device")
    if output_device is not None and not isinstance(output_device, (int, str)):
        raise TypeError("music.output_device must be a device index, name, or null")
    if isinstance(output_device, str) and output_device.isdecimal():
        config["music"]["output_device"] = int(output_device)
    audio_backend = config.get("audio_backend", "sounddevice")
    if audio_backend not in {"auto", "sounddevice", "arecord"}:
        raise ValueError("audio_backend must be auto, sounddevice, or arecord")
    config["audio_backend"] = audio_backend
    alsa_device = config.get("alsa_device", "default")
    if (not isinstance(alsa_device, str) or not _ALSA_DEVICE.fullmatch(alsa_device)
            or _NUMERIC_ALSA_SELECTOR.search(alsa_device)):
        raise ValueError("alsa_device must be a supported ALSA device identifier")
    config["alsa_device"] = alsa_device
    acknowledgement = config.setdefault("wake_acknowledgement", {})
    if not isinstance(acknowledgement, dict):
        raise TypeError("wake_acknowledgement must be a JSON object")
    allowed_acknowledgement = {
        "enabled", "backend", "alsa_device", "duration_ms", "frequency_hz",
        "volume", "sample_rate",
    }
    unknown = set(acknowledgement) - allowed_acknowledgement
    if unknown:
        raise ValueError(
            f"unsupported wake_acknowledgement setting: {min(unknown)}"
        )
    acknowledgement.setdefault("enabled", False)
    acknowledgement.setdefault("backend", "alsa")
    acknowledgement.setdefault("alsa_device", "default")
    acknowledgement.setdefault("duration_ms", 120)
    acknowledgement.setdefault("frequency_hz", 880)
    acknowledgement.setdefault("volume", 0.12)
    acknowledgement.setdefault("sample_rate", 48_000)
    if type(acknowledgement["enabled"]) is not bool:
        raise ValueError("wake_acknowledgement.enabled must be boolean")
    if acknowledgement["backend"] != "alsa":
        raise ValueError("wake_acknowledgement.backend must be 'alsa'")
    if (
        not isinstance(acknowledgement["alsa_device"], str)
        or not _ALSA_DEVICE.fullmatch(acknowledgement["alsa_device"])
        or _NUMERIC_ALSA_SELECTOR.search(acknowledgement["alsa_device"])
    ):
        raise ValueError(
            "wake_acknowledgement.alsa_device must be a supported ALSA device identifier"
        )
    for key, lower, upper in (
        ("duration_ms", 20, 500),
        ("frequency_hz", 200, 4_000),
        ("sample_rate", 8_000, 48_000),
    ):
        value = acknowledgement[key]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"wake_acknowledgement.{key} must be in [{lower}, {upper}]")
    volume = acknowledgement["volume"]
    if type(volume) not in (int, float) or not 0.01 <= float(volume) <= 0.5:
        raise ValueError("wake_acknowledgement.volume must be in [0.01, 0.5]")
    acknowledgement["volume"] = float(volume)
    device = config.get("input_device")
    if isinstance(device, str) and device.isdecimal():
        device = int(device)
    config["input_device"] = device
    return config


class Dashboard:
    def __init__(self, settings: dict, *, demo: bool, allow_unverified: bool = False):
        self.settings = settings
        self.demo = demo
        self.allow_unverified = allow_unverified
        self.lock = threading.RLock()
        self.stop_requested = threading.Event()
        self.worker: threading.Thread | None = None
        self.demo_requests: queue.Queue[dict] = queue.Queue(maxsize=1)
        self.token = secrets.token_urlsafe(32)
        self.events: deque[dict] = deque(maxlen=80)
        self.sequence = 0
        self.mappings = {
            intent: {"enabled": True, "action": DEFAULT_ACTION_BY_INTENT[intent]}
            for intent in INTENTS
        }
        mappings_file = settings["mappings_file"]
        if mappings_file.exists():
            loaded_mappings = json.loads(mappings_file.read_text(encoding="utf-8"))
            # The old dashboard used "display" for most commands. Upgrade only
            # that legacy default in memory; preserve enable/disable choices and
            # leave the user's file untouched until they explicitly save.
            if isinstance(loaded_mappings, dict):
                for intent, mapping in loaded_mappings.items():
                    if (
                        intent in DEFAULT_ACTION_BY_INTENT
                        and isinstance(mapping, dict)
                        and mapping.get("action") == "display"
                    ):
                        mapping["action"] = DEFAULT_ACTION_BY_INTENT[intent]
            self.mappings = self.validate_mappings(loaded_mappings)
        self.supported_slot_values = {
            intent: {values["value"].casefold(): values["value"]}
            for intent, values in DEMO_SLOTS.items()
        } if demo else {}
        self.audio_player = DashboardAudioPlayer(
            sample_rate=settings.get("music", {}).get("sample_rate", 22_050),
            output_device=settings.get("music", {}).get("output_device"),
        )
        acknowledgement = settings.get("wake_acknowledgement", {})
        self.wake_chime = WakeChimePlayer(
            enabled=bool(acknowledgement.get("enabled", False)),
            alsa_device=acknowledgement.get("alsa_device", "default"),
            duration_ms=acknowledgement.get("duration_ms", 120),
            frequency_hz=acknowledgement.get("frequency_hz", 880),
            volume=float(acknowledgement.get("volume", 0.12)),
            sample_rate=acknowledgement.get("sample_rate", 48_000),
            on_complete=self._on_chime_complete,
        )
        self.weather_settings = settings.get("weather", {"city": "Manila"})
        self._weather_thread: threading.Thread | None = None
        self._seen_utterance_ids: deque[str] = deque(maxlen=256)
        self._seen_utterance_set: set[str] = set()
        self._acknowledged_utterance_ids: deque[str] = deque(maxlen=256)
        self._acknowledged_utterance_set: set[str] = set()
        self.data = {
            "mode": "demo" if demo else "microphone",
            "state": "STOPPED",
            "running": False,
            "stopping": False,
            "error": None,
            "notice": (
                "Synthetic demo · no microphone or model predictions"
                if demo else "Microphone is off. Start listening to load and validate models."
            ),
            "device": settings.get("input_device"),
            "rms": 0.0,
            "wake": None,
            "wake_acknowledgement": self._hidden_acknowledgement(),
            "wake_policy": {
                "primary_role": "v5_primary",
                "fallback_role": "not_supported",
                "mode": "not_started",
            },
            "models": {},
            "last_result": None,
            "last_action": None,
            "last_action_status": None,
            "counts": {"wake": 0, "accepted": 0, "rejected": 0, "duplicates": 0},
            "devices": {
                "light": False, "player": "stopped", "volume": 50, "track": 1,
                "brightness": 60, "color": None, "thermostat_c": 22,
                "time": None, "call_demo": None, "message_demo": None,
                "reminders": [], "timers": [], "alarms": [],
                "weather": {
                    "status": "idle", "city": self.weather_settings.get("city", "Manila"),
                    "current": None, "cached_at": None, "error": None,
                },
            },
            "supported_intents": list(INTENTS) if demo else [],
        }
        self.process = psutil.Process()
        self.process.cpu_percent()
        self.metrics_at = 0.0
        self.metrics: dict = {}

    def _hidden_acknowledgement(self) -> dict:
        return {
            "visible": False,
            "message": _ACK_MESSAGE,
            "timestamp": None,
            "model_role": None,
            "audio_status": (
                "disabled" if not self.wake_chime.enabled else "not_requested"
            ),
        }

    def _clear_acknowledgement(self) -> None:
        self.data["wake_acknowledgement"] = self._hidden_acknowledgement()

    def _on_chime_complete(self, status: str, error_type: str | None) -> None:
        with self.lock:
            acknowledgement = self.data["wake_acknowledgement"]
            if acknowledgement["visible"] and acknowledgement["audio_status"] == "queued":
                acknowledgement["audio_status"] = status
        if status == "failed":
            log_event(
                LOGGER,
                "acknowledgement_audio_failed",
                level=logging.WARNING,
                reason="acknowledgement_audio_failed",
                error_type=error_type,
            )

    def _remember_acknowledgement(self, utterance_id: str) -> bool:
        if utterance_id in self._acknowledged_utterance_set:
            return False
        if len(self._acknowledged_utterance_ids) == self._acknowledged_utterance_ids.maxlen:
            expired = self._acknowledged_utterance_ids.popleft()
            self._acknowledged_utterance_set.discard(expired)
        self._acknowledged_utterance_ids.append(utterance_id)
        self._acknowledged_utterance_set.add(utterance_id)
        return True

    def _acknowledge_wake(self, event: PipelineEvent) -> None:
        policy = self.data["wake_policy"]
        if self.demo:
            model_role = "synthetic_demo"
        elif policy.get("mode") == "primary" and policy.get("primary_status") == "ready":
            model_role = "v5_primary"
        else:
            return
        if not event.utterance_id or not self._remember_acknowledgement(event.utterance_id):
            return
        self.data["wake_acknowledgement"] = {
            "visible": True,
            "message": _ACK_MESSAGE,
            "timestamp": now(),
            "model_role": model_role,
            "audio_status": (
                "queued" if not self.demo and self.wake_chime.enabled else "disabled"
            ),
        }
        if not self.demo and self.wake_chime.enabled and not self.wake_chime.request(
            frequency_hz=feedback_tone("wake_accepted")
        ):
            self.data["wake_acknowledgement"]["audio_status"] = "unavailable"

    def _request_feedback_tone(self, kind: str) -> None:
        """Best-effort local tone; inference and fail-closed dispatch never depend on it."""
        if self.demo or not self.wake_chime.enabled:
            return
        if not self.wake_chime.request(frequency_hz=feedback_tone(kind)):
            log_event(LOGGER, "feedback_tone_unavailable", reason="feedback_tone_unavailable")

    @staticmethod
    def validate_mappings(mappings: dict) -> dict:
        if not isinstance(mappings, dict) or set(mappings) != set(INTENTS):
            raise ValueError("Mappings must contain exactly the 19 supported intent names")
        for intent, value in mappings.items():
            if not isinstance(value, dict) or set(value) != {"enabled", "action"}:
                raise ValueError("Each mapping requires enabled and action")
            if (
                type(value["enabled"]) is not bool
                or value["action"] != DEFAULT_ACTION_BY_INTENT[intent]
            ):
                raise ValueError(
                    f"{intent} must use its canonical dashboard action "
                    f"{DEFAULT_ACTION_BY_INTENT[intent]!r}"
                )
        return copy.deepcopy(mappings)

    def save_mappings(self, mappings: dict) -> None:
        validated = self.validate_mappings(mappings)
        with self.lock:
            path = self.settings["mappings_file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            try:
                temporary.write_text(json.dumps(validated, indent=2) + "\n", encoding="utf-8")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            self.mappings = validated

    def snapshot(self) -> dict:
        with self.lock:
            if time.monotonic() - self.metrics_at >= 2:
                temperature = None
                try:
                    temperature = int(
                        Path("/sys/class/thermal/thermal_zone0/temp").read_text()
                    ) / 1000
                except (OSError, ValueError):
                    pass
                self.metrics = {
                    "cpu_percent": self.process.cpu_percent(),
                    "rss_mb": round(self.process.memory_info().rss / 1024**2, 1),
                    "temperature_c": temperature,
                }
                self.metrics_at = time.monotonic()
            return clean_json(copy.deepcopy({
                **self.data,
                "events": list(self.events),
                "metrics": self.metrics,
                "token": self.token,
            }))

    def start(self) -> None:
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                raise ValueError("The pipeline is already running or stopping")
            self.stop_requested.clear()
            while not self.demo_requests.empty():
                self.demo_requests.get_nowait()
            self.data.update(
                running=True, stopping=False, state="STARTING", error=None,
                wake=None, last_result=None, last_action=None, rms=0.0,
            )
            self._clear_acknowledgement()
            self.worker = threading.Thread(target=self._run, name="me2-audio", daemon=True)
            self.worker.start()

    def stop(self) -> None:
        with self.lock:
            self.stop_requested.set()
            if self.data["running"]:
                self.data["stopping"] = True
                self.data["state"] = "STOPPING"
            self._clear_acknowledgement()

    def close(self) -> None:
        self.stop()
        if self.worker:
            self.worker.join(timeout=8)
        try:
            self.audio_player.stop()
        except (OSError, RuntimeError, ValueError) as error:
            LOGGER.warning(
                "Could not close dashboard audio: error_type=%s", type(error).__name__
            )

    def demo_command(self, intent: str, accepted: bool) -> None:
        with self.lock:
            if not self.demo or not self.data["running"] or self.data["stopping"]:
                raise ValueError("Start demo listening before sending an example")
            if intent not in INTENTS or type(accepted) is not bool:
                raise ValueError("Choose a known intent and a boolean accepted value")
            if self.data["state"] != "LISTENING_FOR_WAKE":
                raise ValueError("Wait for the pipeline to return to wake listening")
            try:
                self.demo_requests.put_nowait({"intent": intent, "accepted": accepted})
            except queue.Full as error:
                raise ValueError("An example is already queued") from error

    def on_wake(
        self,
        label: str,
        score: float,
        elapsed: float,
        detected: bool,
        policy: dict | None = None,
    ) -> None:
        with self.lock:
            self.data["wake"] = {
                "label": label, "score": score, "inference_ms": elapsed,
                "detected": detected, "at": now(),
            }
            if policy is not None:
                self.data["wake_policy"] = {
                    key: policy.get(key)
                    for key in (
                        "primary_role", "fallback_role", "mode", "primary_status",
                        "fallback_status", "primary_failure", "fallback_failure",
                    )
                }

    def on_audio(self, chunk: np.ndarray) -> None:
        with self.lock:
            self.data["rms"] = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))

    def on_event(self, event: PipelineEvent) -> None:
        with self.lock:
            self.sequence += 1
            record = {
                "id": self.sequence, "at": now(), "state": event.state.value,
                "message": event.message, "sample_index": event.sample_index,
                "intent": None, "action": None, "action_status": None,
            }
            if not self.stop_requested.is_set():
                self.data["state"] = event.state.value
            if event.message == "wake_detected":
                self.data["counts"]["wake"] += 1
                self._acknowledge_wake(event)
            elif event.result is not None or event.state in {
                PipelineState.COOLDOWN, PipelineState.STOPPED,
            }:
                self._clear_acknowledgement()
            if event.result is not None:
                result = event.result
                self.data["last_result"] = {
                    **dataclasses.asdict(result), "actionable_intent": result.actionable_intent,
                }
                utterance_id = event.utterance_id
                duplicate = bool(
                    utterance_id and utterance_id in self._seen_utterance_set
                )
                if duplicate:
                    self.data["counts"]["duplicates"] += 1
                    action = "Skipped · duplicate utterance event"
                    self.data["last_action_status"] = "duplicate"
                    log_event(
                        LOGGER, "dashboard_action_suppressed",
                        utterance_id=utterance_id, reason="duplicate_utterance",
                    )
                else:
                    if utterance_id:
                        if len(self._seen_utterance_ids) == self._seen_utterance_ids.maxlen:
                            expired = self._seen_utterance_ids.popleft()
                            self._seen_utterance_set.discard(expired)
                        self._seen_utterance_ids.append(utterance_id)
                        self._seen_utterance_set.add(utterance_id)
                    key = "accepted" if result.accepted else "rejected"
                    self.data["counts"][key] += 1
                    action = self.dispatch(result)
                self.data["last_action"] = action
                self._request_feedback_tone(
                    "command_accepted" if result.accepted and result.actionable_intent else "rejection_or_error"
                )
                record.update(
                    intent=result.actionable_intent, action=action,
                    action_status=self.data["last_action_status"],
                )
            self.events.appendleft(record)
            LOGGER.info("pipeline_event=%s", json.dumps(clean_json(record), allow_nan=False))

    def dispatch(self, result: CommandResult) -> str:
        """Apply the one canonical local action for an accepted intent."""
        if self.stop_requested.is_set():
            return self._action_outcome("skipped", "Skipped · pipeline stopping")
        intent = result.actionable_intent
        if intent is None:
            reason = "system error" if result.system_error else "command rejected"
            return self._action_outcome("skipped", f"Skipped · {reason}")
        if intent not in DEFAULT_ACTION_BY_INTENT:
            return self._action_outcome("failed", "Failed · unknown intent")
        mapping = self.mappings.get(intent)
        if mapping is None or not mapping["enabled"]:
            return self._action_outcome(
                "skipped", "Skipped · mapping disabled or unavailable", intent
            )
        if mapping["action"] != DEFAULT_ACTION_BY_INTENT[intent]:
            return self._action_outcome("failed", "Failed · mapping contract mismatch", intent)

        try:
            slot_value = self._validated_slot(intent, result.slots)
            label = self._apply_action(intent, slot_value)
        except ValueError as error:
            LOGGER.warning(
                "Dashboard action failed: intent=%s error_type=%s",
                intent, type(error).__name__,
            )
            return self._action_outcome(
                "failed", f"Failed · {error}", intent
            )
        except (ImportError, OSError, RuntimeError, TypeError) as error:
            LOGGER.warning(
                "Dashboard action failed: intent=%s error_type=%s",
                intent, type(error).__name__,
            )
            return self._action_outcome(
                "failed",
                "Failed · local action unavailable; check the configured device or service.",
                intent,
            )
        if intent == "weather":
            # The asynchronous lookup owns the eventual success/failure state.
            return label
        return self._action_outcome("success", label, intent)

    def _action_outcome(self, status: str, message: str, intent: str | None = None) -> str:
        self.data["last_action_status"] = status
        log_event(
            LOGGER,
            "dashboard_action",
            level=logging.WARNING if status == "failed" else logging.INFO,
            accepted_intent=intent,
            reason=status,
        )
        return message

    def _validated_slot(self, intent: str, slots) -> str | None:
        if not isinstance(slots, dict):
            raise TypeError("invalid slot payload")
        if intent not in SLOT_REQUIRED:
            if slots:
                raise ValueError("unsupported slot supplied for this intent")
            return None
        if set(slots) != {"value"} or not isinstance(slots.get("value"), str):
            raise ValueError("required supported slot is missing")
        value = slots["value"].strip()
        if not value:
            raise ValueError("required supported slot is empty")
        allowed = self.supported_slot_values.get(intent, {})
        normalized = value.casefold()
        if not allowed or normalized not in allowed:
            raise ValueError(
                f"unsupported or unavailable {intent} slot; "
                "use a value explicitly supported by the loaded grammar"
            )
        return allowed[normalized]

    def _apply_action(self, intent: str, slot_value: str | None) -> str:
        devices = self.data["devices"]
        if intent == "play_music":
            if not self.demo and self.settings.get("music", {}).get(
                "playback_enabled", False
            ):
                self.audio_player.play()
            devices["player"] = "playing"
        elif intent == "pause":
            if not self.demo and devices["player"] == "playing":
                self.audio_player.pause()
            devices["player"] = "paused"
        elif intent == "stop":
            if not self.demo:
                self.audio_player.stop()
            devices["player"] = "stopped"
        elif intent == "next":
            if not self.demo and devices["player"] == "playing":
                self.audio_player.next_track()
            devices["track"] += 1
        elif intent in {"volume_up", "volume_down"}:
            delta = 10 if intent == "volume_up" else -10
            updated = min(100, max(0, devices["volume"] + delta))
            if not self.demo and self.settings.get("music", {}).get(
                "playback_enabled", False
            ):
                self.audio_player.set_volume(updated / 100)
            devices["volume"] = updated
        elif intent in {"light_on", "light_off"}:
            devices["light"] = intent == "light_on"
        elif intent == "brightness":
            devices["brightness"] = int(
                slot_value.removesuffix(" percent").removesuffix("%")
            )
        elif intent == "color":
            devices["color"] = slot_value
        elif intent == "temperature":
            devices["thermostat_c"] = int(slot_value.removesuffix(" degrees"))
        elif intent == "time":
            devices["time"] = datetime.now().astimezone().isoformat()
        elif intent == "weather":
            self._start_weather_lookup()
        elif intent == "call":
            if not self.demo and self.settings.get("music", {}).get(
                "playback_enabled", False
            ):
                self.audio_player.play_ringtone()
            devices["call_demo"] = {
                "message": "Simulated call ringtone · no call was placed",
                "at": now(),
            }
        elif intent == "message":
            devices["message_demo"] = {
                "message": "Demo message prepared · not sent",
                "at": now(),
            }
        elif intent == "list_reminders":
            devices["reminders_viewed_at"] = now()
        elif intent == "create_reminder":
            devices["reminders"].append({"text": slot_value, "created_at": now()})
        elif intent == "timer":
            seconds = _duration_seconds(slot_value)
            devices["timers"].append({
                "seconds": seconds, "ends_at": datetime.now(timezone.utc).timestamp() + seconds,
                "label": f"Demo timer · {slot_value}",
            })
        elif intent == "alarm":
            alarm_time = _clock_label(slot_value)
            devices["alarms"].append({"time": alarm_time, "label": "Demo alarm"})
        else:
            raise ValueError("unsupported dashboard action")
        return ACTION_LABELS[intent]

    def _start_weather_lookup(self) -> None:
        weather = self.data["devices"]["weather"]
        if weather["status"] == "loading":
            raise RuntimeError("weather lookup is already in progress")
        ttl = self.weather_settings.get("cache_ttl_seconds", 900)
        if weather.get("current") and weather.get("cached_at") and ttl > 0:
            try:
                age = (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(weather["cached_at"])
                ).total_seconds()
            except (TypeError, ValueError):
                age = ttl + 1
            if 0 <= age < ttl:
                weather.update(status="cached", error=None)
                self.data["last_action_status"] = "success"
                self.data["last_action"] = "Showing recent weather observation"
                return
        previous = copy.deepcopy(weather)
        weather.update(
            status="loading", city=self.weather_settings["city"], error=None,
        )
        self.data["last_action_status"] = "loading"
        self.data["last_action"] = "Weather lookup started"

        def lookup():
            try:
                current = fetch_current_weather(
                    self.weather_settings["city"],
                    timeout_seconds=self.weather_settings.get("timeout_seconds", 4.0),
                )
            except Exception as error:  # noqa: BLE001 - external service failure becomes UI state
                with self.lock:
                    cached = previous.get("current")
                    weather.update(
                        status="stale" if cached else "error",
                        current=cached,
                        cached_at=previous.get("cached_at"),
                        error=(
                            "Weather service unavailable; showing cached observation."
                            if cached else "Weather service unavailable; check network and city settings."
                        ),
                    )
                    self.data["last_action_status"] = "stale" if cached else "failed"
                    self.data["last_action"] = weather["error"]
                LOGGER.warning(
                    "Weather lookup failed: error_type=%s", type(error).__name__
                )
                return
            with self.lock:
                weather.update(
                    status="ready", city=current["city"], current=current,
                    cached_at=now(), error=None,
                )
                self.data["last_action_status"] = "success"
                self.data["last_action"] = ACTION_LABELS["weather"]

        self._weather_thread = threading.Thread(
            target=lookup, name="me2-weather", daemon=True
        )
        try:
            self._weather_thread.start()
        except RuntimeError:
            self.data["devices"]["weather"] = previous
            raise

    def _run(self) -> None:
        pipeline = None
        try:
            if self.demo:
                self._run_demo()
                return
            args = inference_parser().parse_args(["--interactive"])
            args.input_device = self.settings.get("input_device")
            for key, value in self.settings.get("streaming", {}).items():
                if not hasattr(args, key) or key not in {
                    "stream_chunk_ms", "wake_hop_ms", "pre_roll_ms", "onset_timeout_ms",
                    "minimum_speech_ms", "endpoint_silence_ms", "cooldown_ms",
                    "speech_rms_threshold", "noise_floor_calibration_ms",
                    "noise_floor_multiplier", "speech_hysteresis_ratio",
                    "clipping_warning_fraction", "clipping_reject_fraction",
                    "maximum_command_ms", "queue_chunks", "post_wake_guard_ms",
                }:
                    raise ValueError(f"Unknown streaming setting: {key}")
                setattr(args, key, value)
            wake_path, vcm_path = self.settings["wake_model"], self.settings["vcm_model"]
            wake_model, wake_bundle = _model(wake_path, "wake_word")
            vcm_model, vcm_bundle = _model(vcm_path, "vcm")
            if vcm_bundle.get("grammar"):
                grammar = coerce_grammar_entries(vcm_bundle["grammar"])
                intents = sorted({entry.intent for entry in grammar})
                examples = [dataclasses.asdict(entry) for entry in grammar]
                slot_values: dict[str, dict[str, str]] = {}
                for entry in grammar:
                    for value in entry.slots.values():
                        slot_values.setdefault(entry.intent, {})[value.casefold()] = value
                with self.lock:
                    self.supported_slot_values = slot_values
            else:
                intents = [label for label in vcm_bundle["labels"] if label != "out_of_scope"]
                examples = []
            models = {}

            def add_model(name, role, path, model, bundle, load_error=None):
                models[name] = {
                    "role": role,
                    "path": str(path), "architecture": bundle.get("architecture"),
                    "version": bundle.get("model_version"),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "deployment_ready": bundle.get("deployment_ready"),
                    "audio": bundle["audio"], "threshold": bundle.get("threshold"),
                    "streaming_policy": bundle.get("streaming_policy", {}),
                    "grammar_version": bundle.get("grammar_version"),
                    "status": "ready" if model is not None else "load_failed",
                    "load_error": load_error,
                }
            add_model("wake_primary", "v5_primary", wake_path, wake_model, wake_bundle)
            add_model("command", "command_development", vcm_path, vcm_model, vcm_bundle)
            with self.lock:
                self.data.update(models=models, supported_intents=intents)
                self.data["grammar"] = examples
            _require_live_wake_readiness(
                wake_bundle, allow_unverified=self.allow_unverified
            )
            selection = select_capture_input(
                self.settings.get("audio_backend", "auto"),
                input_device=args.input_device,
                alsa_device=self.settings.get("alsa_device", "auto"),
                sample_rate=int(wake_bundle["audio"]["sample_rate"]),
            )
            args.audio_backend = selection.backend
            args.input_device = selection.input_device
            args.alsa_device = selection.alsa_device or self.settings.get("alsa_device", "auto")
            if args.audio_backend == "arecord":
                validate_alsa_device("capture", args.alsa_device)
                device = f"ALSA {args.alsa_device}"
            else:
                import sounddevice as sd

                device = sd.query_devices(args.input_device, kind="input")["name"]
            acknowledgement = self.settings.get("wake_acknowledgement", {})
            if acknowledgement.get("enabled"):
                output_device, _ = resolve_alsa_device(
                    "playback", acknowledgement.get("alsa_device", "auto")
                )
                validate_alsa_device("playback", output_device)
                self.wake_chime.alsa_device = output_device
            if self.settings.get("music", {}).get("playback_enabled", False):
                import sounddevice as sd

                sd.check_output_settings(
                    device=self.settings["music"].get("output_device"), channels=1,
                    dtype="float32", samplerate=self.audio_player.sample_rate,
                )
            pipeline = build_streaming_pipeline(
                args,
                wake_model,
                wake_bundle,
                vcm_model,
                vcm_bundle,
                on_event=self.on_event, on_wake=self.on_wake,
            )
            with self.lock:
                self.data.update(
                    device=device,
                    wake_policy=dict(pipeline.wake_status),
                    notice=(
                        (
                            "Engineering override · v5-only wake model has not passed "
                            "deployment review"
                        )
                        if wake_bundle.get("deployment_ready") is False
                        else "Live microphone · actions update simulated dashboard devices"
                    ),
                )
            if self.stop_requested.is_set():
                return
            self.on_event(PipelineEvent(pipeline.state, 0, "listening_started"))
            run_microphone_stream(
                pipeline, input_device=args.input_device,
                audio_backend=args.audio_backend, alsa_device=args.alsa_device,
                stop_event=self.stop_requested, on_audio=self.on_audio,
            )
        except Exception as error:
            LOGGER.exception("Dashboard pipeline failed")
            with self.lock:
                self.data["error"] = str(error)
                self._clear_acknowledgement()
        finally:
            if pipeline is not None:
                pipeline.stop()
            with self.lock:
                self.data.update(
                    running=False, stopping=False, rms=0.0,
                    state="ERROR" if self.data["error"] else "STOPPED",
                )
                self._clear_acknowledgement()

    def _run_demo(self) -> None:
        active: dict | None = None
        speech_chunks = 0

        def wake_predictor(window):
            detected = active is not None
            self.on_wake("ateko" if detected else "background", 0.96 if detected else 0.03,
                         0.0, detected)
            return detected

        def command_predictor(waveform):
            self.stop_requested.wait(0.6)
            accepted = active["accepted"]
            return CommandResult(
                decoded_text=active["intent"].replace("_", " ") if accepted else "unknown request",
                intent=active["intent"] if accepted else None,
                slots=DEMO_SLOTS.get(active["intent"], {}) if accepted else {},
                variation="", raw_log_score=-1.0, normalized_log_score=-0.1,
                candidate_posterior=0.94, margin=0.4, accepted=accepted,
                rejection_reason=None if accepted else "demo_out_of_scope",
                model_version="synthetic-demo",
            )

        pipeline = StreamingPipeline(
            StreamingConfig(wake_window_samples=6400, wake_hop_samples=1600),
            wake_predictor, command_predictor, on_event=self.on_event,
        )
        self.on_event(PipelineEvent(pipeline.state, 0, "demo_listening_started"))
        try:
            while not self.stop_requested.wait(0.1):
                if pipeline.state == PipelineState.LISTENING_FOR_WAKE:
                    if active is None:
                        try:
                            active = self.demo_requests.get_nowait()
                        except queue.Empty:
                            pass
                    speech_chunks = 0
                elif pipeline.state == PipelineState.COOLDOWN:
                    active = None
                speech = pipeline.state == PipelineState.CAPTURING_COMMAND and speech_chunks < 5
                chunk = np.full(1600, 0.08 if speech else 0.0, dtype=np.float32)
                if pipeline.state == PipelineState.CAPTURING_COMMAND:
                    speech_chunks += 1
                self.on_audio(chunk)
                pipeline.feed(chunk)
        finally:
            pipeline.stop()


def make_server(dashboard: Dashboard, port: int = 8765) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            LOGGER.debug(format, *args)

        def reply(self, status: int, content, content_type="application/json") -> None:
            body = (
                json.dumps(clean_json(content), allow_nan=False).encode()
                if content_type == "application/json" else content
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self'; script-src 'self'; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def trusted_host(self) -> bool:
            port = self.server.server_port
            return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def do_GET(self):
            if not self.trusted_host():
                self.reply(403, {"error": "Use localhost or 127.0.0.1 with the server port"})
                return
            if self.path == "/api/status":
                self.reply(200, dashboard.snapshot())
            elif self.path == "/api/mappings":
                with dashboard.lock:
                    self.reply(200, {"mappings": dashboard.mappings, "actions": ACTIONS})
            elif self.path == "/api/devices":
                try:
                    import sounddevice as sd
                    devices = [
                        {"id": index, "name": device["name"]}
                        for index, device in enumerate(sd.query_devices())
                        if device["max_input_channels"] > 0
                    ]
                    self.reply(200, devices)
                except Exception as error:
                    LOGGER.exception("Could not enumerate audio input devices")
                    self.reply(503, {"error": str(error)})
            elif self.path in {
                "/", "/dashboard.css", "/dashboard-actions.css", "/dashboard.js"
            }:
                name = "index.html" if self.path == "/" else self.path[1:]
                kind = {"html": "text/html", "css": "text/css", "js": "text/javascript"}[
                    name.rsplit(".", 1)[1]
                ]
                self.reply(200, (STATIC / name).read_bytes(), kind + "; charset=utf-8")
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            origin = self.headers.get("Origin")
            if (
                not self.trusted_host()
                or (origin is not None and origin != f"http://{self.headers.get('Host')}")
                or not secrets.compare_digest(self.headers.get("X-ME2-Token", ""), dashboard.token)
            ):
                self.reply(403, {"error": "Open this dashboard locally and refresh before retrying"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    self.reply(413, {"error": "Expected a JSON request of at most 16 KiB"})
                    return
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise TypeError("Expected a JSON object")
                if self.path == "/api/start":
                    dashboard.start()
                elif self.path == "/api/stop":
                    dashboard.stop()
                elif self.path == "/api/mappings":
                    dashboard.save_mappings(payload)
                elif self.path == "/api/demo":
                    dashboard.demo_command(payload.get("intent"), payload.get("accepted"))
                else:
                    self.reply(404, {"error": "Not found"})
                    return
                self.reply(200, {"ok": True})
            except (ValueError, TypeError) as error:
                self.reply(400, {"error": str(error)})
            except OSError as error:
                LOGGER.exception("Dashboard request failed")
                self.reply(500, {"error": str(error)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/dashboard.json")
    parser.add_argument("--demo", action="store_true", help="Synthetic pipeline; no microphone/models")
    parser.add_argument("--autostart", action="store_true", help="Start listening when server starts")
    parser.add_argument("--allow-unverified-wake-model", action="store_true")
    parser.add_argument("--input-device", help="Override configured microphone name or index")
    parser.add_argument("--wake-model", type=Path)
    parser.add_argument("--vcm-model", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-file", type=Path, default=ROOT / "logs/dashboard.log")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.threads < 1:
        parser.error("Choose a port from 1 to 65535 and a positive thread count")
    configure_logging(args.log_file)
    settings = load_settings(args.config.resolve())
    for key in ("wake_model", "vcm_model"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key).resolve()
    if args.input_device is not None:
        settings["input_device"] = (
            int(args.input_device) if args.input_device.isdecimal() else args.input_device
        )
    torch.set_num_threads(args.threads)
    dashboard = Dashboard(
        settings, demo=args.demo, allow_unverified=args.allow_unverified_wake_model
    )
    server = make_server(dashboard, args.port)

    def shutdown(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown)
    print(f"ME2 dashboard: http://127.0.0.1:{server.server_port}", flush=True)
    if args.autostart:
        dashboard.start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.close()
        server.server_close()


if __name__ == "__main__":
    main()
