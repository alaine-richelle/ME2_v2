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
from me2.logging_utils import configure_logging
from me2.streaming import (
    PipelineEvent,
    PipelineState,
    StreamingConfig,
    StreamingPipeline,
    run_microphone_stream,
)
from me2.vcm.ctc import CommandResult, coerce_grammar_entries

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).with_name("dashboard_static")
INTENTS = (
    "play_music", "weather", "time", "light_on", "light_off", "pause", "stop",
    "next", "volume_up", "volume_down", "call", "message", "list_reminders",
    "timer", "alarm", "temperature", "brightness", "color", "create_reminder",
)
# Every handler acts only on the dashboard's in-memory demonstration devices.
ACTIONS = {
    "display": "Display recognized command",
    "light_on": "Demo light · on",
    "light_off": "Demo light · off",
    "play_music": "Demo player · play",
    "pause": "Demo player · pause",
    "stop": "Demo player · stop",
    "next": "Demo player · next track",
    "volume_up": "Demo volume · +10",
    "volume_down": "Demo volume · −10",
}


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


def load_settings(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("wake_model", "vcm_model", "mappings_file"):
        value = Path(config[key]).expanduser()
        config[key] = value if value.is_absolute() else (path.parent / value).resolve()
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
            intent: {"enabled": True, "action": intent if intent in ACTIONS else "display"}
            for intent in INTENTS
        }
        mappings_file = settings["mappings_file"]
        if mappings_file.exists():
            self.mappings = self.validate_mappings(
                json.loads(mappings_file.read_text(encoding="utf-8"))
            )
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
            "models": {},
            "last_result": None,
            "last_action": None,
            "counts": {"wake": 0, "accepted": 0, "rejected": 0},
            "devices": {"light": False, "player": "stopped", "volume": 50, "track": 1},
            "supported_intents": list(INTENTS) if demo else [],
        }
        self.process = psutil.Process()
        self.process.cpu_percent()
        self.metrics_at = 0.0
        self.metrics: dict = {}

    @staticmethod
    def validate_mappings(mappings: dict) -> dict:
        if not isinstance(mappings, dict) or set(mappings) != set(INTENTS):
            raise ValueError("Mappings must contain exactly the 19 supported intent names")
        for value in mappings.values():
            if not isinstance(value, dict) or set(value) != {"enabled", "action"}:
                raise ValueError("Each mapping requires enabled and action")
            if type(value["enabled"]) is not bool or value["action"] not in ACTIONS:
                raise ValueError("Mapping requires a boolean and a listed action")
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
            self.worker = threading.Thread(target=self._run, name="me2-audio", daemon=True)
            self.worker.start()

    def stop(self) -> None:
        with self.lock:
            self.stop_requested.set()
            if self.data["running"]:
                self.data["stopping"] = True
                self.data["state"] = "STOPPING"

    def close(self) -> None:
        self.stop()
        if self.worker:
            self.worker.join(timeout=8)

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

    def on_wake(self, label: str, score: float, elapsed: float, detected: bool) -> None:
        with self.lock:
            self.data["wake"] = {
                "label": label, "score": score, "inference_ms": elapsed,
                "detected": detected, "at": now(),
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
                "intent": None, "action": None,
            }
            if not self.stop_requested.is_set():
                self.data["state"] = event.state.value
            if event.message == "wake_detected":
                self.data["counts"]["wake"] += 1
            if event.result is not None:
                result = event.result
                self.data["last_result"] = {
                    **dataclasses.asdict(result), "actionable_intent": result.actionable_intent,
                }
                key = "accepted" if result.accepted else "rejected"
                self.data["counts"][key] += 1
                action = self.dispatch(result)
                self.data["last_action"] = action
                record.update(intent=result.actionable_intent, action=action)
            self.events.appendleft(record)
            LOGGER.info("pipeline_event=%s", json.dumps(clean_json(record), allow_nan=False))

    def dispatch(self, result: CommandResult) -> str:
        """Dispatch only an accepted, mapped intent to an allowlisted demo action."""
        if self.stop_requested.is_set():
            return "Skipped · pipeline stopping"
        intent = result.actionable_intent
        if intent is None:
            return "Skipped · command rejected"
        mapping = self.mappings.get(intent)
        if mapping is None or not mapping["enabled"]:
            return "Skipped · mapping disabled or unavailable"
        action = mapping["action"]
        devices = self.data["devices"]
        if action in {"light_on", "light_off"}:
            devices["light"] = action == "light_on"
        elif action in {"play_music", "pause", "stop"}:
            devices["player"] = {"play_music": "playing", "pause": "paused", "stop": "stopped"}[
                action
            ]
        elif action == "next":
            devices["track"] += 1
        elif action in {"volume_up", "volume_down"}:
            delta = 10 if action == "volume_up" else -10
            devices["volume"] = min(100, max(0, devices["volume"] + delta))
        return ACTIONS[action]

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
                    "speech_rms_threshold",
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
            else:
                intents = [label for label in vcm_bundle["labels"] if label != "out_of_scope"]
                examples = []
            models = {}
            for name, path, bundle in (
                ("wake", wake_path, wake_bundle), ("command", vcm_path, vcm_bundle),
            ):
                models[name] = {
                    "path": str(path), "architecture": bundle.get("architecture"),
                    "version": bundle.get("model_version"),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "deployment_ready": bundle.get("deployment_ready"),
                    "audio": bundle["audio"], "threshold": bundle.get("threshold"),
                    "streaming_policy": bundle.get("streaming_policy", {}),
                    "grammar_version": bundle.get("grammar_version"),
                }
            with self.lock:
                self.data.update(models=models, supported_intents=intents)
                self.data["grammar"] = examples
            _require_live_wake_readiness(wake_bundle, allow_unverified=self.allow_unverified)
            import sounddevice as sd

            sd.check_input_settings(
                device=args.input_device, channels=1, dtype="float32",
                samplerate=int(wake_bundle["audio"]["sample_rate"]),
            )
            device = sd.query_devices(args.input_device, kind="input")["name"]
            pipeline = build_streaming_pipeline(
                args, wake_model, wake_bundle, vcm_model, vcm_bundle,
                on_event=self.on_event, on_wake=self.on_wake,
            )
            with self.lock:
                self.data.update(
                    device=device,
                    notice=(
                        "Engineering override · wake model has not passed deployment review"
                        if wake_bundle.get("deployment_ready") is False
                        else "Live microphone · actions update simulated dashboard devices"
                    ),
                )
            if self.stop_requested.is_set():
                return
            self.on_event(PipelineEvent(pipeline.state, 0, "listening_started"))
            run_microphone_stream(
                pipeline, input_device=args.input_device,
                stop_event=self.stop_requested, on_audio=self.on_audio,
            )
        except Exception as error:
            LOGGER.exception("Dashboard pipeline failed")
            with self.lock:
                self.data["error"] = str(error)
        finally:
            if pipeline is not None:
                pipeline.stop()
            with self.lock:
                self.data.update(
                    running=False, stopping=False, rms=0.0,
                    state="ERROR" if self.data["error"] else "STOPPED",
                )

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
                slots={}, variation="", raw_log_score=-1.0, normalized_log_score=-0.1,
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
            elif self.path in {"/", "/dashboard.css", "/dashboard.js"}:
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
