"""Local voice-command-only dashboard with browser-side detection feedback."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib
import json
import logging
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

from me2.bundle import load_bundle, predict_command
from me2.logging_utils import configure_logging, log_event
from me2.vcm.ctc import CommandResult, coerce_grammar_entries

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).with_name("vcm_dashboard_static")
SHARED_STATIC = Path(__file__).with_name("dashboard_static")
INTENTS = ("play_music", "weather", "time", "light_on", "light_off", "pause", "stop", "next", "volume_up", "volume_down", "call", "message", "list_reminders", "timer", "alarm", "temperature", "brightness", "color", "create_reminder")
ACTIONS = {intent: intent.replace("_", " ") for intent in INTENTS}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_json(value):
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, dict): return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [clean_json(item) for item in value]
    return value


def load_settings(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version", 1) != 1: raise ValueError("voice-command dashboard schema_version must be 1")
    for key in ("vcm_model", "mappings_file"):
        value = Path(config[key]).expanduser(); config[key] = value if value.is_absolute() else (path.parent / value).resolve()
    device = config.get("input_device")
    config["input_device"] = int(device) if isinstance(device, str) and device.isdecimal() else device
    cooldown = config.get("cooldown_seconds", 1.0)
    if not isinstance(cooldown, (int, float)) or not 0 <= cooldown <= 10: raise ValueError("cooldown_seconds must be between 0 and 10")
    config["cooldown_seconds"] = float(cooldown)
    return config


def _load_vcm_model(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}. Train that component first.")
    model, bundle = load_bundle(path)
    if bundle.get("task") != "vcm":
        raise ValueError(f"Model task mismatch for {path}: expected 'vcm'")
    return model, bundle


def _predict_command(model, bundle: dict, waveform: np.ndarray):
    started = time.perf_counter()
    result = predict_command(model, bundle, waveform, sample_length=waveform.size)
    return result, (time.perf_counter() - started) * 1000.0


class VoiceCommandDashboard:
    """Direct VCM microphone loop; it has no wake-word model or state."""
    def __init__(self, settings: dict, *, demo: bool) -> None:
        self.settings, self.demo, self.lock = settings, demo, threading.RLock()
        self.stop_requested, self.worker = threading.Event(), None
        self.token, self.events, self.sequence = secrets.token_urlsafe(32), deque(maxlen=80), 0
        self.model = self.bundle = None
        self.mappings = {intent: {"enabled": True, "action": intent} for intent in INTENTS}
        path = settings["mappings_file"]
        if path.exists(): self.mappings = self.validate_mappings(json.loads(path.read_text(encoding="utf-8")))
        self.demo_requests: queue.Queue[dict] = queue.Queue(maxsize=1)
        self.process, self.metrics_at, self.metrics = psutil.Process(), 0.0, {}
        self.process.cpu_percent()
        self.data = {"mode": "demo" if demo else "microphone", "state": "STOPPED", "running": False, "stopping": False, "error": None, "notice": "Synthetic demo · no microphone or model predictions" if demo else "Microphone is off. Start listening to load the voice-command model.", "device": settings.get("input_device"), "rms": 0.0, "models": {}, "grammar": [], "supported_intents": list(INTENTS) if demo else [], "last_result": None, "last_action": None, "last_action_status": None, "counts": {"accepted": 0, "rejected": 0}}

    @staticmethod
    def validate_mappings(mappings: dict) -> dict:
        if not isinstance(mappings, dict) or set(mappings) != set(INTENTS): raise ValueError("Mappings must contain exactly the 19 supported intent names")
        for intent, value in mappings.items():
            if not isinstance(value, dict) or set(value) != {"enabled", "action"} or type(value["enabled"]) is not bool or value["action"] != intent: raise ValueError(f"{intent} must use its canonical dashboard action")
        return copy.deepcopy(mappings)

    def save_mappings(self, mappings: dict) -> None:
        validated = self.validate_mappings(mappings)
        with self.lock:
            path = self.settings["mappings_file"]; path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp"); temporary.write_text(json.dumps(validated, indent=2) + "\n", encoding="utf-8"); temporary.replace(path); self.mappings = validated

    def snapshot(self) -> dict:
        with self.lock:
            if time.monotonic() - self.metrics_at >= 2:
                self.metrics = {"cpu_percent": self.process.cpu_percent(), "rss_mb": round(self.process.memory_info().rss / 1024**2, 1)}; self.metrics_at = time.monotonic()
            return clean_json(copy.deepcopy({**self.data, "events": list(self.events), "metrics": self.metrics, "token": self.token}))

    def start(self) -> None:
        with self.lock:
            if self.worker is not None and self.worker.is_alive(): raise ValueError("The pipeline is already running or stopping")
            self.stop_requested.clear(); self.data.update(running=True, stopping=False, state="STARTING", error=None, last_result=None, last_action=None, rms=0.0)
            self.worker = threading.Thread(target=self._run, name="me2-vcm-audio", daemon=True); self.worker.start()

    def stop(self) -> None:
        with self.lock:
            self.stop_requested.set()
            if self.data["running"]: self.data.update(stopping=True, state="STOPPING")

    def close(self) -> None:
        self.stop()
        if self.worker: self.worker.join(timeout=8)

    def demo_command(self, intent: str, accepted: bool) -> None:
        if not self.demo or not self.data["running"] or intent not in INTENTS or type(accepted) is not bool: raise ValueError("Start demo listening and choose a known intent")
        self.demo_requests.put_nowait({"intent": intent, "accepted": accepted})

    def _record(self, message: str, result: CommandResult) -> None:
        with self.lock:
            self.sequence += 1; event = {"id": self.sequence, "at": now(), "message": message, "intent": result.actionable_intent, "action": None}
            self.data["last_result"] = dataclasses.asdict(result) | {"actionable_intent": result.actionable_intent}
            if result.accepted and result.actionable_intent and self.mappings.get(result.actionable_intent, {}).get("enabled"):
                action = ACTIONS[result.actionable_intent]; self.data.update(last_action_status="success", last_action=action); self.data["counts"]["accepted"] += 1; event["action"] = action
            else:
                self.data.update(last_action_status="skipped" if result.accepted else "rejected", last_action="Skipped · mapping disabled" if result.accepted else "Rejected · no action dispatched"); self.data["counts"]["rejected"] += 1
            self.events.appendleft(event); log_event(LOGGER, "vcm_dashboard_event", reason=message, accepted=result.accepted)

    def _set_model_metadata(self) -> None:
        assert self.bundle is not None
        grammar = [dataclasses.asdict(entry) for entry in coerce_grammar_entries(self.bundle["grammar"])] if self.bundle.get("grammar") else []
        intents = sorted({entry["intent"] for entry in grammar}) if grammar else [label for label in self.bundle["labels"] if label != "out_of_scope"]
        path = self.settings["vcm_model"]
        self.data.update(models={"command": {"role": "voice_command", "path": str(path), "architecture": self.bundle.get("architecture"), "version": self.bundle.get("model_version"), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "audio": self.bundle["audio"], "grammar_version": self.bundle.get("grammar_version"), "status": "ready"}}, grammar=grammar, supported_intents=intents)

    def _run(self) -> None:
        try:
            if self.demo: self._run_demo(); return
            self.model, self.bundle = _load_vcm_model(self.settings["vcm_model"])
            with self.lock: self._set_model_metadata(); self.data.update(state="LISTENING", notice="Live microphone · direct voice-command recognition")
            self._run_microphone()
        except Exception as error:
            LOGGER.exception("Voice-command dashboard failed")
            with self.lock: self.data["error"] = str(error)
        finally:
            with self.lock: self.data.update(running=False, stopping=False, state="ERROR" if self.data["error"] else "STOPPED")

    def _run_demo(self) -> None:
        with self.lock: self.data.update(state="LISTENING", notice="Synthetic demo · no microphone or model predictions")
        while not self.stop_requested.wait(.1):
            try: request = self.demo_requests.get_nowait()
            except queue.Empty: continue
            accepted = request["accepted"]
            self._record("command_detected" if accepted else "command_rejected", CommandResult(decoded_text=request["intent"].replace("_", " ") if accepted else "unknown request", intent=request["intent"] if accepted else None, slots={}, variation="", raw_log_score=-1.0, normalized_log_score=-.1, candidate_posterior=.94, margin=.4, accepted=accepted, rejection_reason=None if accepted else "demo_out_of_scope", model_version="synthetic-demo"))

    def _run_microphone(self) -> None:
        try: sd = importlib.import_module("sounddevice")
        except ImportError as error: raise RuntimeError("Microphone input requires: pip install -e '.[microphone]'") from error
        assert self.model is not None and self.bundle is not None
        sample_rate = int(self.bundle["audio"]["sample_rate"]); window_samples = round(sample_rate * float(self.bundle["audio"]["duration_seconds"])); chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=4); buffer = np.empty(0, dtype=np.float32)
        def callback(indata, frames, timing, status):
            del frames, timing
            if status: LOGGER.warning("VCM microphone callback status: %s", status)
            try: chunks.put_nowait(indata[:, 0].copy())
            except queue.Full: LOGGER.warning("VCM microphone queue full; dropping audio")
        device = sd.query_devices(self.settings.get("input_device"), kind="input")["name"]
        with self.lock: self.data["device"] = device
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", device=self.settings.get("input_device"), blocksize=min(1600, window_samples), callback=callback):
            while not self.stop_requested.is_set():
                try: chunk = chunks.get(timeout=.2)
                except queue.Empty: continue
                with self.lock: self.data["rms"] = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
                buffer = np.concatenate((buffer, chunk))
                while buffer.size >= window_samples:
                    waveform, buffer = buffer[:window_samples], buffer[window_samples:]
                    with self.lock: self.data["state"] = "DECODING"
                    result, _ = _predict_command(self.model, self.bundle, waveform); self._record("command_detected" if result.accepted else "command_rejected", result)
                    with self.lock: self.data["state"] = "LISTENING"
                    if self.stop_requested.wait(self.settings["cooldown_seconds"]): return


def make_server(dashboard: VoiceCommandDashboard, port: int = 8766) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args): LOGGER.debug(format, *args)
        def reply(self, status: int, content, content_type="application/json") -> None:
            body = json.dumps(clean_json(content), allow_nan=False).encode() if content_type == "application/json" else content
            self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.send_header("X-Content-Type-Options", "nosniff"); self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self';"); self.end_headers(); self.wfile.write(body)
        def trusted_host(self) -> bool: return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        def do_GET(self):
            if not self.trusted_host(): self.reply(403, {"error": "Use localhost or 127.0.0.1 with the server port"}); return
            if self.path == "/api/status": self.reply(200, dashboard.snapshot())
            elif self.path == "/api/mappings": self.reply(200, {"mappings": dashboard.mappings, "actions": ACTIONS})
            elif self.path in {"/", "/dashboard.css", "/dashboard-actions.css", "/dashboard.js"}:
                name = "index.html" if self.path == "/" else self.path[1:]; source = STATIC / name if (STATIC / name).exists() else SHARED_STATIC / name; kind = {"html": "text/html", "css": "text/css", "js": "text/javascript"}[name.rsplit(".", 1)[1]]; self.reply(200, source.read_bytes(), kind + "; charset=utf-8")
            else: self.reply(404, {"error": "Not found"})
        def do_POST(self):
            origin = self.headers.get("Origin")
            if not self.trusted_host() or (origin is not None and origin != f"http://{self.headers.get('Host')}") or not secrets.compare_digest(self.headers.get("X-ME2-Token", ""), dashboard.token): self.reply(403, {"error": "Open this dashboard locally and refresh before retrying"}); return
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                if self.path == "/api/start": dashboard.start()
                elif self.path == "/api/stop": dashboard.stop()
                elif self.path == "/api/mappings": dashboard.save_mappings(payload)
                elif self.path == "/api/demo": dashboard.demo_command(payload.get("intent"), payload.get("accepted"))
                else: self.reply(404, {"error": "Not found"}); return
                self.reply(200, {"ok": True})
            except (ValueError, TypeError, json.JSONDecodeError, queue.Full) as error: self.reply(400, {"error": str(error)})
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "config/vcm-dashboard.json"); parser.add_argument("--demo", action="store_true"); parser.add_argument("--autostart", action="store_true"); parser.add_argument("--vcm-model", type=Path); parser.add_argument("--input-device"); parser.add_argument("--port", type=int, default=8766); parser.add_argument("--threads", type=int, default=2); parser.add_argument("--log-file", type=Path, default=ROOT / "logs/vcm-dashboard.log"); args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.threads < 1: parser.error("Choose a valid port and positive thread count")
    configure_logging(args.log_file); settings = load_settings(args.config.resolve())
    if args.vcm_model: settings["vcm_model"] = args.vcm_model.resolve()
    if args.input_device is not None: settings["input_device"] = int(args.input_device) if args.input_device.isdecimal() else args.input_device
    torch.set_num_threads(args.threads); dashboard = VoiceCommandDashboard(settings, demo=args.demo); server = make_server(dashboard, args.port)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt)); print(f"ME2 voice-command dashboard: http://127.0.0.1:{server.server_port}", flush=True)
    if args.autostart: dashboard.start()
    try: server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt: pass
    finally: dashboard.close(); server.server_close()


if __name__ == "__main__": main()
