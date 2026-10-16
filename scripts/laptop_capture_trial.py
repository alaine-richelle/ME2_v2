"""One explicitly approved, supervised microphone trial; never saves audio.

Supports a preserved baseline source tree via --source-root. Only logging and
measurement are instrumented; the selected tree supplies the actual pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--device", required=True, help="Use the confirmed device name; indices can change")
    p.add_argument("--trial-id", required=True)
    p.add_argument("--expected-text", default="", help="Public scripted phrase, used only to check grammar")
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--microphone-approved", action="store_true")
    p.add_argument("--allow-unverified-wake-model", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def stop_child(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def worker(args):
    sys.path.insert(0, str(args.source_root.resolve() / "src"))
    import numpy as np
    import sounddevice as sd
    import torch

    from me2.app import (
        _model,
        _prepare_vcm_audio,
        _print_vcm_result,
        _require_live_wake_readiness,
        build_streaming_pipeline,
        parse_args,
    )
    from me2.logging_utils import _FIELDS, configure_logging, log_event
    from me2.streaming import run_microphone_stream
    from me2.vcm.ctc import coerce_grammar_entries, normalize_transcript, required_ctc_frames

    # Baseline runtime lacks these measurement keys. Extend its log allowlist
    # only; no detector/decoder settings or decision logic are overridden.
    _FIELDS.update({"trial_id", "diagnostics", "mock_dispatch", "input_device_index",
                    "implementation_sha256", "accepted_intent"})
    source_bytes = b"".join(
        path.read_bytes() for path in sorted((args.source_root / "src/me2").rglob("*.py"))
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("x", encoding="utf-8"):
        pass
    configure_logging(args.log, context={
        "trial_id": args.trial_id,
        "implementation_sha256": hashlib.sha256(source_bytes).hexdigest(),
    })
    logger = logging.getLogger("me2.trial")
    stop = threading.Event()
    timer = None
    old_handler = signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        runtime = parse_args(["--interactive", "--config", str(args.config),
                              "--input-device", args.device])
        torch.set_num_threads(runtime.threads)
        wm, wb = _model(runtime.wake_model, "wake_word")
        cm, cb = _model(runtime.vcm_model, "vcm")
        _require_live_wake_readiness(wb, allow_unverified=args.allow_unverified_wake_model)
        device = int(args.device) if args.device.isdecimal() else args.device
        info = sd.query_devices(device, "input")
        sd.check_input_settings(device=device, channels=1, dtype="float32",
                                samplerate=wb["audio"]["sample_rate"])
        print(f"Confirmed input: {info['name']}; mono {wb['audio']['sample_rate']} Hz", flush=True)
        normalized = normalize_transcript(args.expected_text)
        legal = {entry.text for entry in coerce_grammar_entries(cb["grammar"])}
        reference = {
            "expected_phrase_supported": normalized in legal,
            "normalization_changed": normalized != args.expected_text,
            "expected_required_frames": required_ctc_frames(normalized),
            "reference_diagnostic": "supported_grammar_terminal" if normalized in legal
                                    else "unsupported_or_normalization_mismatched_phrase",
        }
        observed = {"wake_events": 0, "accepted": 0, "rejected": 0, "samples": 0}
        capture_info = {}
        def event(e):
            if e.message == "wake_detected":
                observed["wake_events"] += 1
            print(f"[{e.state.value}] {e.message}", flush=True)
            if e.result is not None:
                accepted = e.result.actionable_intent is not None
                observed["accepted" if accepted else "rejected"] += 1
                _print_vcm_result(e.result)
                diagnostics = {
                    **capture_info, **dict(getattr(e.result, "diagnostics", {})),
                    **(reference if args.expected_text else {}),
                }
                if args.expected_text and "valid_output_frames" in diagnostics:
                    diagnostics["expected_alignment_feasible"] = (
                        diagnostics["valid_output_frames"] >= reference["expected_required_frames"]
                    )
                log_event(
                    logger, "trial_result", trial_id=args.trial_id, utterance_id=e.utterance_id,
                    sample_index=e.sample_index, accepted=accepted,
                    accepted_intent=e.result.actionable_intent if accepted else None,
                    reason=e.result.rejection_reason, system_error=e.result.system_error,
                    candidate_count=e.result.candidate_count, diagnostics=diagnostics,
                    mock_dispatch="would_dispatch" if accepted else "suppressed",
                )
        pipeline = build_streaming_pipeline(runtime, wm, wb, cm, cb, on_event=event)
        original_predict = pipeline.command_predictor
        def predict(audio):
            capture_info.clear()
            capture_info["captured_samples"] = len(audio)
            prepared, usable = _prepare_vcm_audio(cb, audio, wb["audio"]["sample_rate"])
            usable = len(prepared) if usable is None else usable
            frames = usable // cb["features"]["hop_length"] + 1
            capture_info.update(
                usable_samples=usable, valid_feature_frames=frames,
                valid_output_frames=int(cm.output_lengths(torch.tensor([frames]))[0]),
            )
            return original_predict(audio)
        pipeline.command_predictor = predict
        def audio(chunk):
            observed["samples"] += len(chunk)
            log_event(logger, "trial_audio_level", trial_id=args.trial_id,
                      sample_index=observed["samples"], diagnostics={
                          "chunk_rms": float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))),
                          "chunk_peak": float(np.max(np.abs(chunk))),
                          "clipped_samples": int(np.count_nonzero(np.abs(chunk) >= .999)),
                      })
        log_event(logger, "trial_ready", trial_id=args.trial_id, duration_seconds=args.seconds,
                  sample_rate=wb["audio"]["sample_rate"],
                  diagnostics=reference if args.expected_text else {})
        print(f"READY — trial {args.trial_id}, {args.seconds:g} seconds; Ctrl+C stops.", flush=True)
        started = time.monotonic()
        timer = threading.Timer(args.seconds, stop.set)
        timer.daemon = True
        timer.start()
        run_microphone_stream(pipeline, input_device=device, stop_event=stop, on_audio=audio)
        duration = observed["samples"] / pipeline.config.sample_rate
        log_event(logger, "trial_completed", trial_id=args.trial_id,
                  duration_seconds=duration, total_ms=(time.monotonic() - started) * 1000,
                  mock_dispatch="dry_run_only")
        print(json.dumps({**observed, "audio_seconds": duration, "raw_audio_saved": False,
                          "final_state": pipeline.state.value}), flush=True)
    except KeyboardInterrupt:
        stop.set()
        log_event(logger, "trial_interrupted", trial_id=args.trial_id)
    except Exception as error:  # noqa: BLE001 -- live CLI boundary: log safe type and fail.
        log_event(logger, "trial_failed", level=logging.ERROR, trial_id=args.trial_id,
                  error_type=type(error).__name__)
        print(f"Trial failed: {type(error).__name__}; check device/configuration and safe log.", flush=True)
        raise SystemExit(1) from None
    finally:
        if timer is not None:
            timer.cancel()
        stop.set()
        signal.signal(signal.SIGTERM, old_handler)
        logging.shutdown()


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.microphone_approved:
        parser.error("Obtain explicit approval for this trial, then use --microphone-approved")
    if not 5 <= args.seconds <= 60:
        parser.error("--seconds must be between 5 and 60")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.trial_id):
        parser.error("trial ID must be a short non-private token")
    if args.log.exists():
        parser.error("Use a new log path; previous trial evidence will not be overwritten")
    if args.worker:
        worker(args)
        return
    process = subprocess.Popen(
        [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
        start_new_session=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    try:
        code = process.wait(timeout=args.seconds + 15)
    except subprocess.TimeoutExpired:
        stop_child(process)
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "ERROR", "component": "me2.trial.supervisor",
                "trial_id": args.trial_id, "event": "trial_watchdog_stop",
                "system_error": "CaptureWatchdogTimeout", "raw_audio_saved": False,
            }) + "\n")
        print("Trial watchdog stopped a stalled audio process; outcome is a system fault.")
        code = 1
    except KeyboardInterrupt:
        stop_child(process)
        code = 130
    raise SystemExit(code)


if __name__ == "__main__":
    main()
