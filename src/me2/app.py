"""Combined wake-word -> voice-command command-line application."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from me2.audio import AudioConfig, load_wav, prepare_audio
from me2.bundle import load_bundle, predict_waveform
from me2.logging_utils import configure_logging

LOGGER = logging.getLogger("me2")
ROOT = Path(__file__).resolve().parents[2]


def _model(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}. Train that component first.")
    return load_bundle(path)


def _wake_result(model, bundle: dict, waveform: np.ndarray) -> tuple[bool, str, float, float]:
    started = time.perf_counter()
    label, _, probabilities = predict_waveform(model, bundle, waveform)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    wake_probability = probabilities[bundle["labels"].index("ateko")]
    threshold = float(bundle.get("threshold", 0.7))
    detected = label == "ateko" and wake_probability >= threshold
    return detected, label, wake_probability, elapsed_ms


def _capture(duration_seconds: float, sample_rate: int, input_device: str | int | None = None) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError(
            "Microphone input requires the optional dependency: pip install -e '.[microphone]'"
        ) from error
    try:
        device = sd.query_devices(kind="input") if input_device is None else sd.query_devices(input_device)
        LOGGER.info("Using microphone: %s", device.get("name", "selected input"))
        recording = sd.rec(
            int(duration_seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=input_device,
        )
        sd.wait()
    except Exception as error:
        raise RuntimeError(f"Microphone capture failed: {error}") from error
    return prepare_audio(recording[:, 0], sample_rate, AudioConfig(sample_rate, duration_seconds))


def _predict_vcm(model, bundle: dict, waveform: np.ndarray) -> tuple[str, float, float]:
    started = time.perf_counter()
    command, confidence, _ = predict_waveform(model, bundle, waveform)
    return command, confidence, (time.perf_counter() - started) * 1000.0


def _run_wake(args: argparse.Namespace) -> None:
    model, bundle = _model(args.wake_model)
    if args.audio:
        waveform = load_wav(args.audio, AudioConfig(**bundle["audio"]))
        detected, label, score, elapsed = _wake_result(model, bundle, waveform)
        print(f"Wake class: {label}; wake confidence: {score:.3f}; inference: {elapsed:.1f} ms")
        print("Wake word detected!" if detected else "Wake word not detected.")
        LOGGER.info("Wake file=%s detected=%s class=%s confidence=%.4f", args.audio, detected, label, score)
        return
    print("[LISTENING] Waiting for wake word...")
    duration = float(bundle["audio"]["duration_seconds"])
    rate = int(bundle["audio"]["sample_rate"])
    while True:
        waveform = _capture(duration, rate, args.input_device)
        detected, label, score, elapsed = _wake_result(model, bundle, waveform)
        LOGGER.info("Wake window class=%s confidence=%.4f inference_ms=%.2f", label, score, elapsed)
        if detected:
            print('[WAKE WORD] "ateko" detected')
            return


def _run_vcm(args: argparse.Namespace) -> None:
    model, bundle = _model(args.vcm_model)
    if args.audio:
        waveform = load_wav(args.audio, AudioConfig(**bundle["audio"]))
    else:
        print("[LISTENING] Waiting for command...")
        waveform = _capture(
            float(bundle["audio"]["duration_seconds"]),
            int(bundle["audio"]["sample_rate"]),
            args.input_device,
        )
    command, confidence, elapsed = _predict_vcm(model, bundle, waveform)
    print(f"[VCM] Command: {command}")
    print(f"[VCM] Confidence: {confidence:.2f}")
    LOGGER.info("VCM audio=%s command=%s confidence=%.4f inference_ms=%.2f", args.audio, command, confidence, elapsed)


def _run_interactive(args: argparse.Namespace) -> None:
    wake_model, wake_bundle = _model(args.wake_model)
    vcm_model, vcm_bundle = _model(args.vcm_model)
    microphone = args.wake_audio is None
    if microphone and args.command_audio:
        raise ValueError("--command-audio requires --wake-audio for file-based pipeline testing")

    while True:
        print("[LISTENING] Waiting for wake word...")
        if args.wake_audio:
            wake_waveform = load_wav(args.wake_audio, AudioConfig(**wake_bundle["audio"]))
        else:
            wake_waveform = _capture(
                float(wake_bundle["audio"]["duration_seconds"]),
                int(wake_bundle["audio"]["sample_rate"]),
                args.input_device,
            )
        detected, label, wake_score, wake_ms = _wake_result(wake_model, wake_bundle, wake_waveform)
        LOGGER.info(
            "Wake input=%s detected=%s class=%s confidence=%.4f inference_ms=%.2f",
            args.wake_audio or "microphone",
            detected,
            label,
            wake_score,
            wake_ms,
        )
        if not detected:
            print("[STATUS] Wake word not detected; remaining in wake listening.")
            if args.wake_audio:
                return
            continue

        print('[WAKE WORD] "ateko" detected')
        print("[ASSISTANT] Yes?")
        print("[LISTENING] Waiting for command...")
        if args.command_audio:
            command_waveform = load_wav(args.command_audio, AudioConfig(**vcm_bundle["audio"]))
        else:
            command_waveform = _capture(
                float(vcm_bundle["audio"]["duration_seconds"]),
                int(vcm_bundle["audio"]["sample_rate"]),
                args.input_device,
            )
        command, confidence, inference_ms = _predict_vcm(vcm_model, vcm_bundle, command_waveform)
        print(f"[VCM] Command: {command}")
        print(f"[VCM] Confidence: {confidence:.2f}")
        LOGGER.info("VCM command=%s confidence=%.4f inference_ms=%.2f", command, confidence, inference_ms)
        print("[STATUS] Returning to wake-word detection...")
        LOGGER.info("Returning to wake-word listening")
        if args.wake_audio:
            return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--wake-word", action="store_true", help="Test/listen with the wake model only")
    modes.add_argument("--vcm", action="store_true", help="Test/listen with the command model only")
    modes.add_argument("--interactive", action="store_true", help="Run the full wake -> command loop")
    parser.add_argument("--audio", type=Path, help="WAV input for an individual component")
    parser.add_argument("--wake-audio", type=Path, help="Wake-word WAV for full-pipeline testing")
    parser.add_argument("--command-audio", type=Path, help="Command WAV after a successful wake gate")
    parser.add_argument("--input-device", help="Microphone device name or index (default: system input)")
    parser.add_argument("--wake-model", type=Path, default=ROOT / "wake-word/models/wake_word_model.pkl")
    parser.add_argument("--vcm-model", type=Path, default=ROOT / "vcm/models/vcm_model.pkl")
    parser.add_argument("--log-file", type=Path, default=ROOT / "logs/app.log")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.audio and args.interactive:
        parser.error("Use --wake-audio and optionally --command-audio for combined WAV testing")
    if args.wake_audio and not args.interactive:
        parser.error("--wake-audio is only valid with --interactive")
    if args.command_audio and not args.interactive:
        parser.error("--command-audio is only valid with --interactive")
    if args.input_device and args.input_device.isdecimal():
        args.input_device = int(args.input_device)
    configure_logging(args.log_file)
    try:
        if args.wake_word:
            _run_wake(args)
        elif args.vcm:
            _run_vcm(args)
        else:
            _run_interactive(args)
    except KeyboardInterrupt:
        print("\n[STATUS] Stopped.")
    except Exception as error:
        LOGGER.exception("Inference failed")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
