"""Combined wake-word -> voice-command command-line application."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from me2.audio import (
    AudioConfig,
    load_wav,
    load_wav_sequence,
    prepare_audio,
    prepare_audio_sequence,
)
from me2.bundle import load_bundle, predict_command, predict_waveform
from me2.logging_utils import configure_logging
from me2.streaming import (
    PipelineEvent,
    PipelineState,
    StreamingConfig,
    StreamingPipeline,
    run_microphone_stream,
)
from me2.wake_word.detector import WakeEventDetector, WakeSmoothingPolicy

LOGGER = logging.getLogger("me2")
ROOT = Path(__file__).resolve().parents[2]


def _model(path: Path, expected_task: str):
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}. Train that component first.")
    model, bundle = load_bundle(path)
    if bundle.get("task") != expected_task:
        raise ValueError(
            f"Model task mismatch for {path}: expected {expected_task!r}, "
            f"found {bundle.get('task')!r}"
        )
    return model, bundle


def _require_live_wake_readiness(
    bundle: dict,
    *,
    allow_unverified: bool,
) -> None:
    if bundle.get("deployment_ready") is not False:
        return
    reason = (
        "Wake artifact is explicitly marked deployment_ready=false because "
        "pronunciation review and/or real-speaker validation are incomplete."
    )
    if not allow_unverified:
        raise ValueError(
            f"{reason} Use --allow-unverified-wake-model only for a controlled "
            "engineering replay or microphone experiment."
        )
    LOGGER.warning("%s Proceeding because an explicit override was supplied.", reason)


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


def _predict_vcm(
    model,
    bundle: dict,
    waveform: np.ndarray,
    *,
    sample_length: int | None = None,
):
    started = time.perf_counter()
    result = predict_command(model, bundle, waveform, sample_length=sample_length)
    return result, (time.perf_counter() - started) * 1000.0


def _load_vcm_wav(bundle: dict, path: Path) -> tuple[np.ndarray, int | None]:
    audio_config = AudioConfig(**bundle["audio"])
    if bundle.get("architecture") in {"CharacterAudioCNN", "CompactCTCModel"}:
        return load_wav_sequence(path, audio_config)
    return load_wav(path, audio_config), None


def _run_wake(args: argparse.Namespace) -> None:
    model, bundle = _model(args.wake_model, "wake_word")
    if args.audio:
        waveform = load_wav(args.audio, AudioConfig(**bundle["audio"]))
        detected, label, score, elapsed = _wake_result(model, bundle, waveform)
        print(f"Wake class: {label}; wake confidence: {score:.3f}; inference: {elapsed:.1f} ms")
        print("Wake word detected!" if detected else "Wake word not detected.")
        LOGGER.info("Wake file=%s detected=%s class=%s confidence=%.4f", args.audio, detected, label, score)
        return
    _require_live_wake_readiness(
        bundle,
        allow_unverified=args.allow_unverified_wake_model,
    )
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
    model, bundle = _model(args.vcm_model, "vcm")
    sample_length = None
    if args.audio:
        waveform, sample_length = _load_vcm_wav(bundle, args.audio)
    else:
        print("[LISTENING] Waiting for command...")
        waveform = _capture(
            float(bundle["audio"]["duration_seconds"]),
            int(bundle["audio"]["sample_rate"]),
            args.input_device,
        )
    result, elapsed = _predict_vcm(model, bundle, waveform, sample_length=sample_length)
    _print_vcm_result(result)
    LOGGER.info(
        "VCM audio=%s accepted=%s intent=%s text=%s rejection=%s "
        "score=%.4f margin=%.4f inference_ms=%.2f",
        args.audio,
        result.accepted,
        result.actionable_intent,
        result.decoded_text,
        result.rejection_reason,
        result.normalized_log_score,
        result.margin,
        elapsed,
    )


def _run_interactive(args: argparse.Namespace) -> None:
    wake_model, wake_bundle = _model(args.wake_model, "wake_word")
    vcm_model, vcm_bundle = _model(args.vcm_model, "vcm")
    microphone = args.wake_audio is None
    if microphone and args.command_audio:
        raise ValueError("--command-audio requires --wake-audio for file-based pipeline testing")
    if microphone:
        _require_live_wake_readiness(
            wake_bundle,
            allow_unverified=args.allow_unverified_wake_model,
        )
        _run_continuous_interactive(
            args, wake_model, wake_bundle, vcm_model, vcm_bundle
        )
        return

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
        command_sample_length = None
        if args.command_audio:
            command_waveform, command_sample_length = _load_vcm_wav(
                vcm_bundle, args.command_audio
            )
        else:
            command_waveform = _capture(
                float(vcm_bundle["audio"]["duration_seconds"]),
                int(vcm_bundle["audio"]["sample_rate"]),
                args.input_device,
            )
        print("[DECODING] Decoding captured command...")
        result, inference_ms = _predict_vcm(
            vcm_model,
            vcm_bundle,
            command_waveform,
            sample_length=command_sample_length,
        )
        _print_vcm_result(result)
        LOGGER.info(
            "VCM accepted=%s intent=%s text=%s slots=%s rejection=%s "
            "score=%.4f margin=%.4f inference_ms=%.2f",
            result.accepted,
            result.actionable_intent,
            result.decoded_text,
            dict(result.slots),
            result.rejection_reason,
            result.normalized_log_score,
            result.margin,
            inference_ms,
        )
        print("[COOLDOWN] Interaction complete.")
        print("[STATUS] Returning to wake-word detection...")
        LOGGER.info("Returning to wake-word listening")
        if args.wake_audio:
            return


def _run_continuous_interactive(
    args: argparse.Namespace,
    wake_model,
    wake_bundle: dict,
    vcm_model,
    vcm_bundle: dict,
) -> None:
    def on_event(event: PipelineEvent) -> None:
        if event.state == PipelineState.CAPTURING_COMMAND:
            print('[WAKE WORD] "ateko" detected')
            print("[CAPTURING_COMMAND] Listening for command...")
        elif event.state == PipelineState.DECODING:
            print("[DECODING] Endpoint detected.")
        elif event.state == PipelineState.ACCEPTED_OR_REJECTED and event.result:
            _print_vcm_result(event.result)
        elif event.state == PipelineState.COOLDOWN:
            print("[COOLDOWN] Suppressing duplicate triggers.")
        elif event.state == PipelineState.LISTENING_FOR_WAKE:
            print("[LISTENING_FOR_WAKE] Ready.")

    pipeline = build_streaming_pipeline(
        args, wake_model, wake_bundle, vcm_model, vcm_bundle, on_event=on_event
    )
    print("[LISTENING_FOR_WAKE] Waiting for ateko...")
    run_microphone_stream(pipeline, input_device=args.input_device)


def build_streaming_pipeline(
    args: argparse.Namespace,
    wake_model,
    wake_bundle: dict,
    vcm_model,
    vcm_bundle: dict,
    *,
    on_event=None,
    on_wake=None,
) -> StreamingPipeline:
    """Share the CLI's audio and calibrated wake policy with the dashboard."""
    sample_rate = int(wake_bundle["audio"]["sample_rate"])
    if sample_rate != int(vcm_bundle["audio"]["sample_rate"]):
        raise ValueError("Wake and command models must use the same microphone sample rate")
    wake_samples = round(
        sample_rate * float(wake_bundle["audio"]["duration_seconds"])
    )
    maximum_command_samples = round(
        sample_rate * float(vcm_bundle["audio"]["duration_seconds"])
    )
    stream_config = StreamingConfig(
        sample_rate=sample_rate,
        chunk_samples=round(sample_rate * args.stream_chunk_ms / 1000.0),
        wake_window_samples=wake_samples,
        wake_hop_samples=round(sample_rate * args.wake_hop_ms / 1000.0),
        pre_roll_samples=round(sample_rate * args.pre_roll_ms / 1000.0),
        onset_timeout_samples=round(sample_rate * args.onset_timeout_ms / 1000.0),
        minimum_command_samples=round(sample_rate * args.minimum_speech_ms / 1000.0),
        endpoint_silence_samples=round(
            sample_rate * args.endpoint_silence_ms / 1000.0
        ),
        maximum_command_samples=maximum_command_samples,
        cooldown_samples=round(sample_rate * args.cooldown_ms / 1000.0),
        speech_rms_threshold=args.speech_rms_threshold,
    )
    wake_streaming = wake_bundle.get("streaming_policy", {})
    wake_event_detector = WakeEventDetector(
        WakeSmoothingPolicy(
            threshold=float(wake_bundle.get("threshold", 0.7)),
            smoothing_window=int(wake_streaming.get("smoothing_window", 1)),
            required_hits=int(wake_streaming.get("required_hits", 1)),
        )
    )

    def wake_predictor(window: np.ndarray) -> bool:
        detected, label, score, elapsed = _wake_result(
            wake_model, wake_bundle, window
        )
        LOGGER.info(
            "Wake stream class=%s confidence=%.4f detected=%s inference_ms=%.2f",
            label,
            score,
            detected,
            elapsed,
        )
        smoothed = wake_event_detector.update(label, score)
        if on_wake is not None:
            on_wake(label, score, elapsed, smoothed)
        return smoothed

    command_audio_config = AudioConfig(**vcm_bundle["audio"])

    def command_predictor(captured: np.ndarray):
        waveform, usable_samples = prepare_audio_sequence(
            captured, sample_rate, command_audio_config
        )
        return predict_command(
            vcm_model,
            vcm_bundle,
            waveform,
            sample_length=usable_samples,
        )

    def publish(event: PipelineEvent) -> None:
        if event.message in {"stream_recovered", "stopped", "cooldown_complete"}:
            wake_event_detector.reset()
        if on_event is not None:
            on_event(event)

    return StreamingPipeline(
        stream_config, wake_predictor, command_predictor, on_event=publish
    )


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
    parser.add_argument("--stream-chunk-ms", type=float, default=100.0)
    parser.add_argument("--wake-hop-ms", type=float, default=250.0)
    parser.add_argument("--pre-roll-ms", type=float, default=250.0)
    parser.add_argument("--onset-timeout-ms", type=float, default=1_000.0)
    parser.add_argument("--minimum-speech-ms", type=float, default=250.0)
    parser.add_argument("--endpoint-silence-ms", type=float, default=500.0)
    parser.add_argument("--cooldown-ms", type=float, default=1_000.0)
    parser.add_argument("--speech-rms-threshold", type=float, default=0.008)
    parser.add_argument(
        "--allow-unverified-wake-model",
        action="store_true",
        help="Allow live use of an artifact explicitly marked deployment_ready=false",
    )
    parser.add_argument(
        "--wake-model",
        type=Path,
        default=ROOT / "wake-word/models/wake_word_model.me2",
    )
    parser.add_argument(
        "--vcm-model",
        type=Path,
        default=ROOT / "vcm/models/vcm_ctc_model.me2",
    )
    parser.add_argument("--log-file", type=Path, default=ROOT / "logs/app.log")
    return parser


def _print_vcm_result(result) -> None:
    if result.accepted:
        print(f"[ACCEPTED] Intent: {result.actionable_intent}")
        print(f"[VCM] Decoded text: {result.decoded_text}")
        if result.slots:
            print(f"[VCM] Supported slots: {dict(result.slots)}")
    else:
        print(f"[REJECTED] Reason: {result.rejection_reason}")
        if result.decoded_text:
            print(f"[VCM] Diagnostic candidate: {result.decoded_text}")
    print(
        "[VCM] Decoder score: "
        f"{result.normalized_log_score:.4f}; margin: {result.margin:.4f}"
    )


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
