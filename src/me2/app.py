"""Combined wake-word -> voice-command command-line application."""

from __future__ import annotations

import argparse
import logging
import math
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from me2.audio import (
    AudioConfig,
    load_wav,
    load_wav_sequence,
    prepare_audio,
    prepare_audio_sequence,
)
from me2.audio_devices import AudioDeviceError, select_capture_input, validate_alsa_device
from me2.bundle import load_bundle, predict_command, predict_waveform
from me2.io_utils import file_sha256
from me2.logging_utils import config_hash, configure_logging, log_event
from me2.runtime_config import RuntimeSettings, load_runtime_defaults
from me2.streaming import (
    PipelineEvent,
    PipelineState,
    StreamingConfig,
    StreamingPipeline,
    run_microphone_stream,
)
from me2.vcm.ctc import DECODER_RUNTIME_VERSION
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
    # Used for diagnostics only, never written into the retained artifact.
    bundle["runtime_artifact_sha256"] = file_sha256(path)
    log_event(
        LOGGER, "model_ready", task=expected_task,
        architecture=bundle.get("architecture", "TinyAudioCNN"),
        model_version=bundle.get("model_version", "unknown"),
        grammar_version=bundle.get("grammar_version", "not_applicable"),
        artifact_sha256=bundle["runtime_artifact_sha256"],
        config_sha256=config_hash({k: v for k, v in bundle.items() if k != "state_dict"}),
        decoder_version=DECODER_RUNTIME_VERSION,
        parameters=sum(p.numel() for p in model.parameters()),
        artifact_bytes=path.stat().st_size,
        deployment_ready=bundle.get("deployment_ready"),
    )
    return model, bundle


def _require_live_wake_readiness(
    bundle: dict,
    *,
    allow_unverified: bool,
) -> None:
    if bundle.get("deployment_ready") is True:
        return
    reason = (
        "Wake artifact has deployment_ready=false or missing approval; pronunciation "
        "review and/or real-speaker validation are incomplete."
    )
    if not allow_unverified:
        raise ValueError(
            f"{reason} Use --allow-unverified-wake-model only for a controlled "
            "engineering replay or microphone experiment."
        )
    log_event(LOGGER, "unverified_wake_override", level=logging.WARNING)


def _wake_result(model, bundle: dict, waveform: np.ndarray) -> tuple[bool, str, float, float]:
    started = time.perf_counter()
    label, _, probabilities = predict_waveform(model, bundle, waveform)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    wake_probability = probabilities[bundle["labels"].index("ateko")]
    threshold = float(bundle.get("threshold", 0.7))
    detected = label == "ateko" and wake_probability >= threshold
    return detected, label, wake_probability, elapsed_ms


def _capture(
    duration_seconds: float,
    sample_rate: int,
    input_device: str | int | None = None,
    *,
    audio_backend: str = "sounddevice",
    alsa_device: str = "auto",
) -> np.ndarray:
    if audio_backend == "arecord":
        frames = round(duration_seconds * sample_rate)
        capture_seconds = max(1, math.ceil(duration_seconds))
        try:
            result = subprocess.run(
                ["arecord", "-q", "-D", alsa_device, "-t", "raw", "-f", "S16_LE",
                 "-r", str(sample_rate), "-c", "1", "-d", str(capture_seconds), "-"],
                capture_output=True, check=False, timeout=capture_seconds + 5,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError("Microphone capture failed: capture_device_unavailable") from error
        if result.returncode or len(result.stdout) < frames * 2:
            raise RuntimeError("Microphone capture failed: capture_device_unavailable")
        return np.frombuffer(result.stdout[:frames * 2], dtype="<i2").astype(np.float32) / 32768.0
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
    finally:
        sd.stop()
    return recording[:, 0]


def _select_capture_for_args(args: argparse.Namespace, sample_rate: int) -> None:
    selection = select_capture_input(
        args.audio_backend,
        input_device=args.input_device,
        alsa_device=args.alsa_device,
        sample_rate=sample_rate,
    )
    args.audio_backend = selection.backend
    args.input_device = selection.input_device
    args.alsa_device = selection.alsa_device or args.alsa_device
    if args.audio_backend == "arecord":
        validate_alsa_device("capture", args.alsa_device)
    LOGGER.info("Selected microphone backend=%s source=%s", selection.backend, selection.selection_source)


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


def _log_command(result, bundle: dict, *, utterance_id: str | None = None, total_ms=None):
    log_event(
        LOGGER, "command_result", utterance_id=utterance_id or uuid.uuid4().hex,
        accepted=result.accepted, reason=result.rejection_reason,
        accepted_intent=result.actionable_intent,
        system_error=result.system_error, candidate_count=result.candidate_count,
        score=result.normalized_log_score, stage_timings_ms=dict(result.stage_timings_ms),
        diagnostics=dict(result.diagnostics),
        total_ms=total_ms, model_version=result.model_version,
        grammar_version=result.grammar_version or "not_applicable",
        decoder_version=DECODER_RUNTIME_VERSION,
        artifact_sha256=bundle.get("runtime_artifact_sha256", bundle.get("weights_sha256")),
    )


def _load_vcm_wav(bundle: dict, path: Path) -> tuple[np.ndarray, int | None]:
    audio_config = AudioConfig(**bundle["audio"])
    if bundle.get("architecture") in {"CharacterAudioCNN", "CompactCTCModel"}:
        return load_wav_sequence(path, audio_config)
    return load_wav(path, audio_config), None


def _prepare_vcm_audio(bundle: dict, samples, sample_rate):
    config = AudioConfig(**bundle["audio"])
    if bundle.get("architecture") in {"CharacterAudioCNN", "CompactCTCModel"}:
        return prepare_audio_sequence(samples, sample_rate, config)
    return prepare_audio(samples, sample_rate, config), None


def _run_wake(args: argparse.Namespace) -> None:
    model, bundle = _model(args.wake_model, "wake_word")
    if args.audio:
        waveform = load_wav(args.audio, AudioConfig(**bundle["audio"]))
        detected, label, score, elapsed = _wake_result(model, bundle, waveform)
        print(f"Wake class: {label}; wake confidence: {score:.3f}; inference: {elapsed:.1f} ms")
        print("Wake word detected!" if detected else "Wake word not detected.")
        log_event(LOGGER, "wake_result", utterance_id=uuid.uuid4().hex,
                  wake_detected=detected, score=score, total_ms=elapsed)
        return
    _require_live_wake_readiness(
        bundle,
        allow_unverified=args.allow_unverified_wake_model,
    )
    print("[LISTENING] Waiting for wake word...")
    duration = float(bundle["audio"]["duration_seconds"])
    rate = int(bundle["audio"]["sample_rate"])
    _select_capture_for_args(args, rate)
    while True:
        waveform = _capture(
            duration, rate, args.input_device,
            audio_backend=args.audio_backend, alsa_device=args.alsa_device,
        )
        waveform = prepare_audio(waveform, rate, AudioConfig(**bundle["audio"]))
        detected, label, score, elapsed = _wake_result(model, bundle, waveform)
        log_event(LOGGER, "wake_result", wake_detected=detected, score=score, total_ms=elapsed)
        if detected:
            print('[WAKE WORD] "ateko" detected')
            return


def _run_vcm(args: argparse.Namespace) -> None:
    model, bundle = _model(args.vcm_model, "vcm")
    sample_length = None
    if args.audio:
        waveform, sample_length = _load_vcm_wav(bundle, args.audio)
    else:
        _select_capture_for_args(args, int(bundle["audio"]["sample_rate"]))
        print("[LISTENING] Waiting for command...")
        waveform = _capture(
            float(bundle["audio"]["duration_seconds"]),
            int(bundle["audio"]["sample_rate"]),
            args.input_device,
            audio_backend=args.audio_backend,
            alsa_device=args.alsa_device,
        )
        waveform, sample_length = _prepare_vcm_audio(
            bundle, waveform, int(bundle["audio"]["sample_rate"])
        )
    result, elapsed = _predict_vcm(model, bundle, waveform, sample_length=sample_length)
    _print_vcm_result(result, debug_sensitive=args.debug_sensitive)
    _log_command(result, bundle, total_ms=elapsed)


def _run_interactive(args: argparse.Namespace) -> None:
    wake_model, wake_bundle = _model(args.wake_model, "wake_word")
    vcm_model, vcm_bundle = _model(args.vcm_model, "vcm")
    microphone = args.wake_audio is None
    if microphone and args.command_audio:
        raise ValueError("--command-audio requires --wake-audio for file-based pipeline testing")
    if microphone:
        if args.guided_terminal:
            raise ValueError("--guided-terminal requires approved --wake-audio and --command-audio fixtures")
        _require_live_wake_readiness(
            wake_bundle,
            allow_unverified=args.allow_unverified_wake_model,
        )
        _run_continuous_interactive(
            args,
            wake_model,
            wake_bundle,
            vcm_model,
            vcm_bundle,
        )
        return

    if args.combined_replay:
        if args.command_audio is None:
            raise ValueError("--combined-replay requires --command-audio")
        return _run_combined_replay(args, wake_model, wake_bundle, vcm_model, vcm_bundle)

    while True:
        print("[LISTENING] Waiting for wake word...")
        if args.wake_audio:
            if args.guided_terminal:
                return _run_guided_terminal(args, wake_model, wake_bundle, vcm_model, vcm_bundle)
            wake_waveform = load_wav(args.wake_audio, AudioConfig(**wake_bundle["audio"]))
        else:
            wake_waveform = _capture(
                float(wake_bundle["audio"]["duration_seconds"]),
                int(wake_bundle["audio"]["sample_rate"]),
                args.input_device,
            )
            wake_waveform = prepare_audio(
                wake_waveform, int(wake_bundle["audio"]["sample_rate"]),
                AudioConfig(**wake_bundle["audio"]),
            )
        detected, _, wake_score, wake_ms = _wake_result(wake_model, wake_bundle, wake_waveform)
        utterance_id = uuid.uuid4().hex
        log_event(LOGGER, "wake_result", utterance_id=utterance_id,
                  wake_detected=detected, score=wake_score, total_ms=wake_ms)
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
            command_waveform, command_sample_length = _prepare_vcm_audio(
                vcm_bundle, command_waveform, int(vcm_bundle["audio"]["sample_rate"])
            )
        print("[DECODING] Decoding captured command...")
        result, inference_ms = _predict_vcm(
            vcm_model,
            vcm_bundle,
            command_waveform,
            sample_length=command_sample_length,
        )
        _print_vcm_result(result, debug_sensitive=args.debug_sensitive)
        _log_command(result, vcm_bundle, utterance_id=utterance_id, total_ms=inference_ms)
        print("[COOLDOWN] Interaction complete.")
        print("[STATUS] Returning to wake-word detection...")
        log_event(LOGGER, "listening_resumed", utterance_id=utterance_id)
        if args.wake_audio:
            return


def _guided_pause(stage: str, prompt=input) -> None:
    """Test-only human checkpoint; never used by unattended microphone runtime."""
    response = prompt(f"[GUIDED] {stage}. Press Enter to continue: ")
    if response.strip().casefold() in {"q", "quit", "stop"}:
        raise KeyboardInterrupt


def _run_guided_terminal(args, wake_model, wake_bundle, vcm_model, vcm_bundle, *, prompt=input) -> None:
    """Replay approved fixtures with explicit operator-visible phase checkpoints."""
    if args.wake_audio is None or args.command_audio is None:
        raise ValueError("--guided-terminal requires both --wake-audio and --command-audio")
    for stage in ("start", "wake initialization", "waiting/read wake"):
        _guided_pause(stage, prompt)
    wake = load_wav(args.wake_audio, AudioConfig(**wake_bundle["audio"]))
    detected, _, _, _ = _wake_result(wake_model, wake_bundle, wake)
    if not detected:
        _guided_pause("end (wake rejected)", prompt)
        return None
    for stage in ("wake detected", "acknowledgment chime", "command initialization", "command read"):
        _guided_pause(stage, prompt)
    command, length = _load_vcm_wav(vcm_bundle, args.command_audio)
    result, _ = _predict_vcm(vcm_model, vcm_bundle, command, sample_length=length)
    _guided_pause("command detected" if result.accepted else "command rejected", prompt)
    if result.accepted:
        _guided_pause("execution", prompt)
        _guided_pause("confirmation", prompt)
    _guided_pause("end", prompt)
    return result


def replay_combined_wavs(pipeline: StreamingPipeline, wake: np.ndarray, command: np.ndarray, *, pause_samples: int) -> list[PipelineEvent]:
    """Feed approved wake/pause/command audio through the actual streaming state machine."""
    combined = np.concatenate([wake, np.zeros(pause_samples, dtype=np.float32), command])
    chunks = (combined[offset:offset + pipeline.config.chunk_samples]
              for offset in range(0, combined.size, pipeline.config.chunk_samples))
    events: list[PipelineEvent] = []
    for chunk in chunks:
        events.extend(pipeline.feed(chunk))
    return events


def _run_combined_replay(args, wake_model, wake_bundle, vcm_model, vcm_bundle):
    pipeline = build_streaming_pipeline(args, wake_model, wake_bundle, vcm_model, vcm_bundle)
    rate = int(wake_bundle["audio"]["sample_rate"])
    wake = load_wav(args.wake_audio, AudioConfig(**wake_bundle["audio"]))
    command, _ = _load_vcm_wav(vcm_bundle, args.command_audio)
    events = replay_combined_wavs(
        pipeline, wake, command, pause_samples=round(rate * args.combined_pause_ms / 1000),
    )
    result = next((event.result for event in events if event.result is not None), None)
    if result is None:
        raise RuntimeError("combined_replay_did_not_reach_command_result")
    _print_vcm_result(result, debug_sensitive=args.debug_sensitive)
    return result


def _run_continuous_interactive(
    args: argparse.Namespace,
    wake_model,
    wake_bundle: dict,
    vcm_model,
    vcm_bundle: dict,
) -> None:
    _select_capture_for_args(args, int(wake_bundle["audio"]["sample_rate"]))
    def on_wake(label, score, elapsed, detected, status) -> None:
        del label, score, elapsed, detected
        if status["mode"] == "both_failed":
            print("[DEGRADED_WAKE] v5 wake evaluation failed; command capture is blocked.")

    def on_event(event: PipelineEvent) -> None:
        if event.state == PipelineState.CAPTURING_COMMAND:
            print('[WAKE WORD] "ateko" detected')
            print("[CAPTURING_COMMAND] Listening for command...")
        elif event.state == PipelineState.DECODING:
            print("[DECODING] Endpoint detected.")
        elif event.state == PipelineState.ACCEPTED_OR_REJECTED and event.result:
            _print_vcm_result(event.result, debug_sensitive=args.debug_sensitive)
        elif event.state == PipelineState.COOLDOWN:
            print("[COOLDOWN] Suppressing duplicate triggers.")
        elif event.state == PipelineState.LISTENING_FOR_WAKE:
            print("[LISTENING_FOR_WAKE] Ready.")

    pipeline = build_streaming_pipeline(
        args,
        wake_model,
        wake_bundle,
        vcm_model,
        vcm_bundle,
        on_event=on_event,
        on_wake=on_wake,
    )
    print("[LISTENING_FOR_WAKE] Waiting for ateko...")
    stop_event = threading.Event()
    timer = (
        threading.Timer(args.session_seconds, stop_event.set)
        if args.session_seconds else None
    )
    if timer is not None:
        timer.daemon = True
        timer.start()
    try:
        run_microphone_stream(
            pipeline,
            input_device=args.input_device,
            audio_backend=args.audio_backend,
            alsa_device=args.alsa_device,
            stop_event=stop_event,
        )
    finally:
        if timer is not None:
            timer.cancel()


def build_streaming_pipeline(
    args: argparse.Namespace,
    wake_model,
    wake_bundle: dict | None,
    vcm_model,
    vcm_bundle: dict,
    *,
    on_event=None,
    on_wake=None,
) -> StreamingPipeline:
    """Share the CLI's audio and calibrated wake policy with the dashboard."""
    if wake_bundle is None:
        raise RuntimeError("wake_model_unavailable")
    active_wake_bundle = wake_bundle
    sample_rate = int(active_wake_bundle["audio"]["sample_rate"])
    if sample_rate != int(vcm_bundle["audio"]["sample_rate"]):
        raise ValueError("Wake and command models must use the same microphone sample rate")
    wake_samples = round(
        sample_rate * float(active_wake_bundle["audio"]["duration_seconds"])
    )
    maximum_command_samples = round(
        sample_rate * float(vcm_bundle["audio"]["duration_seconds"])
    )
    configured_maximum = getattr(args, "maximum_command_ms", 0.0)
    if configured_maximum:
        requested = round(sample_rate * configured_maximum / 1000)
        if requested > maximum_command_samples:
            raise ValueError("maximum_command_ms exceeds the VCM artifact duration; use a separately validated artifact")
        maximum_command_samples = requested
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
        noise_floor_calibration_samples=round(
            sample_rate * args.noise_floor_calibration_ms / 1000.0
        ),
        noise_floor_multiplier=args.noise_floor_multiplier,
        speech_hysteresis_ratio=args.speech_hysteresis_ratio,
        clipping_warning_fraction=args.clipping_warning_fraction,
        clipping_reject_fraction=args.clipping_reject_fraction,
        post_wake_guard_samples=round(sample_rate * args.post_wake_guard_ms / 1000.0),
        queue_chunks=args.queue_chunks,
    )
    streaming = wake_bundle.get("streaming_policy", {})
    primary_detector = WakeEventDetector(WakeSmoothingPolicy(
        threshold=float(wake_bundle.get("threshold", 0.7)),
        smoothing_window=int(streaming.get("smoothing_window", 1)),
        required_hits=int(streaming.get("required_hits", 1)),
    ))
    wake_status = {
        "primary_role": "v5_primary",
        "fallback_role": "not_supported",
        "mode": "primary",
        "primary_status": "ready",
        "primary_failure": None,
        "fallback_status": "not_supported",
        "fallback_failure": None,
    }
    # Compatibility variables keep the existing diagnostic shape stable while
    # making every fallback branch unreachable in the v5-only runtime.
    fallback_model = fallback_bundle = fallback_detector = primary_load_error = None

    def wake_predictor(window: np.ndarray) -> bool:
        pipeline_started = time.perf_counter()
        frontend_started = pipeline_started
        diagnostics = {
            "wake_primary_role": "v5_primary",
            "wake_fallback_role": wake_status["fallback_role"],
            "wake_primary_status": wake_status["primary_status"],
            "wake_fallback_status": wake_status["fallback_status"],
            "wake_mode": wake_status["mode"],
            "wake_primary_failure": wake_status["primary_failure"],
            "wake_fallback_failure": wake_status["fallback_failure"],
            "wake_primary_artifact_sha256": (
                wake_bundle.get("runtime_artifact_sha256") if wake_bundle else None
            ),
            "wake_fallback_artifact_sha256": (
                fallback_bundle.get("runtime_artifact_sha256") if fallback_bundle else None
            ),
        }
        try:
            prepared = prepare_audio(
                window, sample_rate, AudioConfig(**active_wake_bundle["audio"])
            )
        except Exception as error:  # noqa: BLE001 -- no valid wake result may escape a frontend fault.
            wake_status.update(
                mode="both_failed",
                primary_status="system_failed",
                fallback_status="system_failed" if fallback_bundle is not None else "not_configured",
                primary_failure=type(error).__name__,
                fallback_failure=type(error).__name__ if fallback_bundle is not None else None,
            )
            diagnostics.update(
                wake_mode=wake_status["mode"],
                wake_primary_status=wake_status["primary_status"],
                wake_fallback_status=wake_status["fallback_status"],
                wake_primary_failure=wake_status["primary_failure"],
                wake_fallback_failure=wake_status["fallback_failure"],
            )
            log_event(LOGGER, "wake_window", level=logging.WARNING, wake_detected=False,
                      total_ms=(time.perf_counter() - pipeline_started) * 1000.0,
                      diagnostics=diagnostics)
            return False
        frontend_ms = (time.perf_counter() - frontend_started) * 1000.0
        primary_label = "unavailable"
        primary_score = None
        primary_elapsed = None
        primary_smoothed = False
        primary_fault = primary_load_error is not None
        if wake_model is not None and wake_bundle is not None:
            try:
                _, primary_label, primary_score, primary_elapsed = _wake_result(
                    wake_model, wake_bundle, prepared
                )
                primary_smoothed = bool(primary_detector and primary_detector.update(
                    primary_label, primary_score
                ))
                if primary_detector is not None:
                    diagnostics.update(primary_detector.diagnostics())
                wake_status.update(mode="primary", primary_status="ready", primary_failure=None)
            except Exception as error:  # noqa: BLE001 -- fault-only fallback is intentional.
                primary_fault = True
                wake_status.update(
                    mode="fallback_active" if fallback_model is not None else "both_failed",
                    primary_status="inference_failed",
                    primary_failure=type(error).__name__,
                )

        fallback_label = "not_scored"
        fallback_score = None
        fallback_elapsed = None
        fallback_smoothed = False
        if fallback_model is not None and fallback_bundle is not None:
            try:
                _, fallback_label, fallback_score, fallback_elapsed = _wake_result(
                    fallback_model, fallback_bundle, prepared
                )
                fallback_smoothed = bool(fallback_detector and fallback_detector.update(
                    fallback_label, fallback_score
                ))
                wake_status.update(fallback_status="ready", fallback_failure=None)
            except Exception as error:  # noqa: BLE001 -- the primary result remains authoritative.
                wake_status.update(
                    fallback_status="inference_failed",
                    fallback_failure=type(error).__name__,
                )
                if primary_fault:
                    wake_status["mode"] = "both_failed"

        smoothed = fallback_smoothed if primary_fault else primary_smoothed
        if primary_fault and not fallback_smoothed:
            smoothed = False
        if not primary_fault and fallback_smoothed:
            diagnostics["wake_fallback_disagreement"] = True
        diagnostics.update(
            wake_mode=wake_status["mode"],
            wake_primary_status=wake_status["primary_status"],
            wake_fallback_status=wake_status["fallback_status"],
            wake_primary_failure=wake_status["primary_failure"],
            wake_fallback_failure=wake_status["fallback_failure"],
            wake_primary_label=primary_label,
            wake_primary_score=primary_score,
            wake_primary_model_ms=primary_elapsed,
            wake_primary_detected=primary_smoothed,
            wake_fallback_label=fallback_label,
            wake_fallback_score=fallback_score,
            wake_fallback_model_ms=fallback_elapsed,
            wake_fallback_detected=fallback_smoothed,
            wake_debounce_state="listening",
            wake_cooldown_remaining_ms=0.0,
        )
        pipeline_ms = (time.perf_counter() - pipeline_started) * 1000.0
        diagnostics.update(
            wake_class_is_ateko=primary_label == "ateko",
            frontend_ms=frontend_ms,
            wake_model_ms=primary_elapsed,
            wake_pipeline_ms=pipeline_ms,
        )
        log_event(
            LOGGER,
            "wake_window",
            level=logging.WARNING if primary_fault else logging.INFO,
            wake_detected=smoothed,
            score=primary_score if primary_score is not None else fallback_score,
            total_ms=pipeline_ms,
            diagnostics=diagnostics,
        )
        if on_wake is not None:
            label = fallback_label if primary_fault else primary_label
            score = fallback_score if primary_fault else primary_score
            elapsed = fallback_elapsed if primary_fault else primary_elapsed
            on_wake(label, score or 0.0, elapsed or 0.0, smoothed, dict(wake_status))
        return smoothed

    def command_predictor(captured: np.ndarray):
        waveform, usable_samples = _prepare_vcm_audio(vcm_bundle, captured, sample_rate)
        return predict_command(
            vcm_model,
            vcm_bundle,
            waveform,
            sample_length=usable_samples,
        )

    def publish(event: PipelineEvent) -> None:
        if event.result is not None:
            _log_command(event.result, vcm_bundle, utterance_id=event.utterance_id)
        if event.message in {"stream_recovered", "stopped", "cooldown_complete"}:
            if primary_detector is not None:
                primary_detector.reset()
            if fallback_detector is not None:
                fallback_detector.reset()
        if on_event is not None:
            on_event(event)

    pipeline = StreamingPipeline(
        stream_config, wake_predictor, command_predictor, on_event=publish
    )
    pipeline.wake_status = wake_status
    return pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--wake-word", action="store_true", help="Test/listen with the wake model only")
    modes.add_argument("--vcm", action="store_true", help="Test/listen with the command model only")
    modes.add_argument("--interactive", action="store_true", help="Run the full wake -> command loop")
    parser.add_argument("--audio", type=Path, help="WAV input for an individual component")
    parser.add_argument("--wake-audio", type=Path, help="Wake-word WAV for full-pipeline testing")
    parser.add_argument("--command-audio", type=Path, help="Command WAV after a successful wake gate")
    parser.add_argument("--guided-terminal", action="store_true",
                        help="Test-only fixture replay with explicit terminal phase confirmations")
    parser.add_argument("--combined-replay", action="store_true",
                        help="Run approved wake/pause/command WAVs through the streaming state machine")
    parser.add_argument("--combined-pause-ms", type=float, default=300.0)
    parser.add_argument("--input-device", help="Microphone device name or index (default: system input)")
    parser.add_argument(
        "--audio-backend",
        choices=("auto", "sounddevice", "arecord"),
        default=RuntimeSettings().audio_backend,
        help="Microphone backend; auto prefers shared system input then direct ALSA fallback",
    )
    parser.add_argument(
        "--alsa-device",
        default=RuntimeSettings().alsa_device,
        help=("ALSA PCM device for --audio-backend arecord "
              "(for example plughw:CARD=Device,DEV=0; avoid unstable numeric card IDs)"),
    )
    parser.add_argument("--config", type=Path, help="Versioned runtime JSON; explicit CLI flags win")
    parser.add_argument("--threads", type=int, default=RuntimeSettings().threads)
    parser.add_argument("--queue-chunks", type=int, default=RuntimeSettings().queue_chunks)
    parser.add_argument("--debug-sensitive", action="store_true",
                        help="Opt in to transcript/path diagnostics; do not share these logs")
    parser.add_argument("--stream-chunk-ms", type=float, default=100.0)
    parser.add_argument("--wake-hop-ms", type=float, default=250.0)
    parser.add_argument("--pre-roll-ms", type=float, default=250.0)
    parser.add_argument("--onset-timeout-ms", type=float, default=1_000.0)
    parser.add_argument("--minimum-speech-ms", type=float, default=250.0)
    parser.add_argument("--endpoint-silence-ms", type=float, default=500.0)
    parser.add_argument("--maximum-command-ms", type=float, default=0.0,
                        help="Speech-span bound; 0 inherits artifact duration; cannot exceed it")
    parser.add_argument("--session-seconds", type=float, default=0.0,
                        help="Stop a continuous microphone session after this many seconds; 0 is indefinite")
    parser.add_argument("--cooldown-ms", type=float, default=1_000.0)
    parser.add_argument("--speech-rms-threshold", type=float, default=0.008)
    parser.add_argument("--noise-floor-calibration-ms", type=float, default=1_000.0)
    parser.add_argument("--noise-floor-multiplier", type=float, default=2.5)
    parser.add_argument("--speech-hysteresis-ratio", type=float, default=0.7)
    parser.add_argument("--clipping-warning-fraction", type=float, default=0.02)
    parser.add_argument("--clipping-reject-fraction", type=float, default=0.10)
    parser.add_argument("--post-wake-guard-ms", type=float, default=150.0,
                        help="Discard only post-wake samples while local feedback tone finishes")
    parser.add_argument(
        "--allow-unverified-wake-model",
        action="store_true",
        help="Allow live use of an artifact explicitly marked deployment_ready=false",
    )
    parser.add_argument(
        "--wake-model",
        type=Path,
        default=ROOT / "wake-word/models/wake_word_v5_regularized_seed43.me2",
    )
    parser.add_argument(
        "--vcm-model",
        type=Path,
        default=ROOT / "vcm/models/vcm_ctc_model.me2",
    )
    parser.add_argument("--log-file", type=Path, default=ROOT / "logs/app.log")
    parser.set_defaults(**asdict(RuntimeSettings()))
    return parser


def _print_vcm_result(result, *, debug_sensitive=False) -> None:
    if result.accepted:
        print(f"[ACCEPTED] Intent: {result.actionable_intent}")
        if debug_sensitive:
            print(f"[VCM] Decoded text: {result.decoded_text}")
        if result.slots and debug_sensitive:
            print(f"[VCM] Supported slots: {dict(result.slots)}")
    else:
        print(f"[REJECTED] Reason: {result.rejection_reason}")
        details = result.diagnostics
        if result.rejection_reason == "command_onset_timeout":
            print(f"[CAPTURE] Onset limit {details.get('onset_timeout_ms')} ms; "
                  f"speech observed: {details.get('speech_observed', False)}.")
        elif result.rejection_reason == "command_too_long":
            print(f"[CAPTURE] Command duration exceeded {details.get('maximum_command_ms')} ms; "
                  "audio discarded without decoding or cropping.")
        elif result.rejection_reason == "command_clipping_excessive":
            print("[CAPTURE] Excessive microphone clipping; reduce input gain or increase distance.")
        if details:
            print(f"[DIAGNOSTIC] {details.get('diagnostic_category', 'unknown')}; "
                  f"valid features/output: {details.get('valid_feature_frames')}/"
                  f"{details.get('valid_output_frames')}; "
                  f"surviving prefixes/terminals: {details.get('surviving_prefix_count')}/"
                  f"{details.get('terminal_count')}; "
                  f"next: {details.get('next_action', 'inspect_structured_log')}")
        if result.decoded_text and debug_sensitive:
            print(f"[VCM] Diagnostic candidate: {result.decoded_text}")
    print(
        "[VCM] Decoder score: "
        f"{result.normalized_log_score:.4f}; margin: {result.margin:.4f}"
    )


def parse_args(argv=None):
    parser = build_parser()
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--config", type=Path)
    config_args, _ = early.parse_known_args(argv)
    if config_args.config is not None:
        try:
            parser.set_defaults(**load_runtime_defaults(config_args.config))
        except (OSError, ValueError, TypeError):
            parser.error("Invalid runtime configuration; check schema, types and file permissions")
    args = parser.parse_args(argv)
    try:
        RuntimeSettings.from_namespace(args)
        from me2.contracts import finite

        finite("session_seconds", args.session_seconds, 0.0, 3600.0)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    if args.audio and args.interactive:
        parser.error("Use --wake-audio and optionally --command-audio for combined WAV testing")
    if args.wake_audio and not args.interactive:
        parser.error("--wake-audio is only valid with --interactive")
    if args.command_audio and not args.interactive:
        parser.error("--command-audio is only valid with --interactive")
    if args.input_device and args.input_device.isdecimal():
        args.input_device = int(args.input_device)
    return args


def main() -> None:
    args = parse_args()
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    logging_ready = False
    def terminate(signum, frame):
        raise KeyboardInterrupt

    try:
        configure_logging(
            args.log_file, logging.DEBUG if args.debug_sensitive else logging.INFO,
            debug_sensitive=args.debug_sensitive,
            context={"config_sha256": config_hash(vars(args)),
                     "decoder_version": DECODER_RUNTIME_VERSION, "threads": args.threads},
        )
        logging_ready = True
        signal.signal(signal.SIGTERM, terminate)
        torch.set_num_threads(args.threads)
        if args.wake_word:
            _run_wake(args)
        elif args.vcm:
            _run_vcm(args)
        else:
            _run_interactive(args)
    except KeyboardInterrupt:
        print("\n[STATUS] Stopped.")
    except Exception as error:
        if logging_ready:
            LOGGER.exception("Inference failed")
        reason = str(error) if isinstance(error, AudioDeviceError) else type(error).__name__
        print(f"Inference failed ({reason}); inspect safe logs and run me2-health.",
              file=sys.stderr)
        raise SystemExit(1) from error
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        if logging_ready:
            log_event(LOGGER, "run_stopped")
            logging.shutdown()


if __name__ == "__main__":
    main()
