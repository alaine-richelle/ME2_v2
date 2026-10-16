"""User-invoked bounded local output diagnostic; it never opens a microphone."""

from __future__ import annotations

import argparse
import sys

import numpy as np


def build_tone(seconds: float, sample_rate: int = 44_100) -> np.ndarray:
    """Build a quiet bounded mono tone entirely in memory."""

    if not 0.1 <= seconds <= 10.0:
        raise ValueError("seconds must be between 0.1 and 10")
    samples = max(1, round(seconds * sample_rate))
    timeline = np.arange(samples, dtype=np.float32) / sample_rate
    envelope_samples = min(round(sample_rate * 0.03), samples // 2)
    envelope = np.ones(samples, dtype=np.float32)
    if envelope_samples:
        ramp = np.linspace(0.0, 1.0, envelope_samples, dtype=np.float32)
        envelope[:envelope_samples] = ramp
        envelope[-envelope_samples:] = ramp[::-1]
    return (0.12 * envelope * np.sin(2 * np.pi * 440.0 * timeline)).astype(np.float32)


def play_output_check(
    *,
    seconds: float,
    output_device: str | int | None = None,
    sample_rate: int = 44_100,
) -> None:
    """Play one short tone through output only; failures do not fall back silently."""

    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError("sounddevice_not_installed") from error
    try:
        sd.check_output_settings(
            device=output_device,
            channels=1,
            dtype="float32",
            samplerate=sample_rate,
        )
        sd.play(build_tone(seconds, sample_rate), samplerate=sample_rate, device=output_device)
        sd.wait()
    except Exception as error:  # native output errors are normalized at the CLI boundary.
        raise RuntimeError("output_device_unavailable") from error
    finally:
        sd.stop()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--output-device", help="Optional output device name or index")
    parser.add_argument(
        "--speaker-approved",
        action="store_true",
        help="Required acknowledgement before the local output-only tone can play",
    )
    args = parser.parse_args(argv)
    if not args.speaker_approved:
        parser.error("--speaker-approved is required before playing output")
    output_device: str | int | None = args.output_device
    if output_device is not None and output_device.isdecimal():
        output_device = int(output_device)
    try:
        play_output_check(seconds=args.seconds, output_device=output_device)
    except (RuntimeError, ValueError) as error:
        print(f"Output diagnostic failed ({error}). Select the earphones in macOS Control Center and retry.", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"Output diagnostic completed: {args.seconds:.1f} seconds. No microphone was opened.")


if __name__ == "__main__":
    main()
