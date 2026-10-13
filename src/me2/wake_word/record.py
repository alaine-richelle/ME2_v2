"""Guided, collision-safe real-speaker recording for wake dataset v2."""

from __future__ import annotations

import argparse
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from me2.audio import write_wav
from me2.wake_word.data import (
    PRONUNCIATION_GUIDE,
    build_manifest_record,
    read_manifest,
    unique_recording_path,
    write_manifest,
)

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
PROMPTS = (
    ("a", "Say only the first syllable shown as “a”."),
    ("te", "Say only the second syllable: “teh”, with an eh-like vowel."),
    ("ko", "Say only the third syllable shown as “ko”."),
    (
        "ateko",
        (
            "Say the complete wake word naturally: “a-teh-ko”. "
            "The hyphens explain the syllables; do not add artificial pauses."
        ),
    ),
    ("ateko", "Say “a-teh-ko” naturally at a slightly faster rate."),
    ("ateko", "Say “a-teh-ko” naturally at a slightly slower rate."),
    ("background", "Say only “a-teh”; stop before ko."),
    ("background", "Say only “teh-ko”; omit the first syllable."),
    ("background", "Say the syllables in the wrong order: “ko-teh-a”."),
    ("background", "Say a normal voice command without the wake word."),
    ("background", "Remain silent and capture the room for two seconds."),
)


def _session_default() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--speaker",
        required=True,
        help="Stable pseudonymous speaker identifier; do not use a full name",
    )
    parser.add_argument("--session", default=_session_default())
    parser.add_argument("--room", default="unspecified")
    parser.add_argument("--device", help="Input device name or numeric index")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "wake-word/data/v3/recordings",
    )
    parser.add_argument("--take-count", type=int, default=22)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    args = parser.parse_args()
    if args.take_count <= 0:
        parser.error("--take-count must be positive")
    if args.sample_rate <= 0:
        parser.error("--sample-rate must be positive")
    try:
        import sounddevice as sd
    except ImportError as error:
        raise SystemExit(
            "Recording requires the microphone extra: pip install -e '.[microphone]'"
        ) from error
    input_device: str | int | None = args.device
    if isinstance(input_device, str) and input_device.isdecimal():
        input_device = int(input_device)
    try:
        device_info = (
            sd.query_devices(kind="input")
            if input_device is None
            else sd.query_devices(input_device, kind="input")
        )
    except Exception as error:
        raise SystemExit(f"Unable to inspect input device: {error}") from error

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.csv"
    rows = read_manifest(manifest) if manifest.exists() else []
    print(
        f"Pronunciation guide: {PRONUNCIATION_GUIDE}. "
        "New takes remain pending until explicitly reviewed."
    )
    for take in range(args.take_count):
        label, prompt = PROMPTS[take % len(PROMPTS)]
        input(
            f"[{take + 1}/{args.take_count}] {prompt} "
            "Press Enter to record; Ctrl-C to stop."
        )
        duration = 2.5 if label == "background" else 2.2
        try:
            recording = sd.rec(
                int(duration * args.sample_rate),
                samplerate=args.sample_rate,
                channels=1,
                dtype="float32",
                device=input_device,
            )
            sd.wait()
        except Exception as error:
            raise SystemExit(f"Microphone recording failed: {error}") from error
        recording_id = uuid.uuid4().hex
        destination = unique_recording_path(
            args.output,
            args.speaker,
            args.session,
            recording_id,
        )
        write_wav(destination, recording[:, 0], args.sample_rate)
        rows.append(
            build_manifest_record(
                destination,
                recording_id=recording_id,
                label=label,
                phrase=PRONUNCIATION_GUIDE if label == "ateko" else prompt,
                synthesis_text="",
                speaker_id=args.speaker,
                session_id=args.session,
                split="unassigned",
                source="real_recording",
                is_synthetic=False,
                kind=(
                    "complete_word"
                    if label == "ateko"
                    else "recorded_negative"
                    if label == "background"
                    else "syllable_cue"
                ),
                review_status="pending",
                review_notes="Awaiting explicit pronunciation and quality review.",
                device=str(device_info.get("name", args.device or "system_default")),
                room=args.room,
            )
        )
        write_manifest(manifest, rows)
        LOGGER.info("Saved %s label=%s review=pending", destination, label)


if __name__ == "__main__":
    main()
