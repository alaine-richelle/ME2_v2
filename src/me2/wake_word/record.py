"""Guided WAV recording helper for replacing synthetic wake data with real voices."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import sounddevice as sd

from me2.audio import write_wav
from me2.wake_word.data import FIELDS

LOGGER = logging.getLogger(__name__)
PROMPTS = (
    ("a", "Say only the first syllable: a"),
    ("te", "Say only the second syllable: te"),
    ("ko", "Say only the third syllable: ko"),
    ("ateko", "Say the complete wake word: a-te-ko"),
    ("ateko", "Say the complete wake word faster: a-te-ko"),
    ("background", "Say a near-miss phrase such as 'take a coat'"),
    ("background", "Stay silent for two seconds"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker", required=True, help="Stable pseudonymous speaker identifier")
    parser.add_argument("--output", type=Path, default=Path("wake-word/data/recordings"))
    parser.add_argument("--take-count", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.csv"
    exists = manifest.exists()
    with manifest.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        for take in range(args.take_count):
            label, prompt = PROMPTS[take % len(PROMPTS)]
            input(f"[{take + 1}/{args.take_count}] {prompt} Press Enter to record; Ctrl-C to stop.")
            duration = 2.5 if label == "background" else 1.8
            recording = sd.rec(
                int(duration * args.sample_rate),
                samplerate=args.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            filename = f"{args.speaker}_{take:04d}.wav"
            destination = args.output / filename
            write_wav(destination, recording[:, 0], args.sample_rate)
            writer.writerow({
                "path": str(destination.resolve()),
                "label": label,
                "phrase": prompt,
                "speaker_id": args.speaker,
                "split": "unassigned",
                "source": "real_recording",
                "kind": "recorded",
            })
            stream.flush()
            LOGGER.info("Saved %s label=%s", destination, label)
