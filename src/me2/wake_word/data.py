"""Generate a labeled wake-word bootstrap corpus and manage recording manifests."""

from __future__ import annotations

import argparse
import csv
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from me2.audio import write_wav

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
CLASSES = ("background", "a", "te", "ko", "ateko")
TTS_VOICE_CANDIDATES = (
    "Samantha", "Daniel", "Karen", "Moira", "Tessa", "Albert", "Aman", "Kathy",
    "Rishi", "Tara", "Fred", "Ralph",
)
TARGET_PHRASES = (("a", "a"), ("te", "te"), ("ko", "ko"), ("a te ko", "ateko"), ("ateko", "ateko"))
HARD_NEGATIVES = ("a te", "a ticket", "take a coat", "tea and cocoa", "hello there", "turn on the light")
FIELDS = ("path", "label", "phrase", "speaker_id", "split", "source", "kind")


def _voice_names() -> list[str]:
    result = subprocess.run(["say", "-v", "?"], check=True, capture_output=True, text=True)
    names: list[str] = []
    for line in result.stdout.splitlines():
        match = re.match(r"^(.+?)\s+([a-z]{2}_[A-Z]{2})\s+#", line)
        if match and match.group(2).startswith("en_"):
            names.append(match.group(1).strip())
    selected = [name for name in TTS_VOICE_CANDIDATES if name in names]
    return selected[:12]


def _voice_split(voice: str, index: int, count: int) -> str:
    train_end = max(1, int(count * 0.65))
    validation_end = max(train_end + 1, int(count * 0.82))
    if index < train_end:
        return "train"
    if index < validation_end:
        return "validation"
    return "test"


def generate_synthetic_dataset(output_dir: Path, seed: int = 42) -> int:
    if subprocess.run(["/usr/bin/which", "say"], capture_output=True, check=False).returncode != 0:
        raise RuntimeError("macOS 'say' is required for bootstrap audio; use the recording tool elsewhere")
    voices = _voice_names()
    if len(voices) < 3:
        raise RuntimeError("At least three distinct English macOS voices are required for voice-disjoint splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    rates = (125, 145, 165, 185, 205, 225)
    phrases = list(TARGET_PHRASES) + [(phrase, "background") for phrase in HARD_NEGATIVES]
    for voice_index, voice in enumerate(voices):
        for phrase, label in phrases:
            for rate in rates:
                safe = re.sub(r"[^a-z0-9]+", "_", phrase.casefold()).strip("_")
                filename = f"{voice_index:02d}_{safe}_{rate}.wav"
                path = output_dir / filename
                try:
                    subprocess.run(
                        ["say", "-v", voice, "-r", str(rate), "-o", str(path),
                         "--file-format=WAVE", "--data-format=LEF32@16000", phrase],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except (OSError, subprocess.CalledProcessError) as error:
                    LOGGER.warning("Speech synthesis failed voice=%s phrase=%r: %s", voice, phrase, error)
                    continue
                records.append({
                    "path": str(path.resolve()),
                    "label": label,
                    "phrase": phrase,
                    "speaker_id": voice,
                    "split": _voice_split(voice, voice_index, len(voices)),
                    "source": "macos_say_synthetic",
                    "kind": "complete_word" if label == "ateko" else ("hard_negative" if label == "background" else "syllable_cue"),
                })

        for background_index in range(12):
            samples = rng.normal(0.0, rng.uniform(0.0001, 0.001), 24_000).astype(np.float32)
            path = output_dir / f"{voice_index:02d}_noise_{background_index:02d}.wav"
            write_wav(path, samples, 16_000)
            records.append({
                "path": str(path.resolve()),
                "label": "background",
                "phrase": "",
                "speaker_id": voice,
                "split": _voice_split(voice, voice_index, len(voices)),
                "source": "generated_noise",
                "kind": "noise_only",
            })

    if not records:
        raise RuntimeError("No wake-word audio was generated")
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    LOGGER.info("Generated %d wake examples from %d synthetic voices at %s", len(records), len(voices), manifest)
    return len(records)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row.get("label") not in CLASSES:
            raise ValueError(f"Unsupported wake label {row.get('label')!r}; expected one of {CLASSES}")
        if not Path(row["path"]).is_file():
            LOGGER.warning("Missing/corrupted wake audio path=%s", row["path"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="Generate synthetic bootstrap audio")
    parser.add_argument("--output", type=Path, default=ROOT / "wake-word/data/generated")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.generate:
        parser.error("Use --generate to synthesize the bootstrap corpus")
    generate_synthetic_dataset(args.output, args.seed)


if __name__ == "__main__":
    main()
