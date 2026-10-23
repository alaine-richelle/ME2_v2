#!/usr/bin/env bash
# Safe Raspberry Pi ALSA capture discovery/probe helper.
#
# With no arguments this only enumerates ALSA devices and their current owners.
# Recording requires an explicit approval environment variable, has a bounded
# duration, writes only beneath results/, and prints technical signal metrics
# rather than speech content. Never add its generated WAV files to Git.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

usage() {
  cat <<'EOF'
Usage:
  scripts/pi5_audio_capture_check.sh [--inspect]
  scripts/pi5_audio_capture_check.sh --probe --device ALSA_PCM
  scripts/pi5_audio_capture_check.sh --validate --input results/name.wav
  ME2_AUDIO_CAPTURE_APPROVED=1 scripts/pi5_audio_capture_check.sh \
    --record --device ALSA_PCM [--seconds 8] [--output results/name.wav]

Examples:
  scripts/pi5_audio_capture_check.sh --inspect
  scripts/pi5_audio_capture_check.sh --probe --device 'plughw:CARD=Device,DEV=0'
  scripts/pi5_audio_capture_check.sh --validate --input results/microphone-check.wav
  ME2_AUDIO_CAPTURE_APPROVED=1 scripts/pi5_audio_capture_check.sh \
    --record --device 'plughw:CARD=Device,DEV=0'

Choose ALSA_PCM from `arecord -L`; do not assume a numeric card/device pair.
`--probe` opens the selected PCM only to inspect supported parameters and does
not save audio. `--record` is a separately approved, bounded validation only.
EOF
}

mode="inspect"
device=""
input=""
seconds="8"
run_id="${ME2_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
output="results/microphone-check-${run_id}.wav"

while (($#)); do
  case "$1" in
    --inspect) mode="inspect" ;;
    --probe) mode="probe" ;;
    --validate) mode="validate" ;;
    --record) mode="record" ;;
    --device)
      device="${2:?--device requires an ALSA PCM name}"
      shift
      ;;
    --input)
      input="${2:?--input requires a WAV path under results/}"
      shift
      ;;
    --seconds)
      seconds="${2:?--seconds requires a positive integer}"
      shift
      ;;
    --output)
      output="${2:?--output requires a path under results/}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

case "$seconds" in
  ''|*[!0-9]*)
    echo "--seconds must be a positive integer" >&2
    exit 2
    ;;
esac
if ((seconds < 1 || seconds > 60)); then
  echo "--seconds must be between 1 and 60" >&2
  exit 2
fi

validate_wav() {
  local wav_path="$1"
  if [[ ! -x .venv/bin/python ]]; then
    echo "Expected project interpreter .venv/bin/python for local WAV validation." >&2
    return 1
  fi
  .venv/bin/python - "$wav_path" <<'PY'
import sys

import numpy as np
import soundfile as sf

path = sys.argv[1]
audio, rate = sf.read(path, dtype="float32", always_2d=True)
peak = float(np.max(np.abs(audio))) if audio.size else 0.0
rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
print({
    "sample_rate": int(rate),
    "channels": int(audio.shape[1]),
    "duration_seconds": round(float(audio.shape[0] / rate), 3),
    "peak": round(peak, 5),
    "rms": round(rms, 5),
    "raw_audio_saved": True,
})
if rate != 16_000 or audio.shape[1] != 1 or not audio.size:
    raise SystemExit("Capture did not produce a nonempty 16-kHz mono WAV")
PY
}

if [[ "$mode" == "inspect" ]]; then
  if ! command -v arecord >/dev/null; then
    echo "arecord is required; install alsa-utils." >&2
    exit 1
  fi
  echo "== ALSA hardware capture devices (no capture opened) =="
  arecord -l
  echo "== ALSA PCM names (no capture opened) =="
  arecord -L
  echo "== ALSA hardware playback devices (no playback opened) =="
  if command -v aplay >/dev/null; then
    aplay -l
    echo "== ALSA playback PCM names (no playback opened) =="
    aplay -L
  else
    echo "aplay is unavailable; install alsa-utils to inspect speaker readiness."
  fi
  echo "== ALSA cards =="
  cat /proc/asound/cards
  echo "== ALSA PCM capabilities =="
  cat /proc/asound/pcm
  echo "== Current audio-node owners (visibility may require sudo) =="
  fuser -v /dev/snd/* 2>&1 || true
  echo "Select a named capture PCM from arecord -L, for example:"
  echo "  plughw:CARD=Device,DEV=0"
  exit 0
fi

if [[ "$mode" == "validate" ]]; then
  if [[ -z "$input" || "$input" != results/* || ! -f "$input" ]]; then
    echo "--validate requires an existing WAV beneath results/ via --input." >&2
    exit 2
  fi
  validate_wav "$input"
  exit 0
fi

if ! command -v arecord >/dev/null; then
  echo "arecord is required; install alsa-utils." >&2
  exit 1
fi

if [[ -z "$device" ]]; then
  echo "--device is required for --${mode}; run --inspect first." >&2
  exit 2
fi

if [[ "$mode" == "probe" ]]; then
  echo "Inspecting ALSA PCM parameters (no audio file will be written): $device"
  arecord -D "$device" --dump-hw-params -f S16_LE -r 16000 -c 1 \
    -d 1 -t raw /dev/null
  echo "Probe completed. A successful open does not validate speech level or ME2 inference."
  exit 0
fi

if [[ "${ME2_AUDIO_CAPTURE_APPROVED:-}" != "1" ]]; then
  echo "Refusing to record. Set ME2_AUDIO_CAPTURE_APPROVED=1 for a separately approved test." >&2
  exit 2
fi
if [[ "$output" != results/* ]]; then
  echo "--output must be beneath results/ so test captures stay out of source paths." >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "Refusing to overwrite existing capture: $output" >&2
  exit 2
fi
mkdir -p "$(dirname "$output")"
echo "Recording ${seconds}s of approved audio to $output using $device"
echo "Speech content is not printed or logged by this helper."
arecord -D "$device" -f S16_LE -r 16000 -c 1 -d "$seconds" -V mono -t wav "$output"

validate_wav "$output"

echo "Delete the approved test capture when validation is complete; it is ignored by Git."
