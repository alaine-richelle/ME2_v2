#!/usr/bin/env bash
# Deterministic Pi wake-model replay. It never opens a microphone or writes raw audio.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

wake_model="${ME2_WAKE_MODEL:-wake-word/models/wake_word_v5_regularized_seed43.me2}"
wake_wav="${ME2_WAKE_WAV:?Set ME2_WAKE_WAV to an approved connected-wake WAV.}"
run_id="${ME2_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

test -f "$wake_model"
test -f "$wake_wav"

echo "Wake model replay: $wake_model"
echo "Input is supplied WAV only; no microphone capture is authorized."
sha256sum "$wake_model"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --wake-word --wake-model "$wake_model" --audio "$wake_wav" --threads 2 \
  --log-file "logs/pi5-v5-wake-replay-${run_id}.jsonl"
