#!/usr/bin/env bash
# Deterministic Pi VCM replay. It never opens a microphone or writes raw audio.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

vcm_model="${ME2_VCM_MODEL:-vcm/models/vcm_ctc_model.me2}"
command_wav="${ME2_COMMAND_WAV:?Set ME2_COMMAND_WAV to an approved supported-command WAV.}"
run_id="${ME2_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

test -f "$vcm_model"
test -f "$command_wav"

echo "VCM model replay: $vcm_model"
echo "This is a supplied-WAV component diagnostic, not a permitted live command path."
sha256sum "$vcm_model"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --vcm --vcm-model "$vcm_model" --audio "$command_wav" --threads 2 \
  --log-file "logs/pi5-vcm-replay-${run_id}.jsonl"
