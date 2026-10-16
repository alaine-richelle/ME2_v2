# Supervised laptop operator guide — 2026-10-16

## What this guide is for

Use this guide only while an operator is present at the laptop with the intended
microphone attached and wireless earphones paired. It is an engineering
validation walkthrough, not a deployment approval. Every live command is
bounded or stopped by the operator, retains no raw audio, and writes only
redacted structured diagnostics.

The Raspberry Pi scope remains separate: it permits package/import checks,
deterministic supplied-WAV model replay, and dashboard `--demo` review only.
Do not use this laptop guide to claim Pi readiness.

## What happens from speech to a simulated action

1. **Audio capture.** The microphone supplies a stream of sound. The program
   converts it to the fixed mono sample rate expected by the models, so an
   input device's native format does not silently change recognition.
2. **Wake windows and features.** The program repeatedly examines short,
   overlapping pieces of sound. Each piece is converted into compact
   frequency features. A *hop* is the amount of time before the next
   overlapping window; overlap helps avoid missing a word that starts between
   checks.
3. **Wake model and smoothing.** A small neural network (CNN) estimates
   whether a window contains the connected wake word “a-teh-ko.” *Smoothing*
   requires consistent nearby wake results, reducing one-window noise
   triggers.
4. **Dual-wake roles.** V5 is the primary decision source. V3 is always
   scored beside it for diagnostics, but a v3 wake never overrides a normal
   v5 non-wake. V3 becomes an active fallback only if v5 cannot load or has
   an inference/system fault. The terminal and dashboard show degraded mode
   when that happens.
5. **Command capture.** A valid wake starts command capture. *Pre-roll*
   keeps sound just before the wake so an immediate command is not clipped.
   The *onset timeout* limits how long the app waits for speech; *endpoint
   silence* ends capture after speech stops; and the duration cap rejects
   longer-than-supported audio instead of silently cutting it.
6. **CTC command decoding and grammar.** The command model creates character
   probabilities. *CTC* is a method for aligning many sound frames with a
   shorter text sequence. The *grammar* is the approved set of command
   phrases, so uncertain sound is not converted into an arbitrary action.
7. **Accept or reject.** A command is accepted only when it satisfies the
   decoder and safety policy. Silence, ordinary speech, unsupported phrases,
   and system faults can be rejected safely.
8. **Dashboard action and cooldown.** An accepted mapped command changes only
   local simulated dashboard state. *Cooldown* prevents the same utterance
   from firing repeatedly. Calls, messages, home-device controls, and other
   external actions remain demonstrations only.

## Current capabilities and limits

| Component | What it can demonstrate now | What it cannot establish yet |
|---|---|---|
| V5 wake candidate | File loading/replay and supervised laptop trials | Reliable unattended detection or a low false-trigger rate |
| V3 wake fallback/fixture | File replay, side-by-side diagnostics, and v5-fault fallback | Reliable live detection; its comparable test detected 0/3 wakes |
| CTC command model | Recognition/rejection for phrases in its legacy grammar | Robust open-ended speech, semantic slots, or zero OOS acceptance |
| Laptop dashboard | Local simulated actions and local `play_music` output after recognition | Physical-device control, calls/messages, or proof of model quality |
| Raspberry Pi 5 | Package/import checks, deterministic model replay, and demo-dashboard reachability | Deployment readiness from deferred live hardware tests |

V5 is a supervised candidate, not a promoted model: its comparable small test
detected 2/3 wake events but made 4/48 false accepts, exceeding the 2/48
guardrail. V3 remains a retained diagnostic/fault fallback and dashboard
fixture because it made 0/48 false accepts, but it detected 0/3 wake events.
The CTC bundle remains development-only because it has 20/76 OOS false
accepts and a legacy uncalibrated acceptance policy.

## Before you begin

Run the following from the repository root. Do not run the environment
installation commands unless `.venv` is missing or needs repair. Select the
wireless earphones as the macOS system output in Control Center before the
output check. The earphones are only for hearing local `play_music`; wake and
command recognition use the microphone only. The command-line app prints
recognition state to the terminal and does not speak a voice response.

```sh
cd '/Users/alaineramos/Documents/2025 Files - Mac/School/AI231/ME2_v2'

# Run only if .venv is missing or the local environment needs repair.
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e '.[dev,microphone]'

# Find the exact microphone name; do not rely on a numeric index.
.venv/bin/python -c 'import sounddevice as sd; print(sd.query_devices())'
export ME2_MIC='EXACT MICROPHONE NAME FROM THE LIST'

# Device visibility only: this does not capture audio.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.health \
  --config config/runtime.example.json \
  --devices \
  --input-device "$ME2_MIC" \
  --output results/laptop-device-check-20261016.json
```

If device visibility fails, fix macOS microphone permission, cable/adapter, or
the selected device first. Do not change a model threshold to work around a
device problem.

## 1. Deterministic file replays — no microphone

These commands prove only that the model files load and make decisions on
supplied WAV files. They do not test the microphone, earphones, live
recognition, or Pi.

```sh
# Primary v5 wake file replay.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --wake-word --config config/runtime.example.json \
  --wake-model wake-word/models/wake_word_v5_regularized_seed43.me2 \
  --audio wake-word/data/v2/audition/43fd11c385fc_guide_connected.wav \
  --log-file logs/laptop-v5-wake-file-replay-20261016.jsonl

# V3 file replay for side-by-side comparison.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --wake-word --config config/runtime.example.json \
  --wake-model wake-word/models/wake_word_v3_binary_seed43_r1.me2 \
  --audio wake-word/data/v2/audition/43fd11c385fc_guide_connected.wav \
  --log-file logs/laptop-v3-wake-file-replay-20261016.jsonl

# CTC command-only file replay.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --vcm --config config/runtime.example.json \
  --audio vcm/data/test_sample.wav \
  --log-file logs/laptop-vcm-file-replay-20261016.jsonl

# Full v5 wake-to-command handoff using files, not a microphone.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --interactive --config config/runtime.example.json \
  --wake-audio wake-word/data/v2/audition/43fd11c385fc_guide_connected.wav \
  --command-audio vcm/data/test_sample.wav \
  --threads 2 --allow-unverified-wake-model \
  --log-file logs/laptop-v5-integrated-file-replay-20261016.jsonl
```

## 2. Supervised microphone tests

State the phrase before each command. Run one command at a time. For the
wake-only commands, say connected “a-teh-ko” once or a few times, then use
`Ctrl-C` after the intended attempts. Keep the same device and configuration
when comparing v5 and v3. A missed wake or rejected command is useful
diagnostic evidence; do not alter thresholds during the comparison.

```sh
# Say “a-teh-ko”; v5 only.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --wake-word --config config/runtime.example.json \
  --wake-model wake-word/models/wake_word_v5_regularized_seed43.me2 \
  --input-device "$ME2_MIC" --allow-unverified-wake-model \
  --log-file logs/laptop-v5-wake-live-20261016.jsonl

# Repeat with the same phrase/device for v3 only.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --wake-word --config config/runtime.example.json \
  --wake-model wake-word/models/wake_word_v3_binary_seed43_r1.me2 \
  --input-device "$ME2_MIC" --allow-unverified-wake-model \
  --log-file logs/laptop-v3-wake-live-20261016.jsonl

# Dual-wake full path: say “a-teh-ko”, then “play music”.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --interactive --config config/runtime.laptop-dual-wake.json \
  --wake-fallback-model wake-word/models/wake_word_v3_binary_seed43_r1.me2 \
  --input-device "$ME2_MIC" --threads 2 \
  --session-seconds 30 --allow-unverified-wake-model \
  --log-file logs/laptop-dual-wake-live-20261016.jsonl
```

The dual-wake command must show v5 primary and v3 diagnostic outcomes. If v5
has a fault, it must print a degraded-mode warning before v3 can trigger
capture. If v5 is healthy and says non-wake, v3 disagreement is only logged;
no command capture or dashboard action should occur.

## 3. Wireless-earphone output check

This is a standalone three-second local tone. It opens an output stream only:
it does not open a microphone, record audio, use a network service, or change
dashboard state.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  scripts/laptop_output_check.py --seconds 3 --speaker-approved
```

If no sound is heard, select the earphones as macOS output in Control Center
and retry. Treat a routing failure as an output-device issue, not evidence
about wake or command recognition.

## 4. Dashboard reviews

First, run this UI-only review. It uses synthetic inputs and no model,
microphone, or speaker output.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.dashboard \
  --demo --port 8765 --threads 2 \
  --log-file logs/laptop-dashboard-demo-20261016.log
```

Open `http://127.0.0.1:8765`, use only visible demo controls, then stop it
with `Ctrl-C`. Demo events prove neither microphone recognition nor model
quality.

Only after the separate microphone and earphone checks, start the supervised
local dashboard. Open the same local address, use its visible control to
start listening, say “a-teh-ko” then “play music,” and verify music reaches
the selected earphones. Stop with the visible control or `Ctrl-C`.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.dashboard \
  --config config/dashboard.laptop-dual-wake.json \
  --input-device "$ME2_MIC" --allow-unverified-wake-model \
  --port 8765 --threads 2 \
  --log-file logs/laptop-dashboard-live-20261016.log
```

The dashboard dispatcher is local and simulated: `call` and `message` do not
contact anyone, and no physical device, GPIO, external account, or network
action is authorized. If recognition fails, record the displayed state and
redacted log outcome. Never use a dashboard demo event as proof that the
microphone/model path works.

## How to report a result

Record the command used, selected microphone/output device category, whether
v5 was primary or v3 was active fallback, visible pipeline state, accepted or
rejected outcome, and the redacted log path. Do not include raw recordings,
transcripts, account details, or credentials. Do not describe any result as
deployment-ready.
