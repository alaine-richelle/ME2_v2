# Supervised laptop operator guide — 2026-10-16

## What this guide is for

Use this guide only while an operator is present at the laptop using the
built-in **MacBook Pro Microphone**. It is controlled local preflight and
replay evidence, not a deployment approval. Every live command is bounded or
stopped by the operator, retains no raw audio, and writes only redacted
structured diagnostics.

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
4. **Wake decision.** V5 is the sole runtime decision source. V3 is retained
   only for separate diagnostics; it cannot override a v5 non-wake or trigger
   command capture. A v5 load, frontend, or inference fault fails closed.
5. **Wake acknowledgment.** Only after the smoothed active wake decision,
   the dashboard displays **“Yes? Listening for your command.”** with a
   visual pulse. In a future approved Pi session, the same event may request
   one short wired-speaker chime. It confirms that wake capture began; it
   does not confirm that a command will be understood.
6. **Command capture.** A valid wake starts command capture. *Pre-roll*
   keeps sound just before the wake so an immediate command is not clipped.
   The *onset timeout* limits how long the app waits for speech; *endpoint
   silence* ends capture after speech stops; and the duration cap rejects
   longer-than-supported audio instead of silently cutting it.
7. **CTC command decoding and grammar.** The command model creates character
   probabilities. *CTC* is a method for aligning many sound frames with a
   shorter text sequence. The *grammar* is the approved set of command
   phrases, so uncertain sound is not converted into an arbitrary action.
8. **Accept or reject.** A command is accepted only when it satisfies the
   decoder and safety policy. Silence, ordinary speech, unsupported phrases,
   and system faults can be rejected safely.
9. **Dashboard action and cooldown.** An accepted mapped command changes only
   local simulated dashboard state. *Cooldown* prevents the same utterance
   from firing repeatedly. Calls, messages, home-device controls, and other
   external actions remain demonstrations only.

## Current capabilities and limits

| Component | What it can demonstrate now | What it cannot establish yet |
|---|---|---|
| V5 wake candidate | File loading/replay and supervised laptop trials | Reliable unattended detection or a low false-trigger rate |
| V3 wake fixture | File replay and side-by-side diagnostics | Reliable live detection; its comparable test detected 0/3 wakes |
| CTC command model | Recognition/rejection for phrases in its legacy grammar | Robust open-ended speech, semantic slots, or zero OOS acceptance |
| Laptop preflight | Built-in-microphone device visibility and controlled replay | Pi readiness, physical-device control, or proof of model quality |
| Raspberry Pi 5 | Package/import checks, deterministic model replay, and demo-dashboard reachability | Deployment readiness from deferred live hardware tests |

V5 is a supervised candidate, not a promoted model: its comparable small test
detected 2/3 wake events but made 4/48 false accepts, exceeding the 2/48
guardrail. V3 remains a retained diagnostic fixture because it made 0/48 false
accepts, but it detected 0/3 wake events.
The CTC bundle remains development-only because it has 20/76 OOS false
accepts and a legacy uncalibrated acceptance policy.

## Before you begin

Run the following from the repository root. Do not run the environment
installation commands unless `.venv` is missing or needs repair. Laptop checks
use the built-in MacBook Pro Microphone and print recognition state only; they
do not play audio or produce a voice response.

```sh
cd '/Users/alaineramos/Documents/2025 Files - Mac/School/AI231/ME2_v2'

# Run only if .venv is missing or the local environment needs repair.
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e '.[dev,microphone]'

# Confirm that macOS exposes the built-in microphone. This does not capture audio.
.venv/bin/python -c 'import sounddevice as sd; print(sd.query_devices())'

# Device visibility only: this does not capture audio.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.health \
  --config config/runtime.example.json \
  --devices \
  --input-device 'MacBook Pro Microphone' \
  --output results/laptop-device-check-20261016.json
```

If device visibility fails, fix macOS microphone permission, cable/adapter, or
the selected device first. Do not change a model threshold to work around a
device problem.

## 1. Deterministic file replays — no microphone

These commands prove only that the model files load and make decisions on
supplied WAV files. They do not test the microphone, live recognition, or Pi.

The two example WAV paths below are available only in the local working tree;
they are not Git-tracked Pi fixtures. Do not copy or substitute them for the
separate Pi replay requirement.

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

# Full canonical v5 wake-to-command handoff using files, not a microphone.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --interactive --config config/runtime.example.json \
  --wake-audio wake-word/data/v2/audition/43fd11c385fc_guide_connected.wav \
  --command-audio vcm/data/test_sample.wav \
  --threads 2 \
  --log-file logs/laptop-v5-integrated-file-replay-20261016.jsonl
```

For the negative gate check, supply a separately reviewed non-wake WAV. The
expected result is `Wake word not detected`; the command WAV must not be
decoded and the log must not contain a `command_result` event.

```sh
export ME2_NON_WAKE_WAV='PATH_TO_REVIEWED_NON_WAKE.wav'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --interactive --config config/runtime.example.json \
  --wake-audio "$ME2_NON_WAKE_WAV" \
  --command-audio vcm/data/test_sample.wav --threads 2 \
  --log-file logs/laptop-v5-gate-negative-replay-20261016.jsonl
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
  --input-device 'MacBook Pro Microphone' --allow-unverified-wake-model \
  --log-file logs/laptop-v5-wake-live-20261016.jsonl

# Repeat with the same phrase/device for v3 only.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --wake-word --config config/runtime.example.json \
  --wake-model wake-word/models/wake_word_v3_binary_seed43_r1.me2 \
  --input-device 'MacBook Pro Microphone' --allow-unverified-wake-model \
  --log-file logs/laptop-v3-wake-live-20261016.jsonl

# Canonical gated path: say “a-teh-ko”, then “play music”.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --interactive --config config/runtime.example.json \
  --input-device 'MacBook Pro Microphone' --threads 2 \
  --session-seconds 30 --allow-unverified-wake-model \
  --log-file logs/laptop-v5-gated-live-20261016.jsonl
```

The canonical runtime has no wake fallback. A healthy v5 non-wake remains in
wake listening, and a v5 load, frontend, or inference fault fails closed with
no command capture. V3 is retained only for the separate file/live diagnostic
commands above; it is not an OR rule and cannot activate VCM.

## 3. Dashboard demo review

First, run this UI-only review. It uses synthetic inputs and no model,
microphone, or speaker output.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.dashboard \
  --demo --port 8765 --threads 2 \
  --log-file logs/laptop-dashboard-demo-20261016.log
```

Open `http://127.0.0.1:8765`, use only visible demo controls, then stop it
with `Ctrl-C`. Demo events prove neither microphone recognition nor model
quality. They do show the visual **“Yes? Listening for your command.”** pulse;
the demo never plays a chime.

Do not use a live laptop dashboard as Pi evidence. The Pi supplied-WAV and
separately approved combined-microphone commands in
`PI5_SSH_SAFE_VALIDATION_2026-10-15.md` are the primary operational workflow.

## Future Raspberry Pi wired-speaker acknowledgment

The Pi dashboard configuration is
`config/dashboard.pi5.json`. A read-only Pi ALSA inventory found the
USB capture/playback interface as `plughw:CARD=Device,DEV=0`; the versioned Pi
profiles use that named device rather than a numeric card index. ALSA
visibility is hardware-path evidence only. It does not prove microphone
recognition, speaker audibility, or model quality.

The acknowledgment chime remains deliberately **disabled** even with that
configured device. This separation matters: the chime is a small generated
tone sent to ALSA, while wake and command recognition are model decisions from
microphone audio. Hearing a chime would prove only that the dashboard received
a valid wake decision.

After hardware delivery and explicit approval, on the Pi first list the
playback devices:

```sh
aplay -l
```

The configured device name remains `plughw:CARD=Device,DEV=0`; do not switch
to Bluetooth or HDMI. Keep `wake_acknowledgement.enabled` set to `false` until
a separately approved bounded playback test confirms the wired output. The
dashboard sends no saved audio file and limits each generated 880-Hz chime to
120 ms. A device failure is non-fatal: the on-screen acknowledgment stays
visible, command capture continues, and a redacted failure event is logged.

## How to report a result

Record the command used, the built-in MacBook Pro Microphone, visible pipeline
state, accepted or rejected outcome, and the redacted log path. Do not include
raw recordings, transcripts, account details, or credentials. Do not describe
any result as deployment-ready.
