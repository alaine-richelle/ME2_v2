# ME2_v2

Wake-word and voice-command recognition with PyTorch CPU inference.
Raspberry Pi 5 is a deployment target, not yet a measured/approved deployment.

## Current validation status — 2026-10-13

The retained legacy CTC weights remain the selected command fixture; no wake
artifact is deployment-approved. Runtime hardening passes 105 regression tests
and preserves all decisions in the fixed 100-row development check. Recognition
accuracy did not improve in this cycle. Live microphone validation is blocked
by native device discovery; no actual Pi measurements are available.

See the [measured hardening report](docs/HARDENING_2026-10-13.md) and
[runtime validation, logging, rollback and Pi runbook](docs/RUNTIME_VALIDATION.md).
The exact local model paths, hashes, matched metrics, and artifact retention
decisions are in the [2026-10-15 model-selection inventory](docs/MODEL_SELECTION_2026-10-15.md).
For the controlled built-in-microphone laptop preflight and plain-language
pipeline guide, see
[the laptop operator guide](docs/LAPTOP_OPERATOR_GUIDE_2026-10-16.md).
`python -m me2.health` is an offline operator/artifact check, not acoustic
approval. Use explicit model/config paths for installed-package execution.
The legacy selected grammar has no supported-slot metadata; the richer slot
contract described below belongs to the development pipeline.

## Layout

```text
ME2_v2/
├── wake-word/                 # wake recordings, generated bootstrap audio, canonical model
├── vcm/                       # HF-derived command data cache and canonical model
├── src/me2/                   # audio frontend, models, training, evaluation, CLI
├── config/default.json        # preprocessing and training settings
├── results/                   # JSON metrics, per-item CSV/JSONL, readable reports
├── logs/                      # epoch logs and runtime logs
└── tests/
```

Raw audio, Hugging Face cache data, generated training WAVs, and temporary
training states are not tracked by Git. Normal inference uses versioned `.me2`
artifacts containing safely loaded tensor arrays plus validated JSON metadata.
Legacy `.pkl` files are retained only for explicit trusted migration.

## Architecture

Both tasks share mono resampling and a 40-bin log-Mel frontend using 16 kHz
audio, a 400-sample FFT window, and a 160-sample hop. Wake v2 uses a
two-second fixed window and `TinyAudioCNN`; the historical artifact retains
its original 1.5-second contract. CTC command audio is trimmed,
right-padded to a five-second maximum, and decoded with its real usable length;
overlong commands reject instead of being silently cropped. No pretrained
audio encoder is used.

The command-stage training path now uses `CompactCTCModel`: a
depthwise-separable temporal CNN, projection layer, one small bidirectional GRU,
per-frame character logits, full-utterance CTC prefix beam search, a
training-only semantic trie, and a validation/calibration-frozen rejection
policy. The 8,052-parameter whole-utterance classifier remains available as a
reproducible baseline. The checked-in 3.65M-parameter `CharacterAudioCNN`
artifact is retained as a migrated legacy comparison and current inference
fixture until a fully trained compact artifact is selected.

The shared character contract preserves lowercase letters, digits, and spaces.
It does not silently delete digits or invent spoken-number expansions. Grammar
terminals carry explicit intent, variation, and supported slot metadata;
conflicting normalized meanings fail validation.

The canonical wake label remains `ateko`; its separate pronunciation guide is
`a-teh-ko`. The optional five-class model distinguishes `background`, `a`,
`te`, `ko`, and `ateko`, where versioned class `te` means the intended spoken
“teh” cue. The three syllable labels are never accepted as a completed wake
word. No exact IPA or stress pattern is asserted without an approved acoustic
anchor.

## VCM Dataset

The required source is [airimonda/ai231-me2-voice-commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) (DOI `10.57967/hf/10723`). The code requests only its default `train`, `test`, and `holdout` splits; it does not request `numerals`, `supplemental_synth`, `supplemental_fil`, or another voice-command dataset. The source card reports 10,733 train rows, 4,443 test rows, and 202 holdout rows; it specifies 16 kHz mono audio and states that speakers/synthetic voices do not cross those splits. The implementation checks speaker disjointness before training. HF audio is fetched undecoded and decoded by `soundfile`, avoiding an FFmpeg/TorchCodec runtime dependency.

Each row includes `audio`, `file`, `transcript`, `command`, `variation`, `slot_value`, `out_of_scope`, `speaker_id`, `source`, `is_synthetic`, `accent_group`, and other audit metadata. CTC targets use the normalized words actually recorded, including reliable out-of-scope transcripts. The legacy classifier uses `command` as the intent target, normalized to lowercase snake case, plus `out_of_scope`. The 19 source intent classes are:

`PLAY_MUSIC`, `WEATHER`, `TIME`, `LIGHT_ON`, `LIGHT_OFF`, `PAUSE`, `STOP`, `NEXT`, `VOLUME_UP`, `VOLUME_DOWN`, `CALL`, `MESSAGE`, `LIST_REMINDERS`, `TIMER`, `ALARM`, `TEMPERATURE`, `BRIGHTNESS`, `COLOR`, and `CREATE_REMINDER`.

The CTC grammar maps accepted terminals to intent, variation, and enumerated
`slot_value` metadata. It does not claim arbitrary free-text, contact, number,
or time recognition. The baseline predicts 20 labels (19 intents plus
`OUT_OF_SCOPE`) and has no transcript or slot output. The card describes the
aggregate release as research and education only, non-commercial, with
component-specific source terms; review the dataset's `LICENSE.md` before use
or redistribution.

Validation and calibration are selected by speaker from official train. Test is
used only after model selection and threshold calibration; holdout remains
separate for deployment checks. Audio is inspected, and invalid examples are
quarantined with reasons and coverage checks. Preprocessing converts to mono
16 kHz, trims low-energy edges, normalizes audible speech without amplifying
very quiet/noise-only audio, and right-pads CTC audio while preserving usable
lengths. Training-only augmentation includes gain, background-like Gaussian
noise, temporal shift, and modest speed/pitch perturbation. Validation,
calibration, and test waveforms are not augmented.

## Wake Dataset

Wake dataset v3 records stable IDs, hashes, parent/augmentation lineage,
pronunciation review, pseudonymous speaker/session/device/room identity,
real/synthetic provenance, TTS settings, duration, sample rate, boundaries,
frozen split, and schema versions. It explicitly synthesizes or prompts
“teh” and the user-confirmed connected synthesis text `a teh ko`. The written
guide remains `a-teh-ko`, but hyphenated TTS text is not used for new positive
generation. Generated speech remains pending until each voice/setting is
auditioned. The original v1 corpus and model are preserved.

Splits are frozen by speaker/voice before augmentation. Training refuses a
corpus with no pronunciation-approved complete-word positives unless
`--allow-pending-review` is explicitly supplied for a non-deployment
experiment. Such artifacts carry `deployment_ready=false`, and live
microphone use blocks them by default. See
[`docs/EXPERIMENTS_2026-10-05.md`](docs/EXPERIMENTS_2026-10-05.md) for the
current measured results and [`docs/WAKE_WORD_V2.md`](docs/WAKE_WORD_V2.md)
for the retained historical audit.

The latest follow-up data expansion, training, calibration, promotion
decisions, and cleanup are recorded in
[`docs/IMPROVEMENT_CYCLE_2026-10-06.md`](docs/IMPROVEMENT_CYCLE_2026-10-06.md).
External data-source decisions and removed redundant files are listed in
[`docs/DATA_SOURCE_LEDGER_2026-10-06.md`](docs/DATA_SOURCE_LEDGER_2026-10-06.md)
and [`docs/CLEANUP_2026-10-06.md`](docs/CLEANUP_2026-10-06.md).

## Setup

Python 3.10+ is required. On macOS/Linux with `uv`:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[train,dev,microphone]'
```

If a microphone is not needed, install only `.[dev]`. PyTorch wheels for the target architecture should be selected from the current PyTorch installation instructions if the default package index does not provide a Raspberry Pi ARM64 wheel. On Raspberry Pi OS 64-bit, install PortAudio system libraries before the optional `sounddevice` package when using a microphone, for example `sudo apt install libportaudio2`. The file-based workflow does not need PortAudio.

## Reproduce

Run commands from the repository root with the virtual environment active (`source .venv/bin/activate` on macOS/Linux). Set the local source import path once:

```sh
export PYTHONPATH=src
```

The first VCM command downloads the required HF splits; the full dataset release is multi-gigabyte, but this project does not request the separate large numerals split.

Inspect the source split structure and observed command counts:

```sh
python -m me2.vcm.data --describe
```

Generate and review a wake audition set before scaling synthesis (macOS only):

```sh
python -m me2.wake_word.data preview \
  --output wake-word/data/v3/audition
python -m me2.wake_word.data audit \
  wake-word/data/v3/audition/manifest.csv
python -m me2.wake_word.record \
  --speaker speaker-001 --session session-01 --take-count 22
```

Audit the pinned dataset revision, persisted speaker partitions, character
contract, grammar collisions, and CTC alignment feasibility:

```sh
python -m me2.vcm.audit
```

Train, calibrate, and score the compact CTC VCM:

```sh
python -m me2.vcm.train \
  --architecture CompactCTCModel \
  --seed 42 \
  --epochs 30 \
  --target-oos-far 0.05
```

This writes the safe artifact `vcm/models/vcm_ctc_model.me2`, epoch history
under `logs/vcm_ctc_training.jsonl`, persisted partitions/quarantine reports,
and test results under `results/vcm_ctc_test_v2`. Model selection uses the
speaker-disjoint validation partition; rejection thresholds use the separate
calibration partition; official test is evaluated after selection and
calibration.

Keep the whole-utterance classifier as an explicit baseline:

```sh
python -m me2.vcm.train \
  --architecture TinyAudioCNN \
  --seed 42 \
  --epochs 30 \
  --output vcm/models/vcm_baseline_model.me2
```

For a cached-data wiring check only:

```sh
python -m me2.vcm.train \
  --architecture CompactCTCModel \
  --epochs 1 \
  --smoke-limit 128 \
  --skip-test-evaluation \
  --output vcm/models/vcm_ctc_smoke.me2
```

The smoke score is not model-quality evidence. Historical, newly reproduced,
and still-unverified results are separated in
[`docs/CTC_PIPELINE.md`](docs/CTC_PIPELINE.md).

Wake training selects model weights on validation, calibrates the threshold on
a distinct calibration partition, then runs read-only test evaluation. VCM
training writes epoch logs and scores the official speaker-disjoint test
split. Explicit evaluation commands are:

```sh
python -m me2.wake_word.train \
  --manifest wake-word/data/v3/recordings/manifest.csv \
  --label-mode binary --seed 42 --epochs 30
python -m me2.wake_word.evaluate \
  --manifest wake-word/data/v3/recordings/manifest.csv \
  --split test
python -m me2.vcm.evaluate \
  --model vcm/models/vcm_ctc_model.me2 \
  --split test \
  --output results/vcm_ctc_test_v2
python -m me2.vcm.evaluate --split holdout --output results/vcm_holdout
python -m me2.vcm.analyze \
  --predictions results/vcm_ctc_test_v2/predictions.jsonl \
  --model vcm/models/vcm_ctc_model.me2 \
  --output results/vcm_ctc_test_v2_analysis
```

Calibrate an existing CTC artifact without changing its acoustic weights:

```sh
python -m me2.vcm.calibrate \
  --model vcm/models/vcm_ctc_model.me2 \
  --output-model vcm/models/vcm_ctc_calibrated.me2 \
  --partitions results/vcm_ctc_partitions.json \
  --output results/vcm_ctc_calibration
```

The calibrated artifact must still satisfy the documented macro-F1,
false-accept, false-reject, and acceptance-coverage promotion guardrails.

Generate the held-out Raspberry Pi evaluation report from structured Pi
events, trial ground truth, and telemetry:

```sh
python -m me2.pi_report \
  --metadata pi-run/run_metadata.json \
  --manifest pi-run/ground_truth.csv \
  --events pi-run/events.jsonl \
  --telemetry pi-run/telemetry.csv \
  --output reports/<run_id>
```

The versioned 19-intent/93-variation schema is
`config/command_schema_v1.json`. See
[`docs/PI_EVALUATION_REPORT.md`](docs/PI_EVALUATION_REPORT.md) for the exact
input contract, scoring rules, and unavailable-field policy. The checked-in
`reports/pi-evaluation-pending-2026-10-06/` artifact is intentionally marked
not run because no Pi event log or telemetry was available; it is not a host
benchmark presented as Pi evidence.

Migrate a locally verified legacy pickle without overwriting it:

```sh
python -m me2.bundle migrate --trusted-legacy \
  vcm/models/vcm_ctc_model.pkl \
  vcm/models/vcm_ctc_model.me2
```

Normal loading refuses `.pkl` artifacts. Migration is only for artifacts whose
provenance has been independently verified. Holdout is a one-time deployment
check, not tuning data.

Training is deterministic where supported by the installed PyTorch backend. Seed `42` is the default; choose and record another seed with `--seed`. Epoch logs are JSONL under `logs/`. Evaluation writes `metrics.json`, `predictions.csv`, `predictions.jsonl`, and `report.txt` under `results/`. Reports include accuracy, macro precision/recall/F1, ROC-AUC when defined, confusion matrix, per-class scores, observed frequent confusions, and measured inference footprint. Per-item VCM output includes input path, gold command, predicted command, confidence, correctness, speaker, and source. False accepts are out-of-scope audio accepted as a command; false rejects are in-scope commands rejected as out of scope.

## Inference

Individual WAV checks:

```sh
python -m me2.app --wake-word --audio path/to/ateko.wav
python -m me2.app --vcm --audio path/to/command.wav
```

Run the full wake-gated workflow from two WAV files, without microphone hardware:

```sh
python -m me2.app --interactive \
  --wake-audio path/to/ateko.wav \
  --command-audio path/to/command.wav
```

With a working microphone (install `.[microphone]`). Use `--input-device` to select a device by name or numeric index when the system default is not the desired microphone:

```sh
python -m me2.app --interactive
python -m me2.app --interactive --input-device "MacBook Pro Microphone"
python -m me2.app --wake-word
python -m me2.app --vcm
```

Microphone mode uses one continuous stream and the explicit state sequence
`LISTENING_FOR_WAKE → CAPTURING_COMMAND → DECODING →
ACCEPTED_OR_REJECTED → COOLDOWN → LISTENING_FOR_WAKE`. Pre-roll preserves
immediate command onset after `ateko`; endpoint, timeout, overlong capture, and
overflow paths reject without dispatching an action. Runtime logs go to the
console and `logs/app.log`.

## Raspberry Pi 5

For the complete boot-media-to-dashboard setup, follow
[`docs/RASPBERRY_PI_5_WORKFLOW.md`](docs/RASPBERRY_PI_5_WORKFLOW.md).
It covers Raspberry Pi Imager, OS/audio setup, selected model transfer,
on-device checks, intent/action mapping, startup service, and rollback.

For Ubuntu selected from Raspberry Pi Imager's Ubuntu menu, or a Pi-specific
Ubuntu image selected through **Use custom**, use
[`docs/RASPBERRY_PI_5_UBUNTU_WORKFLOW.md`](docs/RASPBERRY_PI_5_UBUNTU_WORKFLOW.md).
It includes Ubuntu Server cloud-init templates and the revised installation,
networking, audio and service workflow.

For this deadline's operator-gated target check, see the
[`2026-10-15 Pi SSH-safe validation handoff`](docs/PI5_SSH_SAFE_VALIDATION_2026-10-15.md).
It records the unresolved target/approval gate, frozen artifact identities,
offline checks, and telemetry steps without assuming an SSH host. For a new
approved Pi setup, the user supplies a credential-free remote, branch/tag, and
exact pushed SHA; the Pi clones and checks out that SHA detached rather than
receiving a local tar/ZIP archive or working-tree copy.

Try the included dashboard without a microphone:

```sh
python -m me2.dashboard --demo --autostart
```

Open `http://127.0.0.1:8765`. The Atecco dashboard displays the wake → capture
→ decode → decision → cooldown pipeline, model details, all 19 canonical
intent actions, and local demonstration state for player, light, weather,
time, reminders, timers, alarms, and simulated call/message actions. Demo
scores and sample slots are synthetic. Weather uses the configured city and
an optional network lookup; voice inference remains local. For live input,
configure `config/dashboard.json` and omit `--demo`; the Pi microphone supplies
audio, including when the browser connects through SSH. Current wake artifacts
are not deployment-ready and are blocked by default. The guide documents the
explicit supervised engineering override.

The three-second onset candidate changes only that capture deadline and keeps
the baseline's other settings fixed:
`config/dashboard.natural-pause.json` (dashboard) and
`config/runtime.onset-3s.json` (CLI). It is not the default until matched
validation and approved live testing support promotion. The engineer handoff
prompt is
[`AI231 ME2/ME2_Voice_Command_Dashboard_Mapping_and_Final_Model_Prompt.md`](AI231%20ME2/ME2_Voice_Command_Dashboard_Mapping_and_Final_Model_Prompt.md).
The dated implementation evidence and unresolved release gates are in
[`docs/DEADLINE_PLAN_PROGRESS_2026-10-15.md`](docs/DEADLINE_PLAN_PROGRESS_2026-10-15.md).

Install inference-only dependencies on 64-bit Raspberry Pi OS with `pip install
-e .`; add `.[microphone]` only for live input. Training dependencies are kept
in the separate `train` extra. Normal startup performs no dataset or model
download.

Benchmark a representative command on the actual Pi:

```sh
python -m me2.benchmark path/to/command.wav \
  --model vcm/models/vcm_ctc_model.me2 \
  --iterations 200 \
  --threads 2 \
  --output results/pi5_vcm_benchmark.json
```

Benchmark the wake frontend and CNN separately with an approved wake
recording:

```sh
python -m me2.benchmark path/to/approved-a-teh-ko.wav \
  --model wake-word/models/wake_word_v5_regularized_seed43.me2 \
  --iterations 200 \
  --threads 2 \
  --output results/pi5_wake_benchmark.json
```

The report separates frontend, acoustic-model, decoder/rejection, and complete
inference latency and records model bytes, parameters, process RSS, active-loop
CPU, Python, PyTorch, and machine information. Temperature, throttling,
sustained idle CPU, and endpointer waiting still require target-hardware
measurement.

## Limits and interpretation

The checked-in reports include historical and newly reproduced measurements;
read their provenance before comparison. A compact CTC smoke run validates the
pipeline but is not a deployable trained model. Wake-word generalization still
requires a separate real-speaker/noisy-room corpus, and Raspberry Pi claims
require an actual Pi benchmark. Test and holdout are never threshold-tuning
partitions.
