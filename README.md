# ME2_v2

Small-footprint wake-word and voice-command recognition, trained with PyTorch and designed for CPU inference on a Raspberry Pi 5.

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

Raw audio, Hugging Face cache data, generated training WAVs, and temporary training states are not tracked by Git. The final inference artifacts are `wake-word/models/wake_word_model.pkl` and `vcm/models/vcm_model.pkl`; each is a bundle of CPU-compatible PyTorch state, labels, and preprocessing metadata.

## Architecture

Both tasks share mono resampling, silence trim/pad/crop, a 40-bin log-Mel frontend, and a small depthwise-separable CNN (`TinyAudioCNN`). The frontend uses 16 kHz audio, a 400-sample FFT window, and a 160-sample hop. The command model uses a 5-second window; the wake detector uses 1.5 seconds. No ASR model or pretrained audio encoder is used.

The wake classifier distinguishes `background`, `a`, `te`, `ko`, and `ateko`. The three syllable labels are useful acoustic cues but are explicitly **not** accepted as a completed wake word. Only the complete-word output can pass the wake threshold. This models the ordered `a-te-ko` pronunciation rather than treating arbitrary occurrences of the full spelling as equivalent. Phoneme-level alignment is not available in the bootstrap corpus; these are syllable-level supervised examples, not frame-level phoneme labels.

## VCM Dataset

The required source is [airimonda/ai231-me2-voice-commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) (DOI `10.57967/hf/10723`). The code requests only its default `train`, `test`, and `holdout` splits; it does not request `numerals`, `supplemental_synth`, `supplemental_fil`, or another voice-command dataset. The source card reports 10,733 train rows, 4,443 test rows, and 202 holdout rows; it specifies 16 kHz mono audio and states that speakers/synthetic voices do not cross those splits. The implementation checks speaker disjointness before training. HF audio is fetched undecoded and decoded by `soundfile`, avoiding an FFmpeg/TorchCodec runtime dependency.

Each row includes `audio`, `file`, `transcript`, `command`, `variation`, `slot_value`, `out_of_scope`, `speaker_id`, `source`, `is_synthetic`, `accent_group`, and other audit metadata. The classifier uses `command` as the intent target, normalized to lowercase snake case, plus `out_of_scope`. The 19 source intent classes are:

`PLAY_MUSIC`, `WEATHER`, `TIME`, `LIGHT_ON`, `LIGHT_OFF`, `PAUSE`, `STOP`, `NEXT`, `VOLUME_UP`, `VOLUME_DOWN`, `CALL`, `MESSAGE`, `LIST_REMINDERS`, `TIMER`, `ALARM`, `TEMPERATURE`, `BRIGHTNESS`, `COLOR`, and `CREATE_REMINDER`.

The 93 `variation` values and `slot_value` field are retained in the source rows for analysis but are not classifier outputs. The default model therefore predicts 20 labels (19 intents plus `OUT_OF_SCOPE`), not a phrase/slot combination. The card describes the aggregate release as research and education only, non-commercial, with component-specific source terms; review the dataset's `LICENSE.md` before use or redistribution.

The validation set is selected by speaker from the official train split. Test is used for final evaluation after training and never for model selection. Holdout is kept separate for deployment/live Pi checks. Audio is inspected and corrupted examples are logged and skipped. Preprocessing converts to mono 16 kHz, trims low-energy edges, normalizes audible speech without amplifying very quiet/noise-only audio, and centers/crops or zero-pads to five seconds. Training-only augmentation includes gain, background-like Gaussian noise, temporal shift, and modest speed/pitch perturbation. No validation or test waveform is augmented.

## Wake Dataset

On macOS, the starter corpus uses the built-in `say` synthesizer to generate the individual syllables, the spaced full pronunciation, complete `ateko`, near-miss phrases, varied speaking rates, several available English voices, and low-level synthetic noise. The generated manifest records source, label, voice, phrase, and split. Splits are grouped by voice, not by WAV file. The trainer also accepts the same manifest schema for recordings created with the guided recorder.

These scores are bootstrap diagnostics only: multiple built-in TTS voices are not a substitute for independent human speakers, real microphones, room acoustics, or representative background noise. The default evaluation reports both five-way classification and binary complete-wake false accepts/rejects; isolated `a`, `te`, and `ko` count as negatives for event detection. The acceptance threshold is selected on validation data and then fixed for test reporting. Add multiple consenting speakers and varied real background audio before treating wake performance as deployment-ready.

The recorder is optional and requires the microphone dependency. It writes separate WAVs and a CSV manifest with stable speaker IDs. Keep the recording labels accurate: isolated syllables and truncated phrases are negative wake examples; only a correctly spoken full `ateko` is positive.

## Setup

Python 3.10+ is required. On macOS/Linux with `uv`:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev,microphone]'
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

Generate the synthetic wake bootstrap (macOS only), train both small models, and evaluate:

```sh
python -m me2.wake_word.data --generate
python -m me2.wake_word.train --seed 42 --epochs 30
python -m me2.vcm.train --seed 42 --epochs 30
```

Wake training writes epoch logs, selects its threshold on its voice-disjoint validation group, then scores the held-out synthetic test voices. VCM training writes epoch logs and scores the official speaker-disjoint test split. Explicit evaluation commands are:

```sh
python -m me2.wake_word.evaluate --split test
python -m me2.vcm.evaluate --split test
python -m me2.vcm.evaluate --split holdout --output results/vcm_holdout
```

For a small data-path/training smoke run after the HF dataset has been cached:

```sh
python -m me2.vcm.train --epochs 1 --smoke-limit 128 --skip-test-evaluation --output vcm/models/vcm_model.pkl
```

This intentionally trains on only a small subset and is **not** the final VCM model. Re-run the full training command before reporting final performance. The holdout command is optional and should be used as a one-time deployment check rather than for tuning.

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

Expected state transitions are `[LISTENING] Waiting for wake word...`, `[WAKE WORD] "ateko" detected`, `[ASSISTANT] Yes?`, `[LISTENING] Waiting for command...`, the command and confidence, then `[STATUS] Returning to wake-word detection...`. Runtime logs go to the console and `logs/app.log`. Individual models must be trained before inference; only their canonical bundle paths are used by the CLI.

## Raspberry Pi 5

The network is a small depthwise-separable CNN over 40-bin log-Mel features, with no ASR runtime. Keep inference on CPU and cap thread use if CPU contention is observed (for example `OMP_NUM_THREADS=2`). The evaluation report measures single-example CPU latency on the training/evaluation host; it is not a Pi benchmark. Re-run the same inference benchmark on the target Pi 5 before choosing window/latency settings. The app uses fixed 1.5-second wake windows and a 5-second command capture; this starter implementation is windowed rather than a validated streaming/VAD system. Microphone access and PortAudio are optional; WAV-file inference remains available without them.

## Limits and interpretation

No training or evaluation has been run by this documentation alone. Do not describe expected or synthetic-only performance as observed results. In particular, wake-word generalization requires real-speaker and noisy-room testing. VCM performance must be taken from the generated `results/vcm_test/metrics.json` and accompanying error-analysis files after the full command completes. The test and holdout splits are not for hyperparameter tuning.
