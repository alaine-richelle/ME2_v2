# ME2_v2 Project Map

Use this reference to choose where to inspect or edit. Confirm current details
with targeted searches because active experiments may add files.

## Repository Control Files

| Path | Responsibility |
| --- | --- |
| `README.md` | Architecture, dataset policy, reproducible commands, measured results, deployment limits |
| `pyproject.toml` | Package metadata, dependencies, console entry points, pytest and Ruff settings |
| `config/default.json` | Shared audio, training, augmentation, VCM, and wake-word defaults |
| `requirements.txt` | Alternate dependency list |
| `.gitignore` | Tracked versus local datasets, caches, checkpoints, and generated outputs |

The package uses a `src/` layout. Tests already set `pythonpath = ["src"]`;
manual module commands should run from the repository root with
`PYTHONPATH=src` unless the package is installed editable.

## Shared Implementation

| Path | Responsibility and main consumers |
| --- | --- |
| `src/me2/audio.py` | Mono conversion, resampling, trim/normalize, fixed-duration preparation, WAV I/O |
| `src/me2/features.py` | Log-Mel feature configuration and extraction |
| `src/me2/model.py` | Compact `TinyAudioCNN` classifier |
| `src/me2/training.py` | Seeding, augmentation integration, classifier fitting, validation |
| `src/me2/bundle.py` | Safe `.me2` model save/load, trusted legacy migration, architecture-aware inference |
| `src/me2/metrics.py` | Classification metrics, profiling, reports, CSV/JSONL output |
| `src/me2/logging_utils.py` | Console and file logging setup |
| `src/me2/app.py` | WAV/microphone inference and wake-gated interactive state flow |
| `src/me2/streaming.py` | Continuous wake/capture/decode/cooldown state machine and microphone recovery |
| `src/me2/benchmark.py` | Host/Pi frontend, model, decoder, CPU, and RSS benchmark entry point |

Changes to audio duration, sample rate, feature shape, labels, architecture, or
bundle metadata can affect both tasks. Trace their callers and saved-bundle
compatibility before editing.

## Wake-Word Workflow

```text
wake_word/data.py or wake_word/record.py
    -> wake_word/train.py
    -> wake_word/calibrate.py
    -> bundle.py + wake-word/models/*.me2
    -> wake_word/evaluate.py, benchmark.py, or app.py
    -> results/wake_v2_*
```

| Path | Responsibility |
| --- | --- |
| `src/me2/wake_word/data.py` | Pronunciation-aware audition/synthesis, manifests, review, frozen splits, duplicate/leakage audit |
| `src/me2/wake_word/record.py` | Collision-safe guided real-speaker recording for “a-teh-ko” |
| `src/me2/wake_word/train.py` | Fixed partitions, five-class/binary candidates, source-aware sampling, training-only augmentation |
| `src/me2/wake_word/calibrate.py` | Explicit threshold mutation on calibration only |
| `src/me2/wake_word/evaluate.py` | Read-only complete-wake metrics, uncertainty, and provenance |
| `src/me2/wake_word/detector.py` | Shared deterministic N-of-M streaming smoothing |
| `wake-word/data/` | Raw/generated local WAVs and manifests; potentially large and generally untracked |
| `wake-word/models/*.me2` | Safe historical and provisional wake artifacts |

The complete `ateko` label is the only accepted wake event. Its pronunciation
guide is `a-teh-ko`; versioned class `te` means the intended “teh” cue.
Isolated syllables count as negatives for event detection. Provisional
pending-review artifacts are marked `deployment_ready=false`.

## Voice-Command Workflow

```text
vcm/data.py (official Hugging Face splits)
    -> vcm/train.py
       -> default: TinyAudioCNN + training.py
       -> CTC: CompactCTCModel + ctc_training.py
    -> bundle.py + vcm/models/*.me2
    -> vcm/evaluate.py or app.py
    -> results/vcm_test or results/vcm_ctc_test
```

| Path | Responsibility |
| --- | --- |
| `src/me2/vcm/data.py` | Pinned HF loading/decoding, transcript metadata, persisted speaker partitions |
| `src/me2/vcm/audit.py` | Character, grammar, split, provenance, and CTC feasibility audit |
| `src/me2/vcm/train.py` | CLI dispatch for default classifier and CTC architecture |
| `src/me2/vcm/evaluate.py` | Architecture-aware official split evaluation |
| `src/me2/vcm/ctc.py` | Character contract, compact model, semantic trie, exact prefix beam search, rejection result |
| `src/me2/vcm/ctc_training.py` | Variable-length CTC dataset/collation, feasibility checks, fitting, calibration |
| `vcm/data/` | Local Hugging Face-derived cache/data |
| `vcm/models/vcm_baseline_model.me2` | Reproducible intent-classifier baseline |
| `vcm/models/vcm_ctc_model.me2` | Character CTC/trie artifact |

The default path predicts 19 normalized intents plus `out_of_scope`. The CTC
path predicts characters and decodes against a predefined/approved grammar
without harvesting held-out transcripts. Do not merge baseline, provisional,
smoke, and selected artifact paths.

## Tests and Change Impact

| Path | Coverage focus |
| --- | --- |
| `tests/test_audio_features.py` | Audio preparation, Log-Mel/CNN shape, augmentation determinism, speaker split, bundles, CLI modes |
| `tests/test_ctc_vcm.py` | CTC feasibility, brute-force decoder probabilities, trie semantics, safe bundles |
| `tests/test_streaming.py` | Wake/capture/decode/cooldown, pre-roll, timeout, overflow recovery |
| `tests/test_wake_v2.py` | Pronunciation versions, review gating, leakage, lineage, complete-word safety, calibration separation |

Common impact routes:

| Request | Begin with | Then inspect |
| --- | --- | --- |
| Audio preprocessing or feature bug | `audio.py`, `features.py` | both task trainers/evaluators, `test_audio_features.py`, config |
| New or changed CLI behavior | `app.py`, `pyproject.toml` | bundle format, README inference commands, CLI tests |
| Wake false accepts/rejects | wake train/evaluate | labels, threshold metadata, manifests, saved wake metrics |
| VCM intent data or split issue | `vcm/data.py` | train/evaluate, dataset card assumptions in README |
| Default classifier architecture | `model.py` | training, bundles, both tasks, model profiling tests |
| CTC decoding/training | `vcm/ctc.py`, `vcm/ctc_training.py` | train/evaluate dispatch, CTC bundle test, CTC results |
| Metrics/reporting | `metrics.py` | both evaluators and generated result schemas |
| Model loading failure | `bundle.py` | architecture metadata, producer train module, app/evaluator consumer |

## Generated and Heavy Paths

- `.venv/` is about 1 GB and should not be recursively searched.
- `*.pkl` model bundles are binary; inspect them through `load_bundle`, not text
  tools.
- `wake-word/data/` and `vcm/data/` may contain large local datasets.
- `results/` and `logs/` are experiment evidence, not implementation source.
- `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`, and
  `src/me2_v2.egg-info/` are generated metadata/caches.

Before reporting performance, read the matching `metrics.json` and, when error
patterns matter, `report.txt` or per-item prediction files. Keep official test
and holdout data out of tuning decisions.
