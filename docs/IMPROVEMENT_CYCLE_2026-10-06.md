# ME2 measured improvement cycle — 2026-10-06

## Scope and fixed provenance

- Integration checkout: `52b38f7fdaf35a0f5b5a8dbbd239f4bfb7700e24`.
- CTC reference: `f5d2dc585b49b46e8518ec6122261a9ca7f2f5e4`.
- Command dataset: `bfca0bbebfe17e00d6a685d1da573615d29ecdce`.
- Host: macOS ARM64, Python 3.14.3, PyTorch 2.14.1, CPU.
- No Raspberry Pi 5 or new human wake recordings were available.

Historical reports are preserved in `docs/EXPERIMENTS_2026-10-05.md`.
Everything below was newly executed in this cycle. The official command test
and synthetic wake test had already been inspected in prior development, so
these are reused development assessments, not fresh final estimates.

## Promotion guardrails

Wake candidates had to improve fixed-test recall over 0/3 while keeping clip
false accepts at or below 2/48. This is only a provisional synthetic guardrail;
deployment additionally requires real-speaker event recall and false
activations per hour.

Command candidates had to avoid more than a 0.005 absolute macro-F1 loss,
reduce wrong accepted actions, keep OOS false accepts no worse than 20/76, and
avoid more than a 0.02 absolute acceptance-coverage loss.

## Wake data changes

Wake v4 preserves all 153 validation/calibration/test rows and hashes from v3.
It adds 98 train-only examples: connected `a teh ko` positives at four new
rates for existing training voices, plus matched positives/negatives from Fred
and Ralph. The corpus has 506 rows, ten TTS voices, no real speakers, no
cross-split duplicates, and no leakage.

Wake v5 adds 320 transcript-screened command-without-wake clips from the
pinned Hugging Face official training split, capped at two clips per source
speaker across 185 source speakers. These rows are train-only and remain
`pending` because they were not listened to for incidental wake occurrences.
The v5 corpus has 826 rows and passes structural/leakage checks.

All positive generation remained a single whole-utterance synthesis request
with exact text `a teh ko`. No isolated-syllable stitching was introduced.

## Wake experiments

All models use `TinyAudioCNN`; thresholds use the distinct calibration voice.
The calibration tie-break was fixed to choose the lowest threshold among
identical recall/FAR operating points.

| Candidate | Params | Validation wake recall / FA | Calibration recall / FA | Reused test recall / FA | Decision |
|---|---:|---:|---:|---:|---|
| v3 seed-43 incumbent | 2,858 | 2/3, 2/48 | recorded in retained artifact | 0/3, 0/48 | Retain incumbent only as previous preliminary selection |
| v4 seed 42 | 2,858 | 0/3, 0/48 | 1/3, 0/48 | Not run | Reject |
| v4 seed 43 | 2,858 | 0/3, 0/48 | 0/3, 0/48 | Not run | Reject |
| v4 seed 44 | 2,858 | 0/3, 0/48 | 0/3, 0/48 | Not run | Reject |
| v5 base seed 42 | 2,858 | 0/3, 0/48 | 1/3, 2/48 | Not run | Reject |
| v5 width-16 regularized seed 43 | 4,578 | 0/3, 0/48 | 0/3, 0/48 | Not run | Reject |
| v5 width-12, dropout 0.20, weight decay 5e-4, seed 43 | 2,858 | 2/3, 0/48 | 2/3, 2/48 | 2/3, 4/48 | Do not promote: false-accept guardrail failed |

The best new candidate trained for 26 epochs, restored epoch 20, and took
41.02 seconds on the host. Its host file-inference benchmark measured
0.369/0.387 ms p50/p95 over 100 runs with one PyTorch thread. This excludes
stream buffering/endpointing and is not a Pi benchmark.

The fresh candidate improves the reused synthetic test voice from 0/3 to 2/3
wake recall, but the 4/48 clip false accepts are unsafe. The sample is also
far too small for a reliable rate estimate. No wake artifact is deployment
ready.

## Command rejection experiment

The corrected 3,652,572-parameter legacy CTC model was kept fixed. A new
calibrator decoded 1,454 usable rows from the persisted speaker-disjoint
official-train calibration partition and selected a normalized log-score
threshold of `-0.4263739437`. Selection did not use official test/holdout.

| Policy | Accuracy | Macro F1 | OOS false accepts | Wrong accepted commands | In-scope rejects | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Existing corrected CTC, uncalibrated | 0.8911 | 0.8682 | 20/76 | 148 | 316/4,367 | 0.9163 |
| Calibration-only score threshold | 0.8397 | 0.8414 | 7/76 | 50 | 655/4,367 | 0.8370 |

The policy cut wrong accepted commands by 98 and OOS false accepts by 13, but
rejected 339 additional in-scope requests. It reduced
`light_off → light_on` from 17 to 13, `color → light_on` from 15 to 4,
`brightness → color` from 10 to 5, `stop → play_music` from 7 to 6, and
`volume_down → volume_up` from 5 to 2. The macro-F1 and coverage guardrails
failed, so the calibrated artifact is retained for analysis but not promoted.

The uncalibrated corrected CTC artifact remains the selected command model.
The result confirms that rejection can suppress dangerous confusions, but a
single global threshold is too blunt; class/phrase-aware calibration or a
trained OOS head needs new development data.

## Reliability and integration

- Safe bundle writes now remove temporary archives after injected atomic
  replace failures.
- Runtime bundle prediction caches the frontend object; filter/window tensors
  were already device/config cached.
- Wake manifests now retain external dataset ID, revision, source record, and
  terms notes.
- Wake synthesis now verifies non-empty audio even when the host synthesizer
  reports success.
- A file replay using the v5 experimental wake model and selected corrected
  CTC model detected the approved connected anchor at 0.9391 and accepted
  `vcm/data/test_sample.wav` as `play_music`.
- Decoder exceptions now become explicit non-actionable
  `command_decoder_error` results and return through cooldown.
- Microphone retry exhaustion is fault-injection tested: three bounded retries,
  stream cleanup, stopped state, root error retained.
- Full lint and tests after the reporting/recovery additions: 66 passed.

## Raspberry Pi evaluation report

The pinned HF Arrow shards were inspected directly without modifying the
read-only cache. Official test and holdout each contain exactly 19 intents and
93 unique non-empty variation strings. Train contains five additional empty
variation labels; those are excluded from the versioned command identity
schema at `config/command_schema_v1.json`.

`me2-pi-report` now scores a fixed trial manifest, ordered structured Pi event
log, and telemetry. It preserves `REJECT`, `NO_RESPONSE`, unknown names,
variation identity, slots, first-decision semantics, extra/premature/late
fires, Wilson intervals, speaker-grouped bootstrap intervals when possible,
per-class F2, conditional slot errors, and Pi-only telemetry. Input hashes and
paths are saved with every run.

No Pi, held-out trial manifest, Pi event log, or Pi telemetry was available.
The authoritative artifact
`reports/pi-evaluation-pending-2026-10-06/report.md` therefore has the required
six sections but is marked `not_run`, with explicit `N/A — reason` fields. Host
benchmarks were not substituted.

## Reproduce

```sh
export PYTHONPATH=src

python -m me2.wake_word.data expand-synthetic \
  wake-word/data/v3/synthetic/manifest.csv \
  --output wake-word/data/v4/synthetic-expanded

HF_DATASETS_OFFLINE=1 python -m me2.wake_word.data add-command-negatives \
  wake-word/data/v4/synthetic-expanded/manifest.csv \
  --output wake-word/data/v5/command-negatives \
  --maximum-rows 320 --maximum-per-speaker 2

python -m me2.wake_word.train \
  --manifest wake-word/data/v5/command-negatives/manifest.csv \
  --label-mode binary --allow-pending-review \
  --epochs 30 --width 12 --dropout 0.20 --weight-decay 0.0005 \
  --seed 43 --run-name v5-regularized-seed43 \
  --output wake-word/models/wake_word_v5_regularized_seed43.me2

HF_DATASETS_OFFLINE=1 python -m me2.vcm.calibrate \
  --model vcm/models/vcm_ctc_model.me2 \
  --output-model vcm/models/vcm_ctc_calibrated_2026-10-05.me2 \
  --partitions results/vcm_ctc_partitions.json \
  --output results/vcm_ctc_calibration_2026-10-05

python -m me2.vcm.analyze \
  --predictions results/reproduced_legacy_ctc_calibrated_test_2026-10-05/predictions.jsonl \
  --model vcm/models/vcm_ctc_calibrated_2026-10-05.me2 \
  --output results/analysis_legacy_ctc_calibrated_test_2026-10-05

python -m me2.pi_report \
  --metadata pi-run/run_metadata.json \
  --schema config/command_schema_v1.json \
  --manifest pi-run/ground_truth.csv \
  --events pi-run/events.jsonl \
  --telemetry pi-run/telemetry.csv \
  --output reports/<run_id>

ruff check src tests
pytest -q
```

## Remaining evidence

Before deployment, collect and review independent connected `a-teh-ko`
recordings: approximately 20–30 speakers with 10–20 complete-word takes each
is a pilot target, not a sufficiency guarantee. Include multiple sessions,
actual Pi microphones, intended accents, distances, rooms, and substantial
continuous negative audio. Freeze unseen real speakers before model changes.

Measure event recall, false activations/hour, delay, duplicate triggers,
end-to-end successful interactions, sustained Pi CPU/RSS/temperature, and
throttling. The 320 imported command negatives need waveform review or
replacement with a separately held-out, speaker-partitioned continuous
negative benchmark. For commands, train a compact CTC model to convergence
with explicit slot grammar and OOS supervision; do not promote the preliminary
compact model or the aggressive global-threshold artifact.
