# Character CTC command pipeline

This document separates reviewed historical evidence, results reproduced in
this checkout, assumptions, and work that still requires training or Raspberry
Pi hardware.

## Revisions and environment

- Integration checkout reviewed and used: `52b38f7fdaf35a0f5b5a8dbbd239f4bfb7700e24`.
- CTC reference reviewed from a detached local clone:
  `f5d2dc585b49b46e8518ec6122261a9ca7f2f5e4`.
- Hugging Face dataset revision requested by the loader:
  `bfca0bbebfe17e00d6a685d1da573615d29ecdce`.
- Dataset/configuration: `airimonda/ai231-me2-voice-commands`, default
  `train`, `test`, and `holdout` splits.
- Reproduction host: macOS ARM64, Python 3.14.3, PyTorch 2.14.1, CPU
  inference. This is not Raspberry Pi evidence.

The audit command records these values in
`results/vcm_ctc_data_audit.json`. Development has already inspected the
official test reports, so a future final model should be frozen on
train/validation/calibration and assessed on a fresh, untouched evaluation set
if an unbiased final estimate is required.

## Baseline and historical reports

The checked-in classifier report at `results/vcm_test/metrics.json` is a
historical artifact. It reports an 8,052-parameter 19-intent-plus-OOS model,
53.50% accuracy, 0.4062 macro F1, and 56 false accepts among 76 OOS clips.

The pinned CTC reference report `metrics_v6.json` is also historical. It
reports a 3,652,572-parameter model, 86.60% intent accuracy, and 74.08% exact
transcript accuracy on its Option B test set. Its data preparation, split,
capacity, grammar, and rejection policy differ from ME2_v2, so those numbers
are not a controlled before/after comparison and are not Pi validation.

## Reference audit and implemented corrections

Line references below refer to the pinned CTC reference revision.

- `decoder_v6.py:112-131` and `decoder_v6_fast.py:103-141` omitted the CTC
  repeated-symbol self-loop, merged equivalent states with `max`, and retained
  an early terminal. `me2.vcm.ctc.prefix_beam_search` now keeps separate
  blank/nonblank probability mass, merges with log-sum-exp, handles repeated
  labels correctly, and selects terminals only after every usable frame.
- `decoder_v6_fast.py:204-208` forwarded the five-value `build_flat` result to
  a mismatched wrapper. The new implementation has one public decoder entry
  point and no untested wrapper.
- `train_v6.py:42-44` and `train_v6.py:222-224` clipped targets/IDs while the
  model exposed only 32 output frames. Training now computes real convolution
  output lengths and enforces `T >= U + R`, where `R` is the count of adjacent
  repeated target characters. Impossible examples fail with an explicit
  reason; `zero_infinity=False`.
- `build_features.py:31` imported a missing `grammar` module;
  `build_features.py:68-70` retained only the first 128 frames;
  `build_features.py:121-124` truncated targets; and
  `build_features.py:129-134` could index the original manifest with a
  filtered-row index. ME2_v2 now builds targets directly from the retained
  row, uses complete bounded audio, right-padding plus usable lengths, and a
  shared frontend.
- `grammar_v4.py:28-40` removed digits. The versioned normalizer preserves
  `a-z`, `0-9`, and space, while refusing empty or unsupported targets.
- `train_v6.py:226-229` silently kept the first meaning for a normalized
  phrase. Grammar validation now fails on conflicting intent/variation/slot
  semantics.
- The reference grammar was assembled from all feature splits. The new grammar
  contains only predefined/approved aliases from the internal training
  partition; evaluation transcripts and frequencies are not harvested.
- `evaluate_v6.py:97-102` mapped failed decodes to class zero and exposed an
  exponentiated path score as confidence. Failed and rejected examples now
  remain explicit in prediction records and metric denominators. Acceptance
  uses calibration-frozen score/margin/candidate thresholds; scores are
  described as decoder scores, not correctness probabilities.
- `inference.py:40-49` duplicated preprocessing/model constants and used
  blocking capture. Model, frontend, normalization, decoder, artifact, and
  state-machine contracts are centralized under `src/me2`; microphone mode
  uses one bounded continuous stream.

Integration fixes include module-relative resource roots, seeding before model
construction, cached frontend tensors, safe artifact loading, persisted wake
splits, non-mutating evaluation thresholds, explicit microphone recovery, and
atomic artifact writes.

## Data contract and observed audit

The pinned release contains 10,733 train, 4,443 test, and 202 holdout rows.
Observed speaker counts are 308, 115, and 5. Internal seed-42 speaker partitions
from official train contain:

| Partition | Rows | Speakers | Use |
|---|---:|---:|---|
| Train | 6,805 | 229 | Gradient training and approved grammar aliases |
| Validation | 2,455 | 47 | Model/decoder selection |
| Calibration | 1,462 | 31 | Rejection thresholds only |
| Quarantine | 11 | 1 missing identity | Invalid nonspeech rows |

The audit observed 3,932 digit-bearing train transcripts, 3,854 train
transcripts with adjacent repeated normalized characters, no conflicting
normalized semantic mappings, 571 approved training terminals, and no
declared-duration CTC alignment failures with the compact model's `/2`
temporal stride. Twelve official-train rows have empty normalized transcripts;
the eleven missing-speaker nonspeech rows are quarantined. Empty targets are
not silently treated as ordinary OOS speech.

CTC acoustic targets are the documented normalized words actually associated
with the recording. Reliable OOS transcripts may train character acoustics,
but OOS rows never become actionable grammar terminals. Synthetic tags such as
`<babble>` normalize to a lexical token only when that token is genuinely the
documented target; otherwise the row must be quarantined or handled as a
separate negative policy. Digits are retained verbatim because no unverified
spoken-number expansion is applied.

Supplemental configurations are not loaded by the default pipeline. If enabled
in a future controlled experiment, they must follow their documented
voice/source partition rules; `tune_oos` is calibration-only, and numerals
require separately verified speaker separation. The command dataset is not a
verified source of `ateko` positives; wake data remains separate.

## Final implemented architecture and contracts

Command processing is:

`audio → mono/resample/trim/right-pad → 40-bin log-power Mel → compact
depthwise-separable CNN → optional packed one-layer BiGRU → per-frame
character logits → CTC trie prefix beam → calibrated acceptance → explicit
intent/variation/supported slot`.

The deployment candidate has 203,638 parameters at the current width and
contains a bidirectional GRU. It decodes a completed utterance and is not
described as causal streaming ASR.

Character IDs are the indices of
`abcdefghijklmnopqrstuvwxyz0123456789 `; blank is ID 37. Blank, space,
padding, OOS, and silence are distinct concepts. Feature lengths are computed
from usable samples, output lengths from actual stride geometry, recurrent
padding uses packed sequences, and the decoder consumes only valid frames.

Grammar terminals carry one canonical intent, one variation, enumerated slot
metadata, and provenance. Incomplete prefixes are non-actionable. Unsupported
values or speech with no surviving terminal reject; there is no nearest-command
fallback. Accepted results expose action semantics, while rejected candidates
remain diagnostic only.

The continuous path is:

`LISTENING_FOR_WAKE → CAPTURING_COMMAND → DECODING →
ACCEPTED_OR_REJECTED → COOLDOWN → LISTENING_FOR_WAKE`.

It uses overlapping wake windows, pre-roll, onset timeout, minimum speech,
endpoint silence, maximum duration, bounded queues, duplicate cooldown, and
recoverable stream reopening. Lost samples invalidate the partial utterance.

## Newly reproduced results

These measurements were produced on the same official HF test split and are
useful diagnostic comparisons. They are not a fresh final assessment.

| System | Parameters | Accuracy | Macro F1 | OOS false accepts | In-scope false rejects | Status |
|---|---:|---:|---:|---:|---:|---|
| Migrated baseline classifier | 8,052 | 0.5350 | 0.4062 | 56/76 | 107/4,367 | Exact baseline reproduction |
| Legacy 3.65M CTC weights + corrected trie | 3,652,572 | 0.8911 | 0.8682 | 20/76 | 316/4,367 | Newly reproduced, legacy uncalibrated policy |

The corrected legacy rerun reports 0.4873 normalized exact-transcript accuracy
and 0.9163 acceptance coverage. Supported-slot joint accuracy is 0 because the
legacy tuple grammar contains no slot metadata; it must not be presented as a
slot-capable final artifact. Twenty-three test clips longer than the configured
five-second maximum were explicitly rejected rather than cropped or omitted.

A subsequent read-only rerun added unconstrained greedy output without
changing the model, grammar, or policy. On 4,420 nonempty test transcripts,
greedy exact-transcript accuracy was 0.0873 and character error rate was
0.2214. Exact greedy strings mapped to the correct grammar semantics for
0.1517 of all rows; corrected trie candidates reached 0.8785 semantic accuracy
and final rejection-aware decisions retained 0.8911 accuracy.

Host-only model-forward timing was 0.980 ms mean for the baseline and 4.877 ms
mean for the legacy CTC artifact. These exclude real microphone buffering and
endpointing and are not Raspberry Pi measurements.

The stage benchmark on one 4.645-second WAV, 30 iterations, and two PyTorch
threads measured the 13,550,852-byte legacy artifact at 13.85 ms mean complete
inference (0.41 ms frontend, 4.79 ms model, 8.62 ms decoder/rejection; 15.76 ms
p95). The 756,222-byte compact smoke artifact measured 12.96 ms mean (0.34 ms
frontend, 3.30 ms model, 9.31 ms decoder/rejection; 13.48 ms p95). These are
macOS ARM64 diagnostics, not Pi latency or trained-model quality evidence.

An earlier local report claiming 0.9356 accuracy and 0.9095 macro F1 is retained
as historical pre-hardening evidence only. It used different decoder/data
behavior and must not replace the reproduced corrected-decoder report.

## Compact model smoke verification

The compact model completed a one-epoch, 64-example wiring run with 203,638
parameters. It produced finite loss/gradients, a safe artifact, trie metrics,
and a separate calibration result. Its observed validation accuracy and macro
F1 were both 0.0; those values are expected to be poor and are not model
quality evidence. The run's tiny calibration sample contained one OOS example,
so its threshold is not deployable. The smoke grammar contained 53 terminals;
the full corrected training partition audit contains 571.

Focused regression tests cover repeated CTC labels, blank-separated duplicate
characters, blank-only input, shared prefixes, prefix terminals, trailing
speech, brute-force alignment sums, padding exclusion, invalid logits,
deterministic behavior, grammar collisions, digit preservation, CTC
feasibility, structured rejection, safe loading, persisted splits, streaming
pre-roll/endpoints/timeouts/overlong capture, recovery, and smoke training.

## 2026-10-05 bounded training and error analysis

A balanced 2,048-row development experiment completed on the pinned Hugging
Face data. After audio quarantine it used 2,018 train, 499 validation, and 508
calibration rows. The seven-epoch attempt early-stopped at validation macro F1
0.0039 and exposed a non-finite reject-all calibration threshold. Calibration
now emits a finite, versioned reject-all policy when no eligible threshold
exists, so artifacts remain safely serializable and ordinary failure cannot
be converted to success.

The deterministic best epoch was saved as
`vcm/models/vcm_compact_ctc_preliminary_2026-10-05.me2`. Its frozen test
evaluation rejected all 4,367 in-scope commands and is explicitly a
preliminary negative result. The untracked bundle was pruned on 2026-10-15
after selection review; its report and predictions remain. Regenerate it with
the training command below before rerunning that evaluation.

`me2.vcm.analyze` now exports per-command precision/recall/F1, explicit
rejection outcomes, raw and row-normalized confusion matrices, directional
confusion rankings, representative audio IDs, slot/joint metrics, and
greedy/trie/final semantic comparisons when those fields exist. The complete
current report is `docs/EXPERIMENTS_2026-10-05.md`.

## Reproducible commands

```sh
export PYTHONPATH=src

# Pinned data/grammar/alignment audit
python -m me2.vcm.audit

# Compact CTC training, validation selection, rejection calibration, then test
python -m me2.vcm.train \
  --architecture CompactCTCModel --seed 42 --epochs 30 \
  --target-oos-far 0.05

# Explicit legacy classifier baseline
python -m me2.vcm.train \
  --architecture TinyAudioCNN --seed 42 --epochs 30 \
  --output vcm/models/vcm_baseline_model.me2

# Evaluate without changing thresholds
python -m me2.vcm.evaluate \
  --model vcm/models/vcm_ctc_model.me2 \
  --split test --output results/vcm_ctc_test_v2

python -m me2.vcm.analyze \
  --predictions results/vcm_ctc_test_v2/predictions.jsonl \
  --model vcm/models/vcm_ctc_model.me2 \
  --output results/vcm_ctc_test_v2_analysis

# Deterministic file replay through wake then command
python -m me2.app --interactive \
  --wake-audio path/to/ateko.wav \
  --command-audio path/to/command.wav

# Explicit trusted migration; original pickle remains untouched
python -m me2.bundle migrate --trusted-legacy \
  vcm/models/vcm_ctc_model.pkl vcm/models/vcm_ctc_model.me2

# Target-hardware benchmark
python -m me2.benchmark path/to/command.wav \
  --model vcm/models/vcm_ctc_model.me2 \
  --iterations 200 --threads 2 \
  --output results/pi5_vcm_benchmark.json

# Regression checks
ruff check src tests
pytest -q
```

Normal inference:

```sh
python -m me2.app --wake-word --audio path/to/ateko.wav
python -m me2.app --vcm --audio path/to/command.wav
python -m me2.app --interactive
```

## Artifact and recovery rules

`.me2` files are ZIP containers with validated JSON metadata and NumPy tensor
arrays loaded with `allow_pickle=False`. Metadata binds architecture, frontend,
normalizer, charset/blank, grammar, acceptance policy, versions, and
provenance. Weight hashes detect accidental corruption but do not authenticate
the publisher. Writes use a temporary sibling and atomic rename. Normal loading
refuses `.pkl`; trusted legacy migration is explicit and preserves the source.

Startup validates artifact members, schema, task, labels, frontend, charset,
blank index, grammar semantics, acceptance policy, and strict state-dict
compatibility. Missing/corrupt/incompatible artifacts are fatal. Ordinary
rejection returns to wake listening. Microphone faults use bounded backoff and
resource cleanup; exhausted recovery exits nonzero.

## Remaining required evidence

- Run full compact-model training with adequate compute, select only from
  validation, calibrate only on calibration/OOS data, and freeze before a
  genuinely fresh final assessment.
- Build and license a separate, representative real-speaker `ateko` corpus;
  measure streaming wake recall, false activations per hour, duplicate events,
  and combined unintended actions per hour.
- Run the benchmark on the intended Raspberry Pi 5 ARM64 OS/Python/PyTorch
  stack. Record p50/p95 stage and end-to-end latency including endpoint wait,
  dependency footprint, idle/active CPU, peak RSS, sustained temperature, and
  throttling.
- Evaluate supported slots with the new metadata-bearing grammar. The legacy
  artifact cannot establish slot accuracy.
- Evaluate float and supported target-specific quantization paths separately,
  recalibrate rejection after export, and retain a validated float fallback.
- Measure microphone overflow/disconnect behavior on the real audio device and
  verify one-utterance `ateko turn on the light` boundaries with representative
  speakers and rooms.
