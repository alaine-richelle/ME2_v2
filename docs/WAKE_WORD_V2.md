# Wake-word v3: connected “a-teh-ko”

This document separates historical evidence, newly reproduced results,
assumptions, and unverified deployment requirements. One connected audition
clip was explicitly approved by the user; no other clip is claimed as
listened to or approved.

The current machine-readable and experimental summary is
`docs/EXPERIMENTS_2026-10-05.md`. Sections describing v2 below are retained as
historical engineering evidence.

## Revisions and environment

- Integration checkout inspected and used:
  `52b38f7fdaf35a0f5b5a8dbbd239f4bfb7700e24`.
- Command CTC reference retained for the separate command stage:
  `f5d2dc585b49b46e8518ec6122261a9ca7f2f5e4`.
- Required command-dataset revision retained by the CTC pipeline:
  `bfca0bbebfe17e00d6a685d1da573615d29ecdce`.
- Wake experiments ran on macOS 26.6.2 ARM64, Python 3.14.3, PyTorch
  2.14.1, CPU. These are not Raspberry Pi measurements.

## Pronunciation contract

The application label remains `ateko`. Dataset version
`wake-v3-connected-a-teh-ko` stores the separate guide `a-teh-ko`: three connected
syllables in that order, with an eh-like middle vowel. No exact IPA or stress
pattern is asserted because the user approved an audio example, not a phonetic
transcription.

On 2026-10-05, the user confirmed the audition clip synthesized from
`a teh ko` with Samantha at rate 175 as correct. Future synthetic positives
therefore use only the connected text `a teh ko`, versioned as
`connected-a-teh-ko-v1`. Hyphenated text remains a written pronunciation guide
and audition comparison, not an approved synthesis recipe.

The historical five-class label `te` is retained for compatibility but now
means the intended spoken cue “teh”; the mapping is versioned as
`wake-five-class-te-means-teh-v2`. Only an approved complete `ateko` example
is a positive event. Isolated syllables, incomplete/reordered phrases, wrong
middle-vowel recipes, near misses, and commands without the wake word are
non-wake examples.

## Historical corpus audit

At the reviewed revision, `src/me2/wake_word/data.py:24` synthesized `te`,
`a te ko`, and unqualified `ateko`; `record.py:17-22` prompted `te` and
`a-te-ko`. Renaming those labels would not alter their waveforms.

The preserved v1 manifest contains 936 rows from 12 synthetic voices and no
real speakers: 144 complete-word positives, 72 examples for each syllable,
and 576 background rows. Under the v2 adapter, all 792 speech clips are
pronunciation-unreviewed; the 144 generated non-speech noise clips are
technically accepted. There are 66 within-split exact-duplicate groups and no
cross-split exact duplicates. No historical positive was automatically
approved or relabeled as corrected audio.

The checked-in historical test report records 67.95% five-class accuracy,
0.5476 macro F1, 15/36 accepted wakes, and 8/198 false accepts at threshold
0.764945. It is contextual evidence only because the pronunciation recipes,
partitioning, and review status differ.

## New corpus and review state

The audition set has 12 clips: Samantha, Daniel, Karen, and Moira each speak
three recipes at rate 175—`a teh ko`, `a-teh-ko`, and historical `ateko`.
The reviewed Samantha connected clip is accepted; the remaining 11 clips are
pending. Review coverage is 1/12 and approved-positive coverage is 1/12.

The current v3 synthetic corpus has 408 rows from eight voice identities:
255 train, 51 validation, 51 calibration, and 51 test. Speaker/voice and parent
lineage are disjoint across partitions. It includes 24 complete connected-word
positives and does not generate hyphenated or unqualified text as a positive
recipe. All 312 synthetic speech clips remain pending review; 96 generated
noise clips are technically accepted. There are zero real speakers and zero
approved main-corpus positives. The audit found no exact or near-duplicate
groups, split leakage, missing files, checksum mismatch, or sample-rate
mismatch.

Consequently, the corpus is structurally valid but `valid_for_training=false`
for approved/deployment training. `--allow-pending-review` exists only for
explicit engineering experiments; artifacts produced that way carry
`deployment_ready=false` and live microphone use refuses them without an
explicit override.

## Implemented engineering changes

- `src/me2/wake_word/data.py:27-112` defines the versioned pronunciation,
  class mapping, manifest schema, audition recipes, and realistic negatives.
- `data.py:119-143` creates stable IDs and SHA-256 hashes; recorder paths are
  collision-safe and refuse overwrites.
- `data.py:334-355` validates labels, review states, and frozen split names.
- `data.py:358-499` synthesizes explicit “teh” recipes, records engine/voice/
  rate/settings, and leaves speech pending review.
- `data.py:645-765` audits exact/near duplicates, speaker and parent leakage,
  audio integrity, review coverage, and approved-positive eligibility.
- `src/me2/wake_word/record.py:22-40` gives natural complete-word, isolated
  syllable, incomplete/reordered, command-only, and silence prompts.
- `record.py:95-147` uses pseudonymous speaker/session/device/room metadata,
  UUID filenames, incremental manifests, and no automatic approval.
- `src/me2/wake_word/train.py:49-83` refuses to crop complete wake examples
  longer than the model window.
- `train.py:167-198` rejects speed/shift settings that could move or stretch a
  complete keyword outside the wake window.
- `train.py:123-160` performs label-balanced, speaker/source/kind-aware
  sampling; training-only controlled augmentation follows partitioning.
- `train.py:200-244` enforces audit, review, version, partition, and class
  coverage requirements.
- `train.py:259-367` seeds before construction, tracks clean train and
  validation curves, restores the best checkpoint, supports five-class and
  binary candidates, and embeds review/readiness provenance.
- `src/me2/wake_word/calibrate.py` is the only threshold-mutating path and uses
  the calibration partition. Evaluation remains read-only.
- `src/me2/wake_word/evaluate.py:119-275` keeps rejected wakes in denominators,
  reports Wilson intervals and provenance, and does not mislabel clip false
  accepts as false activations per hour.
- `src/me2/wake_word/detector.py` provides deterministic N-of-M smoothing with
  reset after a trigger.
- `src/me2/app.py` uses the versioned smoothing policy in the continuous
  wake-to-command state machine and blocks explicitly unverified v2 artifacts
  from live use by default.

The CTC/trie command stage remains separate. The handoff uses one continuous
stream with overlapping wake windows, pre-roll, endpointing, bounded command
capture, rejection, cooldown, overflow invalidation, and recovery.

## Provisional matched experiments

All B–E rows use the same unreviewed v2 test partition: one synthetic voice,
54 clips, 6 positives and 48 negatives. Thresholds were selected only on the
separate calibration voice. These results test code and recipes, not correct
human pronunciation or deployment generalization.

| ID | Candidate | Params | Macro F1 | Wake recall | False accepts |
|---|---|---:|---:|---:|---:|
| A | Historical 1.5 s model on v2 | 3,293 | not run | not run | not run |
| B | Corrected five-class, no augmentation/sampler | 3,293 | 0.1677 | 0/6 | 0/48 |
| C | Five-class, source-aware sampling + augmentation | 3,293 | 0.3140 | 0/6 | 0/48 |
| D | C + dropout 0.1 and weight decay 1e-4 | 3,293 | 0.4024 | 0/6 | 3/48 |
| E | Binary, source-aware + regularized | 2,858 | 0.5857 | 2/6 | 4/48 |

A was deliberately not forced: its 1.5-second input would crop a 1.557-second
complete-word v2 clip. B underfit to background. Sampling/augmentation
improved ranking and macro F1, and E is the best provisional direction, but
its 33.3% wake recall and 8.3% clip false-accept rate are unacceptable. No v2
candidate was promoted. Host p95 single-example model inference was
approximately 0.76–2.05 ms, excluding streaming buffering; process RSS was
process-wide and is not a model footprint or Pi result.

A separate 200-iteration host benchmark of E on one pending audition clip
measured 0.171/0.204 ms frontend p50/p95, 0.438/0.539 ms CNN p50/p95, and
0.607/0.724 ms complete p50/p95 with two PyTorch threads. RSS after the run was
287,440,896 bytes process-wide. File decoding, wake-window buffering,
endpointing, and cooldown were outside that timed loop.

The file-based compatibility replay also passed: the provisional wake
candidate opened the gate and the existing CTC/trie command model decoded
`vcm/data/test_sample.wav` as `play music` / `play_music`. The wake clip is
still pending pronunciation review, so this proves only that the handoff
interfaces remain compatible.

Full machine-readable results are in
`results/wake_v2_ablation_2026-10-05.json` and the per-candidate result
directories.

## Reproducible workflow

From the repository root:

```sh
export PYTHONPATH=src

# Generate a small audition set first (macOS).
python -m me2.wake_word.data preview \
  --output wake-word/data/v2/audition

# Inspect integrity and review state.
python -m me2.wake_word.data audit \
  wake-word/data/v2/audition/manifest.csv \
  --output results/wake_v2_audition_audit.json

# Fill review_template.csv with accepted/incorrect/ambiguous/technical_invalid,
# then apply it. Do not approve clips that have not actually been auditioned.
python -m me2.wake_word.data apply-review \
  wake-word/data/v2/audition/manifest.csv \
  wake-word/data/v2/audition/review_template.csv

# Record real sessions. Reuse the pseudonymous speaker ID across sessions.
python -m me2.wake_word.record \
  --speaker speaker-001 \
  --session session-01 \
  --device "USB microphone" \
  --room quiet-bedroom \
  --take-count 22

# After review, freeze speaker assignments once.
python -m me2.wake_word.data freeze-splits \
  wake-word/data/v2/recordings/manifest.csv \
  wake-word/data/v2/recordings/splits.json \
  --seed 42

# Approved five-class candidate.
python -m me2.wake_word.train \
  --manifest wake-word/data/v2/recordings/manifest.csv \
  --label-mode five_class \
  --run-name approved-five-class \
  --epochs 30 \
  --output wake-word/models/wake_word_model_v2_five_class.me2

# Approved binary comparison.
python -m me2.wake_word.train \
  --manifest wake-word/data/v2/recordings/manifest.csv \
  --label-mode binary \
  --run-name approved-binary \
  --epochs 30 \
  --output wake-word/models/wake_word_model_v2_binary.me2

# Explicit recalibration, then read-only test evaluation.
python -m me2.wake_word.calibrate \
  --model wake-word/models/wake_word_model_v2_binary.me2 \
  --manifest wake-word/data/v2/recordings/manifest.csv \
  --split calibration \
  --output results/wake_v2_binary_calibration.json
python -m me2.wake_word.evaluate \
  --model wake-word/models/wake_word_model_v2_binary.me2 \
  --manifest wake-word/data/v2/recordings/manifest.csv \
  --split test \
  --output results/wake_v2_binary_test

# File-based wake check and wake-to-CTC-command handoff.
python -m me2.app --wake-word \
  --wake-model wake-word/models/wake_word_model_v2_binary.me2 \
  --audio path/to/approved-a-teh-ko.wav
python -m me2.app --interactive \
  --wake-model wake-word/models/wake_word_model_v2_binary.me2 \
  --wake-audio path/to/approved-a-teh-ko.wav \
  --command-audio path/to/command.wav

# Live use is accepted only when the bundle says deployment_ready=true.
python -m me2.app --interactive \
  --wake-model wake-word/models/wake_word_model_v2_binary.me2

# Host or Pi wake-path benchmark. Run the same command on the target Pi.
python -m me2.benchmark path/to/approved-a-teh-ko.wav \
  --model wake-word/models/wake_word_model_v2_binary.me2 \
  --iterations 200 \
  --threads 2 \
  --output results/pi5_wake_benchmark.json
```

## Remaining evidence required

1. Audition the remaining connected clips across the other synthetic voices
   and rates. The Samantha/rate-175 connected clip is the current approved
   anchor; approval does not automatically transfer to every voice.
2. Record and review independent real speakers. A useful pilot is roughly
   20–30 speakers with 10–20 complete-word takes each, preferably across more
   than one session, plus comparable negatives. This is a planning estimate,
   not a sufficiency guarantee.
3. Include intended accents, rates, distances, rooms, background conditions,
   and the actual Pi microphone. Keep all derivatives from one speaker/parent
   in one partition.
4. Record substantial continuous negative audio and timestamp wake events.
   Until then, false activations/hour, event delay, and duplicate-trigger rate
   remain unmeasured.
5. Freeze model, smoothing, threshold, and cooldown before the final unseen
   real-speaker/continuous-audio test.
6. Benchmark buffering, frontend, CNN, event logic, CPU, RSS, p50/p95 latency,
   temperature, and throttling on the actual Raspberry Pi 5. No Pi performance
   claim is made here.
