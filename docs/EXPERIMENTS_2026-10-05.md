# ME2 speech experiments — 2026-10-05

This is the current evidence summary for the connected “a-teh-ko” wake
revision and character-CTC command work. Historical reports, newly reproduced
results, preliminary experiments, and unverified deployment requirements are
kept separate.

Follow-up experiments executed on 2026-10-06 are recorded separately in
`docs/IMPROVEMENT_CYCLE_2026-10-06.md`; that report supersedes promotion
decisions here while preserving this document as the earlier evidence state.

## Revisions and environment

- ME2_v2 checkout: `52b38f7fdaf35a0f5b5a8dbbd239f4bfb7700e24`.
- CTC/trie reference: `f5d2dc585b49b46e8518ec6122261a9ca7f2f5e4`.
- Hugging Face command dataset:
  `bfca0bbebfe17e00d6a685d1da573615d29ecdce`.
- Local experiments: macOS 26.6.2 ARM64, Python 3.14.3, PyTorch 2.14.1,
  CPU. No Raspberry Pi 5 was available.

## Connected wake corpus

`wake-v3-connected-a-teh-ko` makes `a teh ko` the only positive synthesis
text. Every generated positive is one whole synthesis request and one
waveform; isolated `a`, `teh`, and `ko` remain non-wake cues. Hyphenated text
is a pronunciation guide, not a TTS command.

The generated corpus contains 408 clips from eight macOS voice identities:
255 train, 51 validation, 51 calibration, and 51 test. It has 24 complete-word
positives, 72 isolated-syllable negatives, 24 command-without-wake negatives,
other speech negatives, and generated noise. Voice and parent lineages do not
cross partitions. The audit found no exact/near duplicates or corrupt files.

Pronunciation review remains incomplete. The user approved one separate
Samantha/rate-175 connected audition clip. The 24 main-corpus positive clips
and other synthetic speech remain pending per-voice acoustic review. There are
no real speakers, so all trained v3 artifacts have `deployment_ready=false`.

Corpus disposition:

| Group | Count | Disposition |
|---|---:|---|
| Historical v1 complete-word positives | 144 | Preserved but excluded from v3 training as pronunciation-unverified |
| User-approved connected audition anchors | 1 | Retained as the current acoustic reference |
| Other audition candidates | 11 | Unresolved/pending review |
| Regenerated v3 connected positives | 24 | Correct whole-word recipe; pending per-voice waveform review |
| New human rerecordings | 0 | Still required |

## Wake training results

All rows below use the same synthetic v3 partitions. Calibration uses its own
voice. Operational wake detection requires both the `ateko` class and the
calibrated threshold.

| Candidate | Parameters | Validation wake result | Test wake result |
|---|---:|---|---|
| Binary seed 42 | 2,858 | recall 0/3; FA 0/48 | not selected |
| Binary seed 43 r1 | 2,858 | recall 2/3; FA 2/48; F1 0.5714 | recall 0/3; FA 0/48; F1 0 |
| Binary seed 44 | 2,858 | recall 1/3; FA 1/48; F1 0.4000 | not selected |
| Five-class seed 42 | 3,293 | recall 3/3; FA 3/48; F1 0.6667 | recall 1/3; FA 3/48; F1 0.2857 |

The seed-43 binary model was selected before test evaluation because it met
the 5% clip false-accept target and had the strongest binary validation F1.
Its complete failure on the held-out synthetic voice demonstrates
voice-specific overfitting/domain shift. It was not promoted.

The corrected seed-43 artifact trained for 30 epochs, selected epoch 26 by
validation loss, and took 29.1 seconds on the host. The user-approved anchor
is detected at confidence 0.7802. This single in-training-voice check does not
offset the unseen-voice failure.

Host-only 200-iteration timing for the selected wake artifact measured
0.565/0.736 ms complete p50/p95, including frontend and CNN but excluding file
decoding, buffering, endpointing, and cooldown. The artifact is 15,138 bytes.
These are not Pi measurements.

## Voice-command results and misunderstood commands

The required dataset has 10,733 train, 4,443 test, and 202 holdout rows. The
following diagnostic results use the same official test split. Development
has already inspected that split, so they are not a fresh final estimate.

| System | Parameters | Accuracy | Macro F1 | OOS false accepts | In-scope rejects |
|---|---:|---:|---:|---:|---:|
| Whole-utterance classifier baseline | 8,052 | 0.5350 | 0.4062 | 56/76 | 107/4,367 |
| Legacy CTC weights + corrected trie | 3,652,572 | 0.8911 | 0.8682 | 20/76 | 316/4,367 |
| New compact CTC preliminary run | 203,638 | 0.0171 | 0.0017 | 0/76 | 4,367/4,367 |

The legacy CTC result is not a controlled capacity-matched improvement: it is
much larger than the baseline and uses legacy weights. Its corrected decoder
does, however, preserve rejection and materially changes the observed error
pattern.

On the rerun that exports all decoding stages, greedy exact-transcript
accuracy is 8.73% and greedy CER is 22.14%. Only 15.17% of greedy strings are
exact legal grammar forms with the correct semantics, while the corrected
trie candidate reaches 87.85% semantic accuracy and the final rejection-aware
decision reaches 89.11%. The final value is higher because correctly rejected
OOS rows count as correct decisions. This isolates a large grammar-decoding
effect for these fixed legacy logits, but it is not a claim that the decoder
alone explains the difference from the small baseline classifier.

Largest baseline wrong-command pairs:

- `color → play_music`: 98/423 (23.2%).
- `alarm → list_reminders`: 94/423 (22.2%).
- `alarm → temperature`: 67/423 (15.8%).
- `alarm → create_reminder`: 57/423 (13.5%).
- `light_off → next`: 54/141 (38.3%).
- `stop → play_music`: 42/141 (29.8%).
- `call → pause`: 41/141 (29.1%).

Largest corrected-CTC wrong-command pairs:

- `light_off → light_on`: 17/141 (12.1%).
- `color → light_on`: 15/423 (3.5%).
- `brightness → color`: 10/423 (2.4%).
- `stop → play_music`: 7/141 (5.0%).
- `weather → light_on`: 6/141 (4.3%).
- `light_on → light_off`: 5/141 (3.5%).
- `volume_down → volume_up`: 5/141 (3.5%).

The largest corrected-CTC rejection rates are `time` 35/141 (24.8%),
`volume_up` 28/141 (19.9%), `weather` 28/141 (19.9%), `call` 17/141
(12.1%), and `light_off` 15/141 (10.6%). Rejection is safer than an unintended
action but remains a command failure.

The legacy tuple grammar contains no slot metadata, so its measured slot joint
accuracy is 0/2,538 and must not be described as slot-capable. The baseline has
no slot output at all. The compact CTC pipeline carries explicit slot metadata,
but its preliminary weights learned no useful transcript evidence.

The compact experiment used a balanced 2,048-row development limit
(2,018 usable train, 499 validation, 508 calibration rows). A seven-epoch run
early-stopped with validation macro F1 0.0039; artifact serialization then
exposed and led to a fix for a non-finite reject-all threshold. The
deterministic best epoch was rerun and saved. Calibration selected an explicit
finite reject-all policy, and the frozen test run confirmed 100% in-scope
rejection. This is a completed preliminary negative result, not a deployment
candidate.

Machine-readable details are under:

- `results/analysis_baseline_test_2026-10-05/`
- `results/analysis_legacy_ctc_corrected_with_greedy_test_2026-10-05/`
- `results/analysis_compact_ctc_preliminary_test_2026-10-05/`

Each directory contains per-command metrics, raw and row-normalized confusion
matrices, confusion pairs ranked by count and rate, and per-error JSONL/CSV
records with audio IDs and decoder diagnostics where available.

Three cached examples were exported for listening review, without claiming
that the assistant listened to them:

- `results/representative_command_errors/000564_light_off_to_light_on.wav`:
  “Lights off” decoded as “lights on.”
- `results/representative_command_errors/003528_color_to_light_on.wav`:
  “I'd like some red lighting” decoded as “lights on.”
- `results/representative_command_errors/000846_stop_to_play_music.wav`:
  “Stop music” decoded as “start music” / `play_music`.

## Integrated behavior

The file-based handoff passed with the user-approved connected audition clip
and `vcm/data/test_sample.wav`. The v3 binary detector opened the gate at
0.7802 and the corrected legacy CTC/trie stage accepted `play music` as
`play_music`. This verifies API/state handoff compatibility only; it does not
measure continuous streaming, wake boundaries, unseen users, or Pi latency.

The focused suite contains 39 passing tests. It covers dataset/model version
compatibility, recorder collision avoidance, duplicate and split-lineage
checks, complete-word augmentation bounds, read-only calibration separation,
CTC repeated-character and brute-force probability correctness, padding
exclusion, rejection semantics, safe artifact loading, deterministic
initialization, and streaming state recovery/handoff behavior.

## Reproducible commands

```sh
export PYTHONPATH=src

python -m me2.wake_word.data audit \
  wake-word/data/v3/synthetic/manifest.csv \
  --output results/wake_v3_synthetic_audit_2026-10-05.json

python -m me2.wake_word.train \
  --manifest wake-word/data/v3/synthetic/manifest.csv \
  --label-mode binary --allow-pending-review \
  --epochs 30 --seed 43 --run-name v3-binary-seed43-r1 \
  --output wake-word/models/wake_word_v3_binary_seed43_r1.me2

python -m me2.wake_word.evaluate \
  --model wake-word/models/wake_word_v3_binary_seed43_r1.me2 \
  --manifest wake-word/data/v3/synthetic/manifest.csv \
  --split test --output results/wake_v3_binary_seed43_r1_test

python -m me2.vcm.train \
  --architecture CompactCTCModel \
  --run-name compact_ctc_preliminary_2026-10-05 \
  --development-limit 2048 --epochs 1 --beam-width 8 \
  --skip-test-evaluation \
  --output vcm/models/vcm_compact_ctc_preliminary_2026-10-05.me2

python -m me2.vcm.evaluate \
  --model vcm/models/vcm_compact_ctc_preliminary_2026-10-05.me2 \
  --split test \
  --output results/vcm_compact_ctc_preliminary_2026-10-05_test

python -m me2.vcm.analyze \
  --predictions results/reproduced_legacy_ctc_corrected_decoder_with_greedy_test_2026-10-05/predictions.jsonl \
  --model vcm/models/vcm_ctc_model.me2 \
  --output results/analysis_legacy_ctc_corrected_with_greedy_test_2026-10-05

python -m me2.app --interactive \
  --wake-model wake-word/models/wake_word_v3_binary_seed43_r1.me2 \
  --vcm-model vcm/models/vcm_ctc_model.me2 \
  --wake-audio wake-word/data/v2/audition/927f756e547f_guide_connected.wav \
  --command-audio vcm/data/test_sample.wav \
  --allow-unverified-wake-model

ruff check src tests
pytest -q
```

## Remaining evidence

Deployment readiness requires review of every retained synthetic positive,
roughly 20–30 independent real speakers as a pilot with multiple sessions
where feasible, realistic continuous negatives, and representative Pi
microphone recordings. Event recall, false activations/hour, delay, duplicate
rate, sustained CPU/RSS/temperature, and throttling remain unmeasured. A
substantially longer full-data compact CTC training run or a compatible
slot-aware checkpoint is also required; the preliminary compact model must
not replace the corrected legacy CTC artifact.
