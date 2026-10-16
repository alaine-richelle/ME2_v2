# Runtime hardening experiment — 2026-10-13

## Starting state and protected protocol

Branch `master`, commit `022452a954d6e9d11c3eb7fee584167cdb1cdf42`,
integration baseline `52b38f7fdaf35a0f5b5a8dbbd239f4bfb7700e24`.
The historical CTC reference audit used
`f5d2dc585b49b46e8518ec6122261a9ca7f2f5e4`; its Option B experiment is not rerun
or compared as a matched score in this cycle.
Tracked files were clean; existing untracked datasets, artifacts, reports,
logs and navigation skills are preserved. No push or artifact replacement is
part of this experiment.

The selected command implementation remains the corrected full-utterance CTC
prefix decoder with the retained `vcm_ctc_model.me2` weights. The
`vcm_baseline_model.me2` classifier and calibration-only
`vcm_ctc_calibrated_2026-10-05.me2` are comparison artifacts, not new promotions.
The preliminary compact CTC and all wake artifacts lack deployment evidence.
The v5 regularized wake artifact is permitted only for explicitly unverified,
controlled engineering checks.

Retention update (2026-10-15): the calibrated and preliminary compact candidate
bundles were pruned after metric review. Their historical reports and
predictions remain; see `docs/MODEL_SELECTION_2026-10-15.md` for exact
dispositions and regeneration requirements.

Read starting evidence: `docs/IMPROVEMENT_CYCLE_2026-10-06.md`, saved
`results/reproduced_*_test_2026-10-05/metrics.json` and
`results/analysis_*_test_2026-10-05/`. Those official-test results were already
inspected in development; they are historical reused assessments, not fresh
final performance. No official test or holdout is decoded or tuned in this cycle.

New matched regression evaluation uses only the persisted **validation**
indices in `results/vcm_ctc_partitions.json`, from official HF train revision
`bfca0bbebfe17e00d6a685d1da573615d29ecdce`. Select up to five rows per label
by SHA-256 of the source index with a fixed namespace. Save exact indices and
partition/shard hashes. This small class-balanced check is a regression
diagnostic, not an estimate of deployment prevalence, generalization or
calibration quality. No thresholds, vocabulary, grammar or weights change.

## Ranked experiments (fixed before implementation)

| Experiment | Evidence and primary change | Acceptance and risk | Rollback |
|---|---|---|---|
| E1: fail-closed contracts | `bundle.py::_validate_metadata` accepts missing task/schema and non-finite acceptance values; safe ZIP loading is unbounded. Validate configuration, archive/tensor limits, finite values and prediction contracts before activation. | All retained safe artifacts load; malformed artifacts/NaN thresholds/logits fail; identical accepted intents/rejections/text on matched dev batch. Risk: intentional rejection of previously tolerated malformed artifacts. | Revert contract commit; retain original artifacts unchanged. |
| E2: bounded decoder | `ctc.py::prefix_beam_search` has no frame/work limit. Add deterministic work limits, explicit exhaustion, and finite-terminal filtering. Preserve normal beam/recurrence and thresholds. | Brute-force probability tests still agree; exact decisions on dev batch; budget exhaustion never dispatches. p95 latency regression below 20% on same host/threads, otherwise investigate. Not a hard real-time guarantee. | Revert decoder changes or use previous source release, not altered weights. |
| E3: privacy/operability | `app.py` logs transcript/slots/paths at INFO; formatter persists exception paths; stream shutdown does not always call `stop`. Add allowlisted structured runtime events, version/correlation context, health/config validation and guaranteed cleanup. | No transcript/path/exception-message leakage in default logs; explicit debug opt-in; lifecycle/fault tests; same valid predictions. Risk: less default free-text detail, recovered through bounded safe reason codes. | Revert observability commit; known-good artifacts remain separate. |

E4 was added only after the unrestricted read-only device probe hung for more
than 48 seconds during validation. Primary change: isolate native device
discovery in a child process with a ten-second timeout. Acceptance: return a
nonzero `device_probe_timeout` diagnostic, terminate the child, and never open
a recording stream. Risk: unusually slow successful discovery may time out;
target validation must measure this. Rollback is the separate device-probe
commit; reverting restores the known unbounded discovery risk.

Comparisons record all 20 labels (19 intents plus OOS), macro F1, accuracy,
OOS false acceptance, in-scope rejection, wrong accepted intents, coverage,
parameter/artifact bytes, stage/total timings, active CPU and process RSS.
Explicit slot recognition is still absent in the legacy artifact; no slot
quality improvement is claimed. Existing onset-boundary uncertainty is not
solved by these changes.

## External gates

Actual Pi SSH target/microphone and independently reviewed real wake recordings
are not yet available. Laptop file/replay success is not acoustic handoff or Pi
readiness. Microphone inspection must not silently record or store raw audio.
Do not choose a target wheel lock or claim operator/thermal performance without
the target OS/Python/runtime. A target-created pinned wheelhouse and health
report will be required before offline deployment.

## Results: new laptop measurements, not Pi measurements

Environment: macOS 26.6.2 ARM64, Python 3.14.3, PyTorch 2.14.1,
NumPy 2.5.3, SciPy 1.18.1, soundfile 0.14.0; two Torch threads.
The fixed subset contains 100 clips, five per label, 23 source identities.
Original validation assignments and both cached official-train shard hashes
were unchanged. Two rows (source indices 2104 and 10015) exceed the CTC
artifact duration and remain explicit failures in the denominator; none were
silently removed. The baseline retains its historical center-cropping behavior.

| Artifact | Parameters / bytes | Accuracy before → after | Macro F1 before → after | OOS false accepts / 5 | In-scope rejects / 95 | Wrong accepted / 95 | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Retained classifier | 8,052 / 34,269 | .43 → .43 | .39494 → .39494 | 4 → 4 | 0 → 0 | 53 → 53 | .99 → .99 |
| Selected legacy CTC | 3,652,572 / 13,550,852 | .78 → .78 | .80278 → .80278 | 1 → 1 | 14 → 14 | 7 → 7 | .82 → .82 |
| Calibration-only CTC | 3,652,572 / 13,551,757 | .73 → .73 | .76387 → .76387 | 1 → 1 | 22 → 22 | 4 → 4 | .74 → .74 |

All per-row accepted intents, rejection reasons, candidate strings and scores
match exactly, not just aggregate metrics. Both CTC artifacts have the same
weight hash. This verifies regression safety; **recognition quality did not
improve**. The cross-architecture differences are not a controlled decoder
gain: capacities, training ancestry and artifact preprocessing differ. Legacy
training provenance is incomplete, so independence of this subset from those
weights cannot be established. Macro F1 includes all 20 labels; rejection is
counted as OOS for that classification metric and retained separately in
per-row diagnostics.

| Artifact | p50 ms before → after | p95 ms before → after | Active CPU, % of one core | Process RSS bytes before → after |
|---|---:|---:|---:|---:|
| Classifier | 1.005 → 1.174 | 1.480 → 1.239 | 111.5 → 121.6 | 1,125,564,416 → 1,138,442,240 |
| Selected CTC | 7.002 → 6.850 | 12.093 → 12.506 | 103.9 → 103.9 | 1,218,265,088 → 1,231,372,288 |
| Calibrated CTC | 6.681 → 6.818 | 12.260 → 11.870 | 104.2 → 105.0 | 1,258,242,048 → 1,271,267,328 |

One measured pass per artifact, three warmups. Latency excludes preprocessing
of source files, capture and endpointing; stage timings are saved per sample.
RSS includes resident Arrow mappings and other loaded Python libraries: it is
neither peak RSS nor the standalone runtime footprint. These observations do
not establish a speed improvement or a sustained thermal result. Selected CTC
p95 increased 3.4%, inside the predeclared 20% regression guardrail.

Machine-readable local evidence (intentionally not committed):
`results/hardening_2026-10-13/before.json`, `after.json`, `health.json`,
`installed_health.json`, `devices.json`, `pi_gate.json`.
They include full artifact hashes, saved indices, shard/partition hashes,
versions, per-row predictions and stage timings. No weights, thresholds,
grammar, training partitions or source recordings were changed. There was no
new training or quantization in this cycle.

A clean tracked-runtime repeat at
`3926a1beeddc5c4944ed216dfd17d9b8ca24aed5`, saved as
`committed_runtime.json`, again matches **100/100 decisions and scores for each
artifact**. Selected-CTC p95 is 12.151 ms in that repeat; baseline 1.541 ms,
calibrated CTC 12.008 ms. Earlier observations above are retained rather than
replaced by the fastest run. The repeat also records the actual repository path,
branch and validation-script hash. Runtime imports passed with `datasets`,
`sklearn` and `pyarrow` explicitly forbidden.

## Implemented changes and limits

- **E1:** `bundle.py` validates schema/task, labels, audio/frontend rates and
  budgets, vocabulary/grammar agreement, finite thresholds and weights, bounded
  dimensions, duplicate ZIP/JSON keys and nested NPZ expansion before model
  activation. Plain numeric tensors still use `allow_pickle=False`. Schema 2
  remains compatible: all **22** retained `.me2` artifacts load. Pickle migration
  remains explicit/trusted-only. Metadata hashes detect changes but are not
  authentication: verify a complete artifact hash from a trusted release.
- **E2:** CTC recurrence and selected beam width remain unchanged. Runtime
  version `bounded-prefix-runtime-v2` caps frames, vocabulary/beam work and
  grammar size. Impossible `-inf` terminals are not candidates. Budget failure
  raises `DecoderBudgetError`; streaming preserves a non-actionable system
  error and recovers through cooldown. These are deterministic work bounds,
  not hard real-time deadlines. Semantic scoring still aggregates legacy
  candidates by intent; a slotted artifact needs semantic-candidate calibration
  before adoption.
- **E3:** validated runtime JSON with explicit CLI precedence; unique atomic
  temporary writes; default privacy-safe rotating JSON logs; model/config/
  artifact versions and utterance IDs; explicit system error versus rejection;
  SIGTERM/Ctrl+C cleanup; stopped event and buffer cleanup on every stream exit.
  Missing wake approval now blocks live use, like explicit `false`.
- Source inspection additionally confirmed runtime preprocessing divergence:
  single-command microphone capture center-padded CTC input, continuous wake
  inference omitted the artifact preprocessing, and continuous baseline
  commands used CTC right-padding. These routes now call the shared appropriate
  preparation function. Array parity is tested. **Continuous acoustic accuracy
  and wake smoothing must be recalibrated on permitted real validation audio**;
  synthetic state-machine tests do not establish an accurate wake boundary.
- `me2-health` verifies archive/hash, CPU forward/operator execution, optional
  device format, actual board identity and available telemetry without opening
  a microphone, accessing a dataset, or dispatching commands. Installed-wheel
  health passed from `/private/tmp`. The wheel uses the existing runtime
  environment; this is not proof of a fresh target dependency installation.

Implementation evidence at the recorded source revision:
`src/me2/bundle.py:55` (bounded tensor archive),
`src/me2/bundle.py:400` (artifact settings),
`src/me2/vcm/ctc.py:36` (decoder budget),
`src/me2/vcm/ctc.py:76` (finite rejection policy),
`src/me2/logging_utils.py:32` (safe log-write failure),
`src/me2/logging_utils.py:83` (default redaction),
`src/me2/app.py:68` (live approval),
`src/me2/app.py:154` (shared command preprocessing),
`src/me2/streaming.py:285` (guaranteed stream cleanup).

The targeted fault tests isolate the three mechanisms; the matched prediction
comparison measures the combined patch. There is no claimed per-mechanism
recognition gain. New tests also preserve prior brute-force CTC probability,
repeated-symbol, padding, streaming timeout, overflow, retry and dispatch tests.
Final recorded full run: **105 passed in 9.81 seconds**, including the local
dashboard server (loopback permission required). Lint and diff checks are part
of the final pre-commit gate. Earlier failed assertions and the initial
overlength-data failure were corrected, not reported as passing runs.

File checks: baseline ran and rejected the sample as OOS; selected CTC accepted
`play_music` with normalized score −0.0004931782. Separate-file wake gating
detected the already approved connected anchor and accepted the same command.
This is not a continuous-audio handoff test or a new pronunciation review.
No external action was dispatched.

## Remaining requirements and selection

Retain the selected command weights and all baselines; accept the fail-safe
runtime engineering changes, **not** a newly promoted acoustic model.
The calibrated command artifact still fails the historical coverage guardrail.
Legacy CTC still lacks supported-slot mappings; arbitrary slot recognition is
not implemented by this patch. The preliminary compact model remains rejected.

Sandboxed microphone enumeration returned no input devices. The unrestricted
read-only probe then hung beyond 48 seconds and its exact process was
terminated. With the new subprocess bound, the unrestricted probe returned
`TimeoutExpired` / `device_probe_timeout` and status 1 as designed; see
`devices_bounded_unsandboxed.json`. This does not prove the physical machine
has no microphone. **Live capture is blocked by native discovery**, and no
audio was recorded. Actual Pi SSH/device information
was requested but not available. Pi installation, pinned target dependency
lock, cold process start, sustained CPU/peak RSS, idle cost, temperature/
throttling, event recall/false activations per hour and true wake-to-command
latency are **not run**. The Pi-only health gate fails on this laptop as intended.
The streaming timeline still uses sample indices rather than verified ADC
timestamps, and the pre-roll is not a verified wake-end annotation. Those
limitations must be resolved or measured before claiming precise continuous
handoff latency.

## Commit and artifact ledger

- `d19bbb6`: inference/configuration validation, bounded decoder, privacy
  events, preprocessing parity, health CLI, cleanup and regression tests.
- `3926a1b`: log-I/O failure handling that does not print original private
  message arguments; fatal log failure is not retried as a microphone fault.
- `fb952df`: bounded native device-discovery subprocess, reproduced timeout
  diagnostic and focused regression.
- The subsequent documentation commit contains this report, the fixed
  validation runner and `docs/RUNTIME_VALIDATION.md`; its identity is recorded
  in Git history and the task handoff.

Committed implementation files:
`src/me2/{app,audio,bundle,contracts,features,health,io_utils,logging_utils,runtime_config,streaming}.py`,
`src/me2/vcm/ctc.py`, `src/me2/wake_word/detector.py`,
`config/runtime.example.json`, `pyproject.toml`,
`tests/test_{audio_features,dashboard,runtime_hardening}.py`.
There was no amend, force push, reset, model replacement or additional deletion.
No commits were pushed.

Intentionally uncommitted: all existing datasets and model artifacts,
historical reports/logs, navigation skills, new local
`results/hardening_2026-10-13/` evidence and `logs/hardening_*` JSON logs, build
output and temporary wheel/install directories. New evidence is separate from
final models. The final source wheel was built offline with
`pip wheel --no-index --no-deps --no-build-isolation .`; its SHA-256 is
`b4bf06efbdb3cb99549b116da977d550dabd8c702e742592109ddb1e143952df`.
It is a host-tested package wheel, **not** an approved ARM64 dependency bundle.
Its hash-verified CTC operator check passed from an installed target outside the
repository; see `installed_release_health.json`.

Next highest-value work: independently reviewed real connected-word wake
recordings and continuous negatives; a target microphone/Pi validation run;
then command training with verified transcript/slot grammar and a better
development-calibrated rejector. Do not tune from the historical inspected
official test; reserve a fresh final assessment.
