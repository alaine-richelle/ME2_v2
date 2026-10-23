# Guided Wake Reliability: Runbook and Progress — 2026-10-24

## Current decision

The engineering flow is **not deployment-approved** and the training gate is
**closed**. Repository-side consent, ingestion-audit, and aggregate
training-proposal gates are implemented and tested. No external derived
manifest or aggregate audit report is available in this workspace, so no
private corpus has been reviewed and no candidate training, calibration,
evaluation, promotion, or model replacement has begun.

Keep raw audio, detailed manifests, recordings, transcripts, credentials, and
all reports under the approved evidence root only. Do not copy them to Git,
ordinary logs, tickets, or chat. The only permitted root named here is
`/home/rpiadmin/me2-validation-evidence/`.

## Phase map

| Phase | Status | Evidence / interpretation |
| --- | --- | --- |
| Fail-closed guided runtime and fixture-only guided terminal flow | Completed | The runtime keeps the normal microphone flow automatic; `--guided-terminal` requires supplied fixtures. Component coverage includes streaming and wake tests. |
| Feedback, capture, and rejection safety | Completed in code; live behavior not established | Feedback remains local-only; failure is non-actionable; wake/decoder faults fail closed. This is implementation/component evidence, not a live result. |
| Local component regression | Completed for the current repository | Focused corpus/proposal/wake tests passed and Ruff passed. The full suite has one known sandbox-only localhost socket-bind failure; it is unrelated to the wake gates. |
| Pi package, model-load, negative replay, and dashboard-demo checks | Completed historically, not current-revision live validation | Prior Pi evidence established installation/model-load and limited supplied-file/negative-path checks. It does not establish current-revision live recognition, timing, or deployment readiness. |
| Consent roster and four-way speaker split | Blocked | Schema v2 tooling requires disjoint `train`, `validation`, `calibration`, and `test` speakers, but no external roster evidence is mounted here. |
| Derived-manifest, provenance, audio metadata, and leakage audit | Blocked | The audit command is implemented. The external evidence host/root is unavailable from this workspace, and no aggregate result has been reviewed. |
| Aggregate label-coverage proposal | Blocked | It may run only after a passing audit. It remains non-authorizing even when it passes. |
| Private training-manifest preparation | Not started | Requires human review of the passing aggregate audit and proposal plus explicit written authorization. |
| Candidate training, validation selection, calibration, and frozen-test evaluation | Not started and prohibited | No candidate may be trained until the preceding authorization; no incumbent may be overwritten. |
| Continuous-negative and true live wake-to-command validation | Not started for the current gated flow | Historical bounded Pi microphone activity did not observe a wake or command result. It is not a wake-recall, false-trigger-rate, endpoint, or connected-command result. |
| Promotion/deployment | Not started and prohibited | Existing artifacts remain supervised/diagnostic only; no promotion criterion has been met. |

## Where we are today

The next required action is evidence-host **consent and corpus auditing**, not
training. A passing audit must be reviewed as aggregate-only evidence, then a
separate aggregate label-coverage proposal must pass. Both outcomes still
leave `training_authorized` false. Stop there until a human provides explicit
written authorization for a distinct private-manifest-preparation plan.

## Ordered operator commands

Run commands in this order. `EXTERNAL_*`, `APPROVED_*`, and
`REDACTED_*` are placeholders; do not substitute their values in shared logs
or chat.

### A. Repository/local — component verification only

```sh
git status --short --branch
git rev-parse HEAD
.venv/bin/python -m pytest -q \
  tests/test_wake_corpus_consent.py \
  tests/test_wake_ingestion_audit.py \
  tests/test_wake_training_proposal.py \
  tests/test_wake_v2.py \
  tests/test_streaming.py
.venv/bin/python -m ruff check src tests
```

Expected safe output: the intended branch/commit, passing component tests, and
`All checks passed!` from Ruff. Stop if the working tree contains unexpected
changes, any command exits nonzero, or the branch/commit is not the reviewed
revision. These checks do not access a microphone, authenticate a Pi, inspect
private evidence, or prove live recognition.

### B. Evidence host/Pi account — required corpus gates

```sh
# These corpus tools run on the rpiadmin Pi account because their code enforces
# the approved evidence root there; they are not local-repository commands.
git status --short --branch
git rev-parse HEAD
git pull --ff-only
git merge-base --is-ancestor 02a13b2f400178121c1e88d8086028f21328c318 HEAD
.venv/bin/python -m pip install -e .
.venv/bin/me2-wake-corpus-bootstrap
.venv/bin/me2-wake-corpus-bootstrap --verify
.venv/bin/me2-wake-corpus-consent --help
.venv/bin/me2-wake-ingestion-audit --help

# 1. Consent roster: four disjoint planned speakers. Module form is the
# fallback when a shell cannot resolve a console-script wrapper.
.venv/bin/python -m me2.wake_word.corpus_consent \
  --manifest EXTERNAL_CONSENT_MANIFEST

# 2. Derived-manifest and audio integrity/leakage audit.
.venv/bin/python -m me2.wake_word.ingestion_audit \
  --manifest EXTERNAL_DERIVED_MANIFEST \
  --output EXTERNAL_AUDIT_REPORT

# 3. Aggregate label-coverage proposal; this does not train.
.venv/bin/python -m me2.wake_word.training_proposal \
  --audit-report EXTERNAL_AUDIT_REPORT \
  --label-mode five_class \
  --output EXTERNAL_TRAINING_PROPOSAL
```

Expected safe outputs, in order:

1. Consent: `schema_version` 2, `recording_enabled` true, four aggregate
   partition names, and `speaker_count` at least 4.
2. Audit: `audit_schema_version` 2, `valid_for_training_proposal` true,
   `issues` as an empty list, four nonzero partition counts, and aggregate
   `labels_by_partition` only.
3. Proposal: `proposal_schema_version` 1,
   `ready_for_private_manifest_preparation` true, and
   `training_authorized` false.

Stop immediately if any command exits nonzero; a consent field is missing; an
audit issue, empty partition, duplicate/leakage finding, or non-aggregate
field appears; label coverage is incomplete; or the proposal authorizes
training. Retain only the two aggregate JSON reports externally. Do not run a
manifest-preparation or training command after this sequence without separate
written authorization. Stop before the consent command if the checkout is
dirty, the fast-forward/ancestry check fails, editable installation fails,
either `--help` command is unavailable, or the inputs are outside the approved
evidence root.

When the evidence root is empty, run only the bootstrap command, inspect its
aggregate inventory, and stop. It creates static schemas but no real consent
or corpus manifest. Then run `me2-wake-corpus-bootstrap --verify`; proceed
only when it reports two valid templates. Human consent registration for four independent opaque
speakers is the next gate; recording and derived-manifest creation follow only
after that registration.

### C. Pi — deterministic component checks after the evidence review and a reviewed deployment

These commands require a separately approved Pi session and a checked-out,
reviewed repository revision. They must not be used as a substitute for the
corpus gates above. The same Pi account is the evidence host for the
root-enforced corpus audit; after that audit, Pi work returns to runtime/audio
and hardware validation.

```sh
# 4. Verify the Pi and configured runtime without opening a microphone.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.health \
  --config APPROVED_PI_CONFIG --require-pi5 --output EXTERNAL_HEALTH_REPORT

# 5. Run a supplied-WAV integrated replay only; no microphone is opened.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --interactive --config APPROVED_PI_CONFIG \
  --wake-audio APPROVED_WAKE_FIXTURE \
  --command-audio APPROVED_COMMAND_FIXTURE \
  --threads 2 --log-file EXTERNAL_REDACTED_REPLAY_LOG

# 6. Score exactly five approved private wake samples; this never tunes a model.
me2-wake-private-calibrate --model APPROVED_WAKE_MODEL \
  --sample PRIVATE_WAKE_SAMPLE_1 --sample PRIVATE_WAKE_SAMPLE_2 \
  --sample PRIVATE_WAKE_SAMPLE_3 --sample PRIVATE_WAKE_SAMPLE_4 \
  --sample PRIVATE_WAKE_SAMPLE_5 --output EXTERNAL_CALIBRATION_REPORT
```

Expected safe outputs, in order:

4. Health reports a Pi-compatible configuration and artifact health; this is
   not deployment approval.
5. The replay reaches a wake-gated command result using supplied fixtures.
   This is deterministic component integration, not a microphone result.
6. Calibration prints aggregate counts only: five valid samples and at least
   four full detections are needed for `passed` true; `training_permitted`
   remains false.

Stop if health fails, an artifact/config identity differs from the reviewed
deployment, the replay fails closed or produces an unexpected action, a sample
is invalid, calibration does not pass, or any command emits/retains raw audio,
transcripts, a private filename, or sensitive logs. A passing step 6 does not
open the training gate.

### D. Pi — true live wake-to-command validation (not yet authorized)

No live command is authorized by this runbook. Only after separate written
approval for a bounded, supervised trial—and after the evidence-host gates and
candidate evaluation criteria are satisfied—use an approved combined-pipeline
command with a fixed duration, no raw-audio retention, redacted aggregate
logging, and the explicit engineering override required by an unapproved wake
candidate. Stop on the first device fault, unintended wake, rejected/ambiguous
command, unexpected external action, clipping, or timeout. Record aggregate
event counts and durations only.

## What the evidence means

Component tests, health checks, file replays, dashboard demos, device
enumeration, and tone routing show bounded portions of the system. They do
**not** measure real wake recall, continuous false-trigger rate, endpoint
quality, connected command success, physical audibility, or deployment
readiness. Only the separately authorized, supervised, bounded live combined
pipeline—using a reviewed candidate and aggregate evidence—can produce those
measurements. Even then, promotion requires the documented candidate versus
baseline comparison and all rejection, compatibility, privacy, and deployment
guardrails.
