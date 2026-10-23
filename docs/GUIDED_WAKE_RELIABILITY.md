# Guided Wake-to-Command Reliability Upgrade

This is an engineering validation workflow, not a deployment approval. Normal
microphone runtime stays automatic. `--guided-terminal` is fixture-only: it
requires both approved WAV inputs and pauses at start, wake initialization,
read, detection, acknowledgement, command initialization/read/detection,
execution, confirmation, and end.

## Private five-sample calibration

On the Pi, keep recordings and manifests under
`/home/rpiadmin/me2-validation-evidence/wake-calibration/`, never in Git.
Run `me2-wake-private-calibrate --model MODEL --sample WAV` five times and
`--output` under that same root. The tool checks format, duration, silence and
clipping, then uses the active model preprocessing and decision threshold. It
stores only per-sample index/validity/decision plus aggregate counts. Passing
requires five valid samples and at least four complete wake detections. It
never authorizes training; do not tune a model on five samples.

## Feedback and capture safety

Dashboard feedback is local only: wake accepted is 880 Hz, command accepted is
1046 Hz, and rejection/error is 440 Hz. There is no TTS. Playback failure is
nonfatal and cannot make a rejected command actionable. The 150 ms post-wake
guard discards only newly arriving samples after wake; pre-roll remains intact.

## Promotion and collection gates

Collect a consented, speaker-disjoint corpus with a provenance/consent manifest
outside Git. Split speakers before training, reserve held-out speakers for
evaluation, and retain continuous negative sessions separately. Promotion
requires documented wake recall, negative-session false-trigger, endpoint, and
connected-command targets on the same candidate/baseline policy. Fine-tune VCM
only if evaluation of correctly segmented commands demonstrates a need.

Create an external `manifest.json` under
`/home/rpiadmin/me2-validation-evidence/wake-corpus/` with only
`schema_version: 2` and `speakers`. Each speaker entry requires opaque
`speaker_id` (`spk-...`), `consent_status: "granted"`, `consent_time`,
`recording_condition`, `device_profile`, and one `partition` (`train`,
`validation`, `calibration`, or `test`). No names or raw identifiers are
permitted. Run `me2-wake-corpus-consent --manifest MANIFEST` before recording;
it requires at least four independent, partition-disjoint consented speakers.

For an existing real-recording corpus, do not add raw WAVs or its detailed
manifest to Git. First produce a separate external derived manifest under the
same evidence root. It must contain its source-manifest SHA-256, a consented
opaque speaker roster, and per-record opaque IDs, external paths, hashes,
labels, review statuses, original partition, and proposed train/validation/
calibration/test partition. Its `schema_version` must be `2`; the proposed
partition is final for the candidate and may not be reassigned during training.
Run:

```sh
me2-wake-ingestion-audit --manifest EXTERNAL_MANIFEST --output EXTERNAL_REPORT
```

The audit permits only granted-consent speakers, accepted-review records,
opaque `rec-...`/`src-...` record IDs, declared source and proposed partitions,
nonempty 16 kHz mono PCM-16 WAVs, matching hashes, and speaker-disjoint
train/validation/calibration/test partitions. The output must be a new
external `.json` file; it prints and writes aggregate counts only. A passing
audit merely permits a
future training proposal; it does not authorize training, calibration, model
promotion, or replacement.

After a passing audit, review its aggregate label coverage for the intended
label mode before preparing a private training manifest:

```sh
me2-wake-training-proposal --audit-report EXTERNAL_REPORT \
  --label-mode five_class --output EXTERNAL_PROPOSAL
```

The proposal requires every frozen partition to contain all five labels for
`five_class` (or both wake/non-wake classes for `binary`). It only confirms
readiness to prepare a private manifest; its `training_authorized` field is
always false until a human explicitly approves a candidate run.

### Evidence-host operator sequence

Run the following on the authorized evidence host: the `rpiadmin` Pi account
that exposes the approved evidence root. The tools enforce that root in code,
so an arbitrary local checkout cannot audit the corpus. The `me2-wake-*` names
are project console scripts and do not exist until the current project checkout
has been installed into its virtual environment. Keep command output and all
resulting JSON in the external evidence store; do not redirect it into the
repository, normal logs, tickets, or chat.

```sh
# Confirm the clean checkout is at the reviewed revision, then fast-forward it.
git status --short --branch
git rev-parse HEAD
git pull --ff-only
git merge-base --is-ancestor 02a13b2f400178121c1e88d8086028f21328c318 HEAD

# Refresh console-script wrappers and source from this checkout only. This
# installs no dev or train extras and does not touch models or evidence data.
.venv/bin/python -m pip install -e .
.venv/bin/me2-wake-corpus-consent --help
.venv/bin/me2-wake-ingestion-audit --help

# 1. Validate the consent roster before any corpus processing. The module form
# is the fallback if a shell cannot resolve a console-script wrapper.
.venv/bin/python -m me2.wake_word.corpus_consent \
  --manifest EXTERNAL_CONSENT_MANIFEST

# Expected safe output: schema_version 2, recording_enabled true, four
# aggregate partitions (train, validation, calibration, test), and speaker_count >= 4.
# STOP if the command exits non-zero or any listed condition is absent.

# 2. Validate the derived corpus and write a new aggregate-only report.
.venv/bin/python -m me2.wake_word.ingestion_audit \
  --manifest EXTERNAL_DERIVED_MANIFEST \
  --output EXTERNAL_AUDIT_REPORT

# Expected safe output: audit_schema_version 2,
# valid_for_training_proposal true, issues [], four nonzero partition counts,
# and aggregate labels_by_partition only.
# STOP if the command exits non-zero, valid_for_training_proposal is false,
# issues is nonempty, a partition is empty, or the report contains a path,
# speaker identifier, recording identifier, transcript, or raw metadata.

# 3. Confirm aggregate label coverage without preparing or training a model.
.venv/bin/python -m me2.wake_word.training_proposal \
  --audit-report EXTERNAL_AUDIT_REPORT \
  --label-mode five_class \
  --output EXTERNAL_TRAINING_PROPOSAL

# Expected safe output: proposal_schema_version 1,
# ready_for_private_manifest_preparation true, and training_authorized false.
# STOP if the command exits non-zero, readiness is false, authorization is
# true, a required label is absent from any partition, or any output is not aggregate-only.
```

The ancestry command exits zero only when the checkout contains the reviewed
corpus-audit/proposal implementation. Stop if the checkout is dirty, the
fast-forward or ancestry check fails, editable installation fails, either
`--help` command is unavailable, or the inputs are outside the approved
evidence root. A local repository host without that root runs only repository
tests and lint; it must not attempt a corpus audit. Do not run a training, calibration, evaluation,
manifest-preparation, or model-promotion command at this stage. The next action
is a human review of the two aggregate reports and explicit written
authorization for a separate private-manifest-preparation proposal.

## Pi regression matrix

For muted playback then approved normal-speaker trials, record only aggregate
event/endpoint/clipping outcomes. Separately exercise dashboard feedback. For
each baseline, reboot, and hub re-enumeration condition: require clean SHA,
non-numeric device selection, input/output health, supervised session, then
conditional supplied-WAV Step 7. A live wake rejection, device fault, timeout,
clipping, or ambiguity is reported as such and never dispatched as an action.
