# AI231 ME2 final v5 selection, cleanup, and Raspberry Pi 5 readiness prompt

Act as the release engineer for the AI231 ME2 voice-command project. Complete
the **final commit preparation** in this checkout, beginning from the current
dirty worktree. Preserve user changes, do not reset or broadly clean the
repository, and do not push, alter a remote, create a PR, or deploy to the
Pi. The user, not the agent, will push the reviewed current local changes
before the Raspberry Pi 5 transfer.

## User-owned push and exact Pi source identity

The Pi must receive the exact repository revision that the user has already
pushed. Before any Pi transfer/write step, record the pushed commit SHA,
remote name/URL without credentials, and branch/tag in the finalization
record. Confirm the local revision is clean/reviewed and that the intended
commit is reachable from the user-owned remote. Do not use an uncommitted
working tree, a ZIP/TAR copy of local files, `git push`, force-push, or a
branch name alone as the Pi source identity.

The agent may prepare the commands below, but the user performs the push.
After the user confirms it is pushed, the Pi procedure must clone/fetch that
exact commit into a new isolated directory and check out the commit in
detached-HEAD mode. This makes the Pi result traceable to the final commit
and prevents accidental transfer of unrelated local files.

## Remaining authorized Pi blockers — complete these now

The user has confirmed that the Pi is powered on, the reviewed repository
revision is on GitHub, and the time-consuming Pi setup is authorized. Resume
from the isolated, detached checkout only after re-verifying its exact commit,
model/config hashes, virtual-environment path, and no-overwrite rule.

Complete the remaining *direct-model and demo-dashboard* gates in this order:

1. **Package/install evidence.** Finish the binary-only dependency audit and
   isolated venv installation if either is incomplete. Record `pip check`,
   dependency versions, `pip freeze --all`, Python, glibc, and architecture.
   Stop on a missing wheel, failed non-audio import, or mismatched artifact
   hash; do not source-build, use `sudo`, alter system Python, or replace a
   package silently.
2. **Headless audio-library limitation.** If `sounddevice`/PortAudio cannot
   initialize because the noninteractive SSH session lacks PulseAudio, record
   it as a live-audio limitation only. It is not a blocker for offline model
   loading, supplied-WAV replay, or `me2.dashboard --demo`. Do not start a
   PulseAudio service, alter system audio, or open a microphone to work around
   it.
3. **Fixture audit.** List tracked WAV files, identify their source and
   intended role, and use only a suitable, already-tracked fixture for each
   direct replay. Do not upload personal recordings, record new audio, or use
   an unreviewed file merely to obtain an accepted result. If no appropriate
   tracked fixture exists, report the exact missing fixture and stop that
   replay gate instead of substituting data.
4. **Direct model checks.** Run individual v5 wake, v3 wake, and CTC command
   replays. Then run the full supplied-WAV wake-to-command replay, using the
   implemented v5-primary/v3-diagnostic configuration. Save redacted logs
   and result files with a new run ID. Capture model load status, resolved
   paths/hashes, scores/decisions, valid frames, explicit rejections, and the
   dual-wake role/disagreement outcome. These are file-based results, not
   live recognition measurements.
5. **Dashboard demonstration.** Start only
   `python -m me2.dashboard --demo` without `--autostart`, bind it to Pi
   localhost, and verify its local status/API response. Give the user the
   normal SSH port-forward command so they can inspect the UI from the laptop.
   Do not start dashboard listening, use a microphone/speaker, or perform
   external actions.
6. **Final record.** Update the Pi/finalization documentation with the exact
   Git revision, package state, artifact/config hashes, replay outcomes,
   dashboard status, and blockers that remain deferred. Distinguish
   *passed direct file/model check*, *blocked/inconclusive*, and *not run*.
   Never upgrade the release state from supervised/final-commit-ready to
   deployment-ready.

The only explicitly deferred blockers after this scope are live microphone
recognition, speaker audibility, Bluetooth, dashboard listening, continuous
stream stability/telemetry, false-activations-per-hour, and deployment
approval. Do not perform them without another explicit user instruction.

When the user separately approves a future Raspberry Pi 5 speaker test, use
the wired USB/ALSA speaker output identified by the approved device inventory
(for example, the confirmed `plughw:CARD,DEVICE` value). Do not use Bluetooth,
HDMI, or a guessed card/device index. First list the
available ALSA playback devices, state the exact selected wired device and
bounded playback command to the user, then wait for approval. A successful
software playback command establishes only the wired ALSA output path; ask the
user whether the tone/sample was physically audible and record that answer
separately.

## Current handoff and non-negotiable interpretation

Read these records before changing files:

1. `docs/DEADLINE_PLAN_PROGRESS_2026-10-15.md`
2. `docs/MODEL_SELECTION_2026-10-15.md`
3. `docs/PI5_SSH_SAFE_VALIDATION_2026-10-15.md`
4. `docs/V5_FINALIZATION_2026-10-15.md`
5. `docs/CLEANUP_2026-10-06.md`
6. `AGENTS.md`

The safe dashboard demonstration and offline checks are complete. Raspberry
Pi 5 deployment validation is not complete. The historical records include
an isolated Pi runtime deployment, package/import checks, synthetic forwards,
and earlier hardware trials; first obtain a fresh read-only inventory before
assuming that remote files or environments still exist. Do not represent
host-file replay, ALSA enumeration, a hardware-only microphone probe, or
software playback as wake/command quality or deployment validation.

## Scope change — direct Pi model testing only

The operator does **not** want another live test session on the Raspberry Pi
5 for this deadline pass. On the Pi, do not open a microphone, capture audio,
play a speaker tone, start the dashboard's live listening loop, test
Bluetooth, run a continuous stream, or ask for a spoken phrase/`GO`/`DONE`
sequence. This Pi restriction does not prohibit the separately approved,
supervised laptop test plan below.

The Raspberry Pi work in this prompt is limited to deterministic,
file-based testing of the copied v5 wake and CTC command bundles on the Pi:
package/import validation, artifact/config hashes, individual model replay,
one supplied-WAV integrated replay, and a visual dashboard review in
`--demo` mode. The dashboard demo is allowed only to inspect the local UI and
its simulated actions; it must not use `--autostart` or start listening.
These checks establish only that the models can load and run against frozen
files on the Pi, and that the dashboard UI is reachable. They do not measure
live wake recall, false activations per hour, microphone reliability, speaker
audibility, endpoint latency, sustained streaming, or deployment readiness.
Record all of those as explicitly deferred rather than failed, passed, or
implied by the file tests or demo UI.

## New scope — controlled built-in-microphone laptop preflight

Use the built-in MacBook Pro Microphone for a separate, human-readable laptop
preflight walkthrough. This is controlled local evidence with the operator
present, not a deployment approval. It must use bounded sessions, create no
raw recordings, avoid credentials and transcripts in shared logs, and keep
Pi validation as the primary operational workflow.

The walkthrough must explain that:

- the built-in MacBook Pro Microphone is used for bounded local checks;
- the command-line application itself prints recognition state to the
  terminal; it does not speak a voice response;
- the dashboard's `call` and `message` actions remain simulations, and no
  physical device, call, message, GPIO, or external account action is
  allowed;
- a failed wake, rejected command, or absent sound is useful diagnostic
  evidence. Do not alter thresholds or mark a model ready merely to obtain a
  more convenient demonstration result.

## Superseded dual-wake implementation

The following historical request is superseded. The implemented policy is
v5-only: V3 is not scored or loaded by the runtime, and a v5 load, frontend,
or inference fault blocks command capture rather than activating a fallback.

Implement the two retained wake artifacts as a clearly documented
**primary-plus-fallback pair**:

| Role | Model | Why it remains |
| --- | --- | --- |
| Primary supervised candidate | `wake-word/models/wake_word_v5_regularized_seed43.me2` | It detected 2/3 wakes on the comparable small test, but made 4/48 false accepts. |
| Diagnostic/fault fallback | `wake-word/models/wake_word_v3_binary_seed43_r1.me2` | It made 0/48 false accepts, but detected 0/3 wakes; it remains the dashboard file-replay fixture. |

“Side by side” must mean both models receive the same prepared wake window
for comparison and structured diagnostics. It must **not** mean an unsafe
automatic OR rule. Implement and test the following policy:

1. V5 remains the primary decision source.
2. For each window, score v3 and v5 with their own embedded threshold and
   smoothing policy; log only allowlisted model role, score, decision,
   artifact hash, and latency fields.
3. When v5 loads and runs normally, a v5 non-wake stays a non-wake even if
   v3 disagrees. Record the disagreement for review; do not trigger command
   capture.
4. Invoke v3 as an active fallback only when v5 cannot load or has an
   inference/system fault. Emit a visible degraded-mode warning and preserve
   the v5 failure reason.
5. If both models fail, fail closed: do not capture a command or dispatch an
   action. Recovery must return the stream to wake listening safely.
6. Do not change either bundle, readiness flag, threshold, smoothing,
   cooldown, CTC model, grammar, or command acceptance policy.

Use the explicit v5-only runtime configuration. Do not silently overload the existing
single-model path. Make the primary/fallback order visible in the CLI,
dashboard status, health output, and redacted logs. Add focused tests for
normal v5 selection, dual-score diagnostics, v5 model-load failure, v5
inference failure, v3 fallback success/failure, disagreement without command
capture, and no unintended action dispatch.

The requested final-commit selection is:

| Role | Artifact | Evidence that must remain in the final record |
| --- | --- | --- |
| Final supervised runtime/trial wake candidate | `wake-word/models/wake_word_v5_regularized_seed43.me2` | SHA-256 `d7a4898afdc5a5a52fcd305424df4b62582d7bdeea58451d2a90e62646a49222`; test 2/3 wake events and 4/48 false accepts |
| Dashboard file-replay fixture | `wake-word/models/wake_word_v3_binary_seed43_r1.me2` | SHA-256 `35ec56088979aea4d16f547dfcda047fac969b8f9d46a60176ee1703d853170b`; test 0/3 wake events and 0/48 false accepts |
| Selected command-development model | `vcm/models/vcm_ctc_model.me2` | SHA-256 `b5a358e2815ee2471f29956e63936af87c4359bf28b21c21ab420c21e128eeac`; official matched test 0.8911 accuracy, 0.8682 macro F1, 20/76 OOS false accepts, and 316/4,367 in-scope rejections |

This is a **final-commit selection for supervised engineering evaluation**,
not a production or unattended deployment promotion. V5 is the retained
runtime/trial candidate because it has higher wake recall than v3, but its
4/48 false accepts exceed the documented 2/48 guardrail. Therefore:

- Never set `deployment_ready=true`, remove the supervised override, relax
  wake/command thresholds, or call v5 Pi-ready.
- Do not replace the v3 dashboard fixture merely to make every entry point
  identical. A fixture with 0/3 wake recall is not suitable for live use,
  but it remains evidence for the dashboard file replay.
- Keep the fallback/baseline bundles and trusted migration inputs. Do not
  overwrite `wake_word_model.me2`, discard v3, or delete the T01–T13 logs.
- If a code/config change is needed to make v5 the explicitly named final
  supervised runtime candidate, keep it narrow, versioned, and reversible.
  Prefer a release manifest or an explicit final-evaluation config over
  changing generic examples or a silent canonical-path replacement.

## Required work

### 1. Inspect and establish the final selection

- Confirm the resolved artifact paths and SHA-256 values from the files, not
  only this prompt. Confirm which configs use v3 and which use v5.
- Compare v3 and v5 using the saved, comparable test reports. Keep the
  operational recall/false-accept outcomes separate from aggregate accuracy.
- Add or update a concise final-selection record that states the selection
  scope, hashes, evaluation split, metrics, known limitations, exact runtime
  config(s), and rollback/reference artifact.
- Ensure documentation consistently distinguishes:
  - dashboard file fixture (v3),
  - final supervised runtime/trial candidate (v5),
  - deployment-ready model (none),
  - selected command-development bundle (CTC).
- Retain the existing evidence separation. Do not rewrite historical reports
  to imply that a failed guardrail was passed.

### 2. Perform a safe, evidence-preserving cleanup

- Start with `git status --short`, `git diff --check`, and a repository search
  of every proposed deletion. Treat all existing unstaged/untracked files as
  user work unless they are an exact, demonstrably regenerable cache or
  compiled by-product.
- The known cleanup history already pruned superseded candidate bundles while
  retaining their reports, predictions, logs, and provenance. Do not repeat
  or broaden that deletion.
- `tests/test_pi5_telemetry.pyc` is an apparent generated bytecode file:
  verify it is not tracked or referenced, remove it if confirmed, and record
  the disposition. Preserve source, docs, models, results, logs, reports,
  raw/user datasets, fixtures, and evidence cited by the deadline documents.
- Do not delete a model, result, log, report, config, or document just
  because it is untracked. For anything ambiguous, list the exact path,
  rationale, references checked, and leave it untouched.
- Update the cleanup ledger only with actual, verified cleanup actions. State
  that existing experiment evidence was deliberately retained.
- Treat the current retained model inventory as intentional: v5 and v3 are
  the dual-wake pair; `wake_word_model.me2`/`.pkl` preserve the fallback and
  upstream provenance; `vcm_ctc_model.me2`/`.pkl` preserve the selected model
  and trusted migration input; and `vcm_model.pkl`/`vcm_baseline_model.me2`
  preserve the command baseline. Do not delete one of these merely because it
  is not the primary model.
- Remove only exact, untracked/generated caches or compiled by-products after
  reference checks. The known `tests/test_pi5_telemetry.pyc` is a candidate
  for removal if it remains untracked and unreferenced. If no obsolete
  candidate bundle remains in the model directories, document that outcome
  instead of inventing a model deletion. Historical metrics, predictions,
  calibration outputs, reports, and redacted trial logs remain evidence and
  are not cleanup targets.

#### Target folder structure and final-artifact cleanup

The operator wants the checkout compact and understandable. Reorganize only
after tracing code/document/config references, keeping this as the visible
working structure:

```text
ME2_v2/
├── wake-word/                 # wake recordings/bootstrap audio and retained wake bundles
├── vcm/                       # command-data cache and retained command bundle
├── src/me2/                   # active frontend, models, runtime, training, evaluation, CLI
├── config/                    # versioned runtime/model settings
├── docs/                      # human guide, decisions, provenance, runbooks
├── results/                   # compact final metrics and retained provenance summaries
├── logs/                      # only current/reproducible runtime or training evidence
└── tests/                     # active regression and safety tests
```

The requested cleanup is authorized only for *exact failed/superseded
artifacts* after a recorded reference trace. Apply these rules:

1. Keep v5 and v3 in the wake-model working set because they form the
   required primary/fault-fallback pair; keep the selected CTC command model.
   Do not delete an active runtime bundle, v3/v5 comparison artifact, or
   required config simply to make `models/` look smaller.
2. Keep source required by the runtime, dashboard, dual-wake path, training,
   evaluation, migration, and active tests. Delete a Python module or test
   file only when `rg` confirms it is unreferenced, it is demonstrably an
   abandoned/failed experimental path, and its removal leaves the focused
   and full regression suite intact. A test that verifies rejection, a model
   fault, or another failure-safe behavior is not a “failed test” to delete.
3. Remove generated caches, bytecode, abandoned smoke outputs, duplicate
   artifacts, and failed candidate *binary bundles* only when they are not
   selected, not a fallback, not a trusted migration input, not needed by a
   reproducible baseline, and not referenced by a retained document/config.
4. Do not delete results, logs, predictions, reports, or source merely
   because the underlying experiment failed. The repository policy requires
   keeping negative evidence and provenance when it supports the selection
   decision or is cited by documentation. First consolidate duplicate
   summaries, then remove only an exact superseded duplicate whose replacement
   is retained and documented.
5. If non-active baseline/provenance models are moved out of a `models/`
   working directory, move them to a versioned `reports/model-artifact-archive-<date>/`
   directory with a manifest, SHA-256, original path, recovery instructions,
   and documentation/config updates. Do not delete them. Normal inference
   must never load archived pickles.
6. Do not remove `docs/`, `config/`, `README.md`, `AGENTS.md`, or project
   metadata merely because they are not shown in the abbreviated tree above.
   The human guide and decision records are final deliverables.
7. Before every deletion or move, write a proposed cleanup manifest with
   exact path, size, reason, references checked, replacement/archival path,
   and recovery method. After the action, update the cleanup ledger and rerun
   `git status --short`, the affected tests, and `git diff --check`.

If the only safe deletion remains generated bytecode/caches, state plainly
that no failed source, test, model, result, or log can be removed without
breaking reproducibility or violating the retained-evidence policy. Do not
make deletions merely to satisfy a directory-size target.

### 3. Document Raspberry Pi 5 package compatibility

Check actual compatibility for the observed target, rather than assuming that
the macOS environment applies:

```text
Target: Raspberry Pi 5 Model B Rev 1.0
OS: Ubuntu 26.04.1 LTS
Architecture: aarch64 / 64-bit
Python: 3.14.4
Declared runtime dependencies: numpy>=1.26, scipy>=1.11, soundfile>=0.12,
torch>=2.2, psutil>=5.9; microphone extra sounddevice>=0.4.6
```

- Inspect current package metadata/release artifacts from the authoritative
  package publishers or PyPI. Record the date checked, exact candidate
  versions, Python 3.14 support, Linux ARM64 wheel availability, and any
  required system libraries.
- Do not claim the existing lower bounds themselves guarantee Python 3.14
  compatibility. `torch>=2.2` and the other open-ended ranges may resolve to
  a newer compatible release, but that must be verified on the Pi or pinned
  in a tested constraints file.
- In particular, verify the currently selected PyTorch wheel supports CPython
  3.14 on Linux ARM64 and is compatible with the target glibc/Ubuntu release.
  Verify NumPy, SciPy, SoundFile, psutil, and sounddevice separately.
- Record whether `sounddevice` requires a usable PortAudio library and whether
  the system needs ALSA/PortAudio development/runtime packages. Treat USB
  ALSA enumeration as hardware readiness only; it does not prove Python audio
  library availability.
- Add a Pi-specific, exact-version constraints/requirements file only when
  every pin is supported by published ARM64/Python-3.14 artifacts. Otherwise
  document the blocker and provide a safe installation-verification command;
  do not guess pins, force source builds, alter the system Python, or install
  packages on the Pi.

### 4. Finish the implementation-plan documentation

Update the implementation-plan/progress documentation so the project can be
handed to the next operator without rediscovering decisions. It must include:

- completed dashboard/automated-test work;
- final supervised v5-selection rationale and the explicit non-deployment
  limitation;
- retained v3 fixture and selected CTC model roles;
- all cleanup actions and retained-evidence policy;
- a package compatibility table for the observed Pi target;
- a sequence of remaining Pi gates: approved transfer/write, isolated venv
  install, hash/config verification, and deterministic file replay; mark live
  microphone, speaker, live dashboard listening, Bluetooth, telemetry, and
  deployment validation as deferred at the operator's request;
- the facts that physical speaker audibility remains unconfirmed, Bluetooth
  was not tested, and no Pi inference timings/quality measurements exist;
- precise release criteria and a clear final state of
  **final-commit-ready documentation, not deployment-ready**.

Do not include a Pi hostname/IP address, credentials, private keys, raw
recordings, or transcripts.

### 5. Add a human-friendly operator guide and glossary

Create or update `docs/LAPTOP_OPERATOR_GUIDE_2026-10-16.md` and link it
prominently from `README.md`. It must be understandable to a student who did
not build the system. Use a plain-English explanation the first time any
technical term appears, and keep the detailed implementation material in the
engineering documents.

Explain the pipeline and why every stage exists:

1. **Audio capture** — the attached microphone supplies a stream of sound.
   The program converts it to the fixed sample rate and mono format expected
   by the models, so a device's native format does not silently change
   results.
2. **Wake window and features** — the program examines short overlapping
   pieces of audio and converts the waveform into compact frequency features.
   A *hop* is how often it advances to the next piece; overlap helps avoid
   missing a word that begins between checks.
3. **Wake model and smoothing** — a wake CNN estimates whether a window
   contains connected “a-teh-ko.” *Smoothing* means requiring consistent
   results across nearby windows, reducing one-window noise triggers.
4. **Dual-wake roles** — v5 is primary; v3 is scored beside it and becomes a
   degraded-mode fallback only after a v5 fault. A disagreement is evidence,
   not permission to trigger a command.
5. **Command capture** — after a valid wake, *pre-roll* preserves sound from
   just before the trigger so an immediate command is not clipped. *Onset
   timeout* limits how long the program waits for speech; *endpoint silence*
   ends capture after speech stops; and the duration cap prevents unsupported
   long commands from being silently shortened.
6. **CTC command decoding and grammar** — the command model produces
   character probabilities. *CTC* is the alignment method that maps many
   sound frames to a shorter text sequence. The *grammar* is the approved list
   of command phrases, preventing an uncertain sound from becoming an
   arbitrary action.
7. **Accept/reject decision** — a result is accepted only when it satisfies
   the decoder and safety policy. Rejection is a deliberate safe outcome for
   ordinary speech, silence, unsupported phrases, and system faults.
8. **Local dashboard action and cooldown** — an accepted mapped command
   changes only simulated dashboard state. *Cooldown* prevents repeated
   firing from the same utterance. Calls, messages, and home-device controls
   remain demonstrations only.

Include a plain-language capability/limitation table:

| Component | What it can demonstrate now | What it cannot establish yet |
| --- | --- | --- |
| v5 wake candidate | File loading/replay and supervised laptop trials | Reliable unattended detection or low false-trigger rate |
| v3 wake fixture | File replay and side-by-side diagnostics | Reliable live detection; its comparable test detected 0/3 wakes |
| CTC command model | Recognition/rejection for phrases represented by its legacy grammar | Robust open-ended speech, semantic slots, or zero OOS acceptance |
| Laptop preflight | Built-in-microphone device visibility and controlled replay | Pi readiness, physical-device control, or model-quality proof |
| Raspberry Pi 5 | Package/import, deterministic model replay, and demo-dashboard reachability | Deployment readiness from deferred live hardware tests |

### 6. Provide a separate laptop test walkthrough

Do not run the following hardware commands yourself. Prepare them in the
human guide for the operator to run one at a time using the built-in MacBook
Pro Microphone. Each command must use a new log/result path and must never
retain raw audio. These are local preflight checks; Pi supplied-WAV and
approved combined live checks remain the operational workflow.

From the repository root, prepare the local environment and choose the
microphone by its stable name:

```sh
cd '/Users/alaineramos/Documents/2025 Files - Mac/School/AI231/ME2_v2'

# Run only if .venv is missing or the local environment needs repair.
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e '.[dev,microphone]'

# Confirm that macOS exposes the built-in microphone without capturing audio.
.venv/bin/python -c 'import sounddevice as sd; print(sd.query_devices())'

# Device visibility only: this does not capture audio.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.health \
  --config config/runtime.example.json \
  --devices \
  --input-device 'MacBook Pro Microphone' \
  --output results/laptop-device-check-20261016.json
```

Document these deterministic, no-microphone checks separately. They prove
that the model files load and make decisions on supplied WAV files:

```sh
# Primary v5 wake file replay.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --wake-word --config config/runtime.example.json \
  --wake-model wake-word/models/wake_word_v5_regularized_seed43.me2 \
  --audio wake-word/data/v2/audition/43fd11c385fc_guide_connected.wav \
  --log-file logs/laptop-v5-wake-file-replay-20261016.jsonl

# v3 file replay for side-by-side comparison.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --wake-word --config config/runtime.example.json \
  --wake-model wake-word/models/wake_word_v3_binary_seed43_r1.me2 \
  --audio wake-word/data/v2/audition/43fd11c385fc_guide_connected.wav \
  --log-file logs/laptop-v3-wake-file-replay-20261016.jsonl

# CTC command-only file replay.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --vcm --config config/runtime.example.json \
  --audio vcm/data/test_sample.wav \
  --log-file logs/laptop-vcm-file-replay-20261016.jsonl

# Full v5 wake-to-command handoff using files, not a microphone.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --interactive --config config/runtime.example.json \
  --wake-audio wake-word/data/v2/audition/43fd11c385fc_guide_connected.wav \
  --command-audio vcm/data/test_sample.wav \
  --threads 2 --allow-unverified-wake-model \
  --log-file logs/laptop-v5-integrated-file-replay-20261016.jsonl
```

Provide these controlled built-in-microphone tests. State the phrase before
each command, start one command only, and stop
a wake-only command with `Ctrl-C` after the intended attempts. Do not change
thresholds/configuration while comparing v5 and v3.

```sh
# Say the connected wake word “a-teh-ko”; v5 only.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --wake-word --config config/runtime.example.json \
  --wake-model wake-word/models/wake_word_v5_regularized_seed43.me2 \
  --input-device 'MacBook Pro Microphone' --allow-unverified-wake-model \
  --log-file logs/laptop-v5-wake-live-20261016.jsonl

# Repeat with the same phrase/device for v3 only.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --wake-word --config config/runtime.example.json \
  --wake-model wake-word/models/wake_word_v3_binary_seed43_r1.me2 \
  --input-device 'MacBook Pro Microphone' --allow-unverified-wake-model \
  --log-file logs/laptop-v3-wake-live-20261016.jsonl

# Canonical gated path: say “a-teh-ko”, then a supported command.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --interactive --config config/runtime.example.json \
  --input-device 'MacBook Pro Microphone' --threads 2 \
  --session-seconds 30 --allow-unverified-wake-model \
  --log-file logs/laptop-v5-gated-live-20261016.jsonl
```

Finally, document the visual dashboard checks:

```sh
# UI-only review: no microphone or model input.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.dashboard \
  --demo --port 8765 --threads 2 \
  --log-file logs/laptop-dashboard-demo-20261016.log

```

The dashboard demo is local UI review only. Use the Pi supplied-WAV and,
after explicit approval, bounded combined live command for operational
validation; never use a laptop dashboard event as model or Pi evidence.

### 7. Verify and prepare a focused final commit

- Run the focused tests affected by changes, then the full safe regression
  command when the environment permits:

  ```sh
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/pytest -q \
    -k 'not http_assets_and_control_require_local_host_and_token'
  .venv/bin/ruff check src tests
  git diff --check
  ```

- If the localhost test is deselected because the sandbox cannot bind
  loopback, report it as deselected—not passed.
- Review the staged diff for unrelated pre-existing user work, credentials,
  raw audio, and bulky generated output. Stage only the final-selection,
  cleanup, compatibility, and documentation changes that are directly needed.
- Before committing, report the exact staged paths and the proposed focused
  commit subject. Commit only if the staged diff is clean and all required
  verification has passed. Do not use a broad `git add .`.

## SSH functionality-test commands

Use the following commands as the staged Raspberry Pi 5 validation runbook.
They are intentionally divided by approval level. Replace only the two local
shell variables; do not place the actual target, credentials, passwords, SSH
keys, or raw recordings in a repository file. Let SSH use its normal
interactive authentication or configured agent; do not use `sshpass`, copy
keys, or disable host-key verification.

### A. Local terminal: define the target and run the read-only preflight

First, obtain the operator's explicit approval to connect and to perform
read-only inventory. In a terminal on the development Mac, from any directory:

```sh
export ME2_PI_USER='YOUR_PI_USERNAME'
export ME2_PI_HOST='YOUR_PI_HOSTNAME_OR_IP'
export ME2_PI_TARGET="${ME2_PI_USER}@${ME2_PI_HOST}"

ssh -o ConnectTimeout=5 "$ME2_PI_TARGET" '
  printf "%s\n" "--- board ---"
  tr -d "\000" </proc/device-tree/model 2>/dev/null || true
  printf "%s\n" "--- architecture / OS / Python ---"
  uname -m
  getconf LONG_BIT
  . /etc/os-release && printf "%s %s\n" "$PRETTY_NAME" "$VERSION_ID"
  python3 --version
  printf "%s\n" "--- available storage / memory ---"
  df -h "$HOME"
  free -h
  printf "%s\n" "--- audio device listings ---"
  id
  arecord -l || true
  aplay -l || true
  printf "%s\n" "--- optional thermal state ---"
  cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || true
  command -v vcgencmd >/dev/null && vcgencmd measure_temp || true
  command -v vcgencmd >/dev/null && vcgencmd get_throttled || true
'
```

Stop if this does not identify a 64-bit Raspberry Pi 5 (`aarch64`) or if the
audio devices expected for the test are absent. This preflight must not start
the microphone, play sound, install packages, create files, or modify groups.

### B. Local terminal: approved Git-based Pi transfer and package check

Run this section only after a separate approval for remote writes, file
transfer, and package installation. The user must first push the reviewed
local changes. Set the remote URL and the exact pushed commit SHA; use a new,
disposable Pi directory and never overwrite an existing checkout, service,
virtual environment, model, or logs.

```sh
export ME2_ROOT='/Users/alaineramos/Documents/2025 Files - Mac/School/AI231/ME2_v2'
export ME2_PI_RUN="/home/${ME2_PI_USER}/me2-v5-final-20261015"
export ME2_GIT_REMOTE='YOUR_REPOSITORY_SSH_OR_HTTPS_URL'
export ME2_GIT_REV='EXACT_PUSHED_COMMIT_SHA'

# Local verification after the user has pushed. It must print the same SHA
# placed in ME2_GIT_REV and show no uncommitted changes intended for the Pi.
cd "$ME2_ROOT"
git rev-parse HEAD
git status --short
git branch --contains "$ME2_GIT_REV"
git ls-remote "$ME2_GIT_REMOTE" | rg "$ME2_GIT_REV"

# Pi transfer: clone and detach at the exact pushed revision. Authentication
# uses the Pi user's existing Git/SSH configuration; do not put a token/key in
# this command, environment variable, or repository file.
ssh "$ME2_PI_TARGET" "git clone --no-checkout '$ME2_GIT_REMOTE' '$ME2_PI_RUN' && \
  git -C '$ME2_PI_RUN' fetch --depth 1 origin '$ME2_GIT_REV' && \
  git -C '$ME2_PI_RUN' checkout --detach '$ME2_GIT_REV' && \
  git -C '$ME2_PI_RUN' rev-parse HEAD && \
  git -C '$ME2_PI_RUN' status --short"
```

On the Pi, create an isolated venv and check that published binary packages
can satisfy the target's CPython 3.14/Linux ARM64 environment. Run the
download and install only after the compatibility record has identified
acceptable exact versions or constraints. `--only-binary=:all:` is deliberate:
stop and document the blocker instead of silently compiling scientific
packages on the Pi.

```sh
ssh "$ME2_PI_TARGET" "cd '$ME2_PI_RUN' && \
  python3 -m venv .venv && \
  .venv/bin/python -m pip install --upgrade pip && \
  mkdir -p logs results wheel-audit && \
  .venv/bin/python -m pip download --only-binary=:all: --dest wheel-audit \
    'numpy>=1.26' 'scipy>=1.11' 'soundfile>=0.12' 'torch>=2.2' \
    'psutil>=5.9' 'sounddevice>=0.4.6' && \
  .venv/bin/python -m pip install --only-binary=:all: \
    'numpy>=1.26' 'scipy>=1.11' 'soundfile>=0.12' 'torch>=2.2' \
    'psutil>=5.9' 'sounddevice>=0.4.6' && \
  .venv/bin/python -m pip install --no-deps -e . && \
  .venv/bin/python -m pip check && \
  .venv/bin/python -m pip freeze --all > results/pi5-v5-package-freeze.txt && \
  .venv/bin/python -c \"import numpy, scipy, soundfile, torch, psutil, sounddevice; print('numpy', numpy.__version__); print('scipy', scipy.__version__); print('soundfile', soundfile.__version__); print('torch', torch.__version__); print('psutil', psutil.__version__); print('sounddevice', sounddevice.__version__); print('torch_cuda', torch.cuda.is_available()); print('audio_devices', len(sounddevice.query_devices()))\""
```

If package download or installation fails, preserve its terminal output and
the `pip` version, then stop. Do not substitute a source build, add `sudo`,
change system Python, or claim Pi compatibility.

### C. Pi: hash, health, and file-based model functionality

After the isolated environment is installed, SSH to the Pi and run these
commands. They do not open a microphone or play audio:

```sh
ssh -t "$ME2_PI_TARGET" "cd '$ME2_PI_RUN' && exec \${SHELL:-/bin/sh}"

pwd

sha256sum \
  config/runtime.example.json \
  wake-word/models/wake_word_v5_regularized_seed43.me2 \
  vcm/models/vcm_ctc_model.me2

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.health \
  --require-pi5 \
  --config config/runtime.example.json \
  --output results/pi5-v5-health.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --interactive \
  --config config/runtime.example.json \
  --wake-audio wake-word/data/v2/audition/43fd11c385fc_guide_connected.wav \
  --command-audio vcm/data/test_sample.wav \
  --threads 2 \
  --allow-unverified-wake-model \
  --log-file logs/pi5-v5-file-replay.jsonl
```

Confirm the printed SHA-256 values match the selection table in this prompt.
The expected classification is a file-replay/health result only; it must not
be called a live wake, speaker, continuous-stream, or deployment result.

### D. Pi: direct individual-model replay only

Do not run a live microphone, speaker, or telemetry command for this deadline
pass. Reuse the Pi SSH shell opened in the remote run directory in step C and
run only the supplied WAV files through the individual models:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --wake-word \
  --config config/runtime.example.json \
  --audio wake-word/data/v2/audition/43fd11c385fc_guide_connected.wav \
  --allow-unverified-wake-model \
  --log-file logs/pi5-v5-wake-file-replay.jsonl

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --vcm \
  --config config/runtime.example.json \
  --audio vcm/data/test_sample.wav \
  --log-file logs/pi5-v5-vcm-file-replay.jsonl
```

Retain the three file-only outputs:

- `results/pi5-v5-health.json`
- `logs/pi5-v5-wake-file-replay.jsonl`
- `logs/pi5-v5-vcm-file-replay.jsonl`
- `logs/pi5-v5-file-replay.jsonl`

Report model load success, resolved paths/hashes, wake score/decision,
command prediction/decision, valid frame counts, and explicit rejection
status. Do not report live latency, queue/drop counts, wake recall, false
activations per hour, microphone health, speaker health, dashboard behavior,
or Pi deployment readiness: none are tested in this direct-model-only scope.

### E. Pi: view the dashboard safely through SSH

After the repository, isolated venv, and direct file-model checks are in
place, run the dashboard **directly on the Raspberry Pi 5**. The development
Mac is only used as a browser screen through an SSH tunnel; it does not run
the dashboard process or models. This is a visual/UI check only. Start the Pi
dashboard in synthetic demo mode without `--autostart`, so it does not load
live audio, open a microphone, or start the wake-to-command listener.

In the existing Pi SSH shell, from the transferred remote run directory:

```sh
cd /home/YOUR_PI_USERNAME/me2-v5-final-20261015

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.dashboard \
  --demo \
  --port 8765 \
  --threads 2 \
  --log-file logs/pi5-dashboard-demo.log
```

Leave that process running. In a separate terminal on the development Mac,
open a local SSH tunnel:

```sh
ssh -N -L 8765:127.0.0.1:8765 "$ME2_PI_TARGET"
```

Then open `http://127.0.0.1:8765` in the Mac's browser. The dashboard binds
only to Pi localhost, and the SSH tunnel preserves that restriction. Inspect
the dashboard's visible simulated state and use its demo controls to check
the 19 local demonstration actions. In particular, call/message actions must
remain visibly simulated; no external call, message, appliance, GPIO, audio
playback, or live recognition may occur.

Stop the dashboard with `Ctrl-C` in its Pi SSH terminal and stop the tunnel
with `Ctrl-C` in the Mac terminal. Retain `logs/pi5-dashboard-demo.log` only
as UI-demo evidence. A visible dashboard proves neither model accuracy nor
Pi deployment readiness.

## Required final report

Return:

1. Selection status and the precise non-deployment caveat.
2. Changed/removed/retained paths, including checksums and cleanup reasons.
3. Pi 5 package compatibility findings, exact package versions/constraints
   when verified, and any installation blocker.
4. Test/lint/diff results, distinguishing passed, deselected, blocked, and
   not run.
5. The staged paths and final commit hash only if a commit was actually made.
6. The remaining explicit approval and hardware-validation gates, with live
   testing clearly marked as deferred at the operator's request.

## Dashboard wake acknowledgment implementation

Implement and verify the dashboard acknowledgment without changing wake
thresholds, grammar, or acceptance policy. After a smoothed active wake
decision, show the accessible visual pulse and exact text **“Yes? Listening
for your command.”**, then continue directly into command capture. Expose
`wake_acknowledgement` through `/api/status` with visible state, message,
timestamp, model role, and audio status. It may appear only for v5 primary,
or for v3 while v5 has a load/inference fault. Clear it after a command
decision, cooldown, stop, or error. Do not show it for a healthy-v5/v3
disagreement, rejected wake, duplicate wake, or system failure.

Keep the chime separate from the existing PortAudio dashboard music player.
Use a generated, short, asynchronous ALSA tone sent through a validated
wired-speaker device with argument-list execution and a strict timeout; do
not save audio or use shell interpolation. Chime failure must not stop capture
or change a dashboard action. Log only the redacted
`acknowledgement_audio_failed` event. Request at most one chime for one valid
wake event.

Create and document a versioned Pi dashboard profile with chime disabled by
default. After the wired speaker arrives and the user separately approves a
live playback diagnostic, run `aplay -l`, select its exact wired
`plughw:CARD,DEVICE` identifier, then enable the chime. Do not assume
`plughw:0,0`, Bluetooth, HDMI, or wireless output. Until that approval, use
the dashboard demo only to confirm the visual acknowledgment.
