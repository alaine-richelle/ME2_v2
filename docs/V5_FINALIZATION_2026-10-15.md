# V5 finalization record — 2026-10-15

## Release state

This is **final-commit-ready documentation for supervised engineering
evaluation**, not a production, unattended, or Raspberry Pi deployment
promotion. The safe dashboard demonstration and automated/offline checks are
complete. No wake candidate is deployment-ready.

The 2026-10-15 Raspberry Pi handoff identified an Ubuntu 26.04.1 LTS ARM64
target with CPython 3.14.4, but did not itself run a current Git-based model
validation. A subsequent, exact-commit Pi direct-model and demo check is
recorded below. Earlier hardware-only or host-file evidence remains historical
evidence only. Physical speaker audibility remains unconfirmed, Bluetooth was
not tested, and there are no Pi live-inference quality or timing measurements.

## Retained roles and evidence

| Role | Artifact | SHA-256 | Comparable evidence and disposition |
|---|---|---|---|
| Final supervised runtime/trial wake candidate | `wake-word/models/wake_word_v5_regularized_seed43.me2` | `d7a4898afdc5a5a52fcd305424df4b62582d7bdeea58451d2a90e62646a49222` | Frozen `test` report: 2/3 wake events, 4/48 false accepts, 0.9020 accuracy, 0.6953 macro F1. Primary decision source; retain only for supervised trials. |
| Dashboard fixture and diagnostic/fault fallback | `wake-word/models/wake_word_v3_binary_seed43_r1.me2` | `35ec56088979aea4d16f547dfcda047fac969b8f9d46a60176ee1703d853170b` | Same frozen `test` cohort: 0/3 wake events, 0/48 false accepts, 0.9216 accuracy, 0.6458 macro F1. Score beside v5 for diagnostics and use only after a v5 load/inference fault; never accept a v3 disagreement as an OR rule. |
| Command-development model | `vcm/models/vcm_ctc_model.me2` | `b5a358e2815ee2471f29956e63936af87c4359bf28b21c21ab420c21e128eeac` | Matched official `test`: 0.8911 accuracy, 0.8682 macro F1, 20/76 OOS false accepts, 316/4,367 in-scope rejections. Retain for command development, not Pi approval. |

The wake reports are small synthetic clip cohorts (3 positive and 48 negative
rows, one independent speaker). They do not provide false-activations-per-hour
or live-stream evidence. The selected CTC uses legacy grammar and an
uncalibrated acceptance policy. Do not set `deployment_ready=true`, relax
thresholds, replace the v3 dashboard fixture, or overwrite fallback/baseline
bundles or protected T01–T13 evidence.

## 2026-10-16 wake training-readiness check

No wake candidate was trained, calibrated, evaluated, promoted, or written in
this follow-up. The current frozen v3 and v5 manifests passed structural
integrity, file, and split-leakage checks, but both failed the required review
gate:

| Frozen manifest | Rows | Approved connected positives | Speech-review coverage | Valid for approved training |
| --- | ---: | ---: | ---: | --- |
| `wake-word/data/v3/synthetic/manifest.csv` | 408 | 0 | 0% | No |
| `wake-word/data/v5/command-negatives/manifest.csv` | 826 | 0 | 0% | No |

All 24 v3 positives and all 50 v5 positives remain pending connected-delivery
and pronunciation review. The v5 manifest includes 116 non-synthetic
command-negative speakers, but those negatives do not supply reviewed
connected-wake evidence. Since there are no trainable approved positives,
`valid_for_training=false` for both manifests. Their frozen assignments,
thresholds, v5/v3 bundles, calibration/test partitions, metrics, and
provenance remain unchanged.

The next compliant training attempt requires independently reviewed
complete-`a-teh-ko` positives, explicit accepted connected-delivery status,
and a preserved partition plan. Only then may a new versioned candidate be
trained, selected on validation, calibrated on calibration, and read-only
evaluated on the frozen test partition. It must remain unpromoted unless it
meets every comparable rejection, compatibility, and deployment guardrail.

## Frozen supervised profile and rollback

`config/runtime.v5-final-supervised-20261015.json` is the explicit,
reversible single-wake supervised profile. It has the one-second-onset
settings of `config/runtime.example.json`, points to v5 and the selected CTC
bundle, and writes only a distinct runtime log. Its SHA-256 is recorded after
local verification as
`9bca207c37b6bea8abb102be38ef876f7a8364c5e7fed4db3e95493c8130105e`.
`config/runtime.laptop-dual-wake.json` is the separate, explicit dual-wake
profile for supervised laptop evaluation. It is not a default-path replacement
or deployment configuration.

Rollback/reference artifacts remain:

- `config/dashboard.json` and `config/dashboard.natural-pause.json` retain
  the v3 dashboard fixture.
- `wake-word/models/wake_word_model.me2` remains the legacy fallback.
- `config/runtime.example.json` remains the generic runtime example.

The dual-wake policy is fail-closed: both models score each prepared wake
window using their own embedded threshold and smoothing policy. A healthy v5
non-wake remains non-wake even if v3 disagrees. V3 becomes active only after a
v5 load or inference/system fault; two failures do not capture a command or
dispatch an action. Redacted logs, CLI state, dashboard status, and health
output expose the primary/fallback roles and degraded state.

## Raspberry Pi package compatibility

Publisher artifact metadata was checked on 2026-10-15 using each package's
PyPI JSON release endpoint and the PyTorch CPU simple index. The exact direct
candidate set is in `config/requirements.pi5-cp314-aarch64.txt`; it is a
direct-requirements record, not a transitive lock.

| Dependency | Exact candidate | CPython 3.14 / Linux ARM64 artifact evidence | Pi-side prerequisite or remaining check |
|---|---|---|---|
| NumPy | 2.5.4 | `cp314-cp314-manylinux_2_27_aarch64.manylinux_2_28_aarch64` | Target `ldd --version` must satisfy the wheel's glibc baseline. |
| SciPy | 1.18.1 | `cp314-cp314-manylinux_2_27_aarch64.manylinux_2_28_aarch64` | Same glibc check; do not allow a source build. |
| SoundFile | 0.14.0 | `py2.py3-none-manylinux_2_28_aarch64` | Verify import and libsndfile availability in the isolated venv. |
| psutil | 7.2.2 | `cp36-abi3-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64`; ABI3 is compatible with CPython 3.14 | Verify import on the target. |
| sounddevice (microphone extra) | 0.5.6 | `py3-none-any` | A usable PortAudio runtime is still required; verify `sounddevice` import and PortAudio version without opening a stream. |
| PyTorch CPU | 2.14.1+cpu | `cp314-cp314-manylinux_2_28_aarch64` on the PyTorch CPU index | Verify the target glibc is at least 2.28 and import this CPU wheel in the isolated venv. |

The open-ended project dependency ranges do not themselves guarantee this
environment. The published artifact check supports these direct pins. The
2026-10-16 direct Pi check completed a binary-only download and isolated
installation at the exact Git revision, with `pip check`, non-audio imports,
and `pip freeze --all` retained in its Pi-local evidence. Do not source-build
packages, alter system Python, or substitute a different wheel silently.

## 2026-10-16 measured Pi direct check

GitHub `master` resolved to
`050f5c34f5f6930fa0a4d79e13fc9202f04c03a5`, which was checked out detached
in `/home/rpiadmin/me2-pi5-20261016-050f5c3` without modifying the prior Pi
checkout. The target was a real 64-bit Raspberry Pi 5 on Ubuntu 26.04.1 LTS,
glibc 2.43, and CPython 3.14.4. The direct pinned package set installed from
binary wheels only; `pip check` and NumPy, SciPy, SoundFile, psutil, and Torch
imports passed.

The dual-wake health check loaded the v5 primary, v3 diagnostic/fault fallback,
and CTC bundle with their recorded SHA-256 identities, returning
`passed_operator_health_not_deployment_approval`. Three tracked cached
command-error WAVs completed CTC replay and V5/V3 non-wake checks. CTC
accepted `pause`, `next`, and `color`; V5 and V3 rejected each clip as a wake
event. Those fixtures are diagnostic error examples, not a held-out accuracy
evaluation.

No tracked approved connected-`ateko` WAV was available, so the positive
wake-to-command replay was not run. V3 fault-only activation was also not run:
it requires a V5 load/inference fault in the continuous pipeline, and no
fault injection or live stream was authorized. In a headless SSH session,
`sounddevice` could not initialize PortAudio because PulseAudio was absent;
this is a deferred live-audio limitation, not an offline model/demo blocker.

The loopback-only `me2.dashboard --demo` server returned its local API status
and was exposed only by an SSH local forward. Demo mode loaded no microphone or
recognition models. The exact Pi evidence remains under
`/home/rpiadmin/me2-pi5-20261016-050f5c3/results/`, `logs/`, and
`wheel-audit/`; it was not pushed and does not change deployment status.

Publisher references:

- `https://pypi.org/pypi/numpy/2.5.4/json`
- `https://pypi.org/pypi/scipy/1.18.1/json`
- `https://pypi.org/pypi/soundfile/0.14.0/json`
- `https://pypi.org/pypi/psutil/7.2.2/json`
- `https://pypi.org/pypi/sounddevice/0.5.6/json`
- `https://download.pytorch.org/whl/cpu/torch/`
- `https://python-sounddevice.readthedocs.io/en/0.5.6/installation.html`

## Remaining Pi gates — all deferred pending operator approval

1. Obtain explicit approval for a read-only connection; confirm board, ARM64,
   Python, storage, audio enumeration, and `ldd --version`.
2. Obtain a separate approval for remote writes and package installation.
   The user must first push the reviewed local changes themselves and provide
   a credential-free remote URL, the containing branch or tag, and the exact
   full pushed commit SHA. Confirm that the remote is reachable and that the
   branch/tag resolves to that commit. Then clone into a new Pi run directory,
   fetch that SHA, and check it out detached; log the remote, requested
   branch/tag, requested SHA, and checked-out SHA. Do not transfer a tar/ZIP
   file or a local working tree. Required configs, v5/CTC bundles, and replay
   fixtures must be available from that reviewed revision or an explicitly
   approved published artifact source; otherwise stop rather than copying
   untracked local files. Create an isolated venv only after this gate.
3. Run `pip download --only-binary=:all:` and install only the direct,
   CPython-3.14/Linux-ARM64 candidates in
   `config/requirements.pi5-cp314-aarch64.txt`. Install ME2 itself with
   `--no-deps` after those candidates are verified. Record `pip check`,
   imports, the PortAudio version, and `pip freeze --all`.
4. Verify the frozen config and all three artifact hashes on the target; run
   individual v5 wake and CTC file replays, then one supplied-WAV integrated
   replay. These are deterministic file tests only.
5. Review the dashboard in `--demo` mode without `--autostart` or listening.
   The demo uses synthetic inputs and does not establish model or microphone
   behavior.

Do not run a live microphone, speaker, live dashboard-listening, Bluetooth,
continuous-stream, telemetry, or deployment-validation session in this
deadline pass. Those items are deferred, not passed or failed.

Use the approval-separated command forms in
`PI5_SSH_SAFE_VALIDATION_2026-10-15.md`, substituting only an operator-supplied
target at execution time. The final Pi commands must use the frozen config,
verify the detached source identity and artifact hashes before replay, and
invoke the dashboard only as
`python -m me2.dashboard --demo` (without `--autostart`).

## Cleanup and completion criteria

The sole finalization cleanup action was removal of
`tests/test_pi5_telemetry.pyc`, an ignored, untracked, unreferenced generated
bytecode file. Models, datasets, fixture sources, metrics, predictions, logs,
reports, historical records, and user work were retained. See
`CLEANUP_2026-10-06.md` for the verification record.

## Local finalization verification

- The new runtime profile parsed successfully through
  `me2.runtime_config.load_runtime_defaults`.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/pytest -q -k 'not
  http_assets_and_control_require_local_host_and_token'` completed with
  **172 passed, 1 deselected**. The deselected localhost-only test is not
  counted as passed.
- `.venv/bin/ruff check src tests` and `git diff --check` passed.
- No path was staged or committed. Existing unrelated worktree changes remain
  untouched.

The release remains not deployment-ready until separate comparable wake/OOS
evidence, calibrated command rejection, approved live Pi trials, sustained
stream measurements, and physical audio validation meet documented guardrails.

The student-facing, supervised laptop microphone/earphone procedure is
`LAPTOP_OPERATOR_GUIDE_2026-10-16.md`. It is deliberately separate from the
Pi direct-model/demo-only runbook and no hardware command was run while
preparing it.
