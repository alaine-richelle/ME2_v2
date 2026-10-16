# Offline runtime validation and recovery

Use a separate release directory and explicit artifact/configuration paths.
Do not overwrite baseline artifacts. `config/runtime.example.json` is a
**controlled engineering** configuration, not a deployment approval.

For the pause/long-command investigation, see
[CAPTURE_TIMING_2026-10-14.md](CAPTURE_TIMING_2026-10-14.md). The separate
natural-pause profiles are provisional: offline timing failures are repaired,
but a development replay exposed an additional OOS accept and a quiet-command
rejection. Do not promote them based on model-only accuracy. Use
`scripts/laptop_capture_trial.py` for separately approved, bounded microphone
trials with fresh privacy-safe logs; it also supports the preserved baseline.

## Configuration and compatibility

Settings precedence is built-in `RuntimeSettings` defaults, then
`--config runtime.json`, then explicit CLI options. JSON-relative paths resolve
relative to that JSON, not the working directory. Audio/frontend, vocabulary,
grammar and acceptance thresholds come from the loaded artifact and cannot be
overridden independently through this runtime config. Unknown JSON fields and
non-finite settings fail before capture.

Normal loading accepts only safe schema-2 `.me2` bundles, checks archive members
and weights SHA-256, and validates model/configuration contracts. The runtime
does not silently migrate an incompatible artifact or initialize random
weights. New writes use unique sibling temporary files and atomic replacement.
Keep full-file hashes in a trusted release manifest: an embedded hash is not
an authenticity signature.

Explicit trusted legacy migration (never a normal startup step):

```sh
python -m me2.bundle migrate --trusted-legacy trusted-input.pkl separate-output.me2
```

Confirm provenance independently before executing this operation. Do not use
it for downloaded or untrusted pickles.

## Laptop ladder

From the repository, use its validated interpreter (`.venv/.venv/bin/python`
in the recorded host run) or an installed `me2-v2` environment. Set
`PYTHONPATH=src` only for source checkout execution. Installed execution from
another directory requires explicit model/config paths.

| Gate | Exact command | Pass criterion / safe failure action |
|---|---|---|
| Static and imports | `python -m ruff check src tests scripts/runtime_validation.py` | No findings. Fix code, not artifact metadata. |
| Config, safe models, operators | `python -m me2.health --config config/runtime.example.json --output results/tmp/health.json` | `passed_operator_health_not_deployment_approval`; inspect warnings, never treat it as acoustic approval. |
| Hash verification | `python -m me2.health --model vcm/models/vcm_ctc_model.me2 --sha256 b5a358e2815ee2471f29956e63936af87c4359bf28b21c21ab420c21e128eeac` | Hash matches this retained artifact. Obtain another version's hash from its trusted release. |
| Frontend and invalid inputs | `python -m pytest -q tests/test_audio_features.py tests/test_runtime_hardening.py` | Parity, bad metadata/audio/logits, privacy and rollback assertions pass. |
| Baseline file | `python -m me2.app --vcm --vcm-model vcm/models/vcm_baseline_model.me2 --audio vcm/data/test_sample.wav --log-file logs/baseline-check.jsonl` | Loads and returns a structured outcome; this sample is historically rejected. |
| Selected file | `python -m me2.app --vcm --audio vcm/data/test_sample.wav --log-file logs/ctc-check.jsonl` | Retained selected artifact accepts `play_music`; no dispatch. |
| Wake gate, separate files | `python -m me2.app --interactive --config config/runtime.example.json --wake-audio wake-word/data/v2/audition/927f756e547f_guide_connected.wav --command-audio vcm/data/test_sample.wav --log-file logs/gate-check.jsonl` | Wake detection then command result. Not evidence of continuous boundary quality. |
| Replay/recovery | `python -m pytest -q tests/test_streaming.py tests/test_dashboard.py` | Immediate onset pre-roll, timeout, duplicate suppression, overflow/retry, stopped state and non-dispatch assertions pass. Requires loopback permission for one dashboard test. |
| Device format, no capture | `python -m me2.health --devices --input-device 0 --sample-rate 16000 --output results/tmp/devices.json` | Selected input supports mono float32/16 kHz. Re-enumerate indices; check OS microphone permission and cable. No automatic alternate device. |
| Supervised live interaction | `python -m me2.app --interactive --config config/runtime.example.json --input-device 0 --allow-unverified-wake-model` | Only after device check; explicitly experimental with current wake artifact. Ctrl+C/SIGTERM must stop and close streams. Never enable real actions from this smoke check. |
| Full regression | `python -m pytest -q` | All tests pass in a fresh process. |

The actual October 13 run completed file/operator/regression checks.
Sandboxed discovery reported no inputs; unrestricted native discovery hung.
The new bounded health probe returns `device_probe_timeout` after ten seconds
and terminates its child; it never opens a recording stream. Check OS audio
permissions/backend health before retrying. Physical microphone availability
is unresolved. The live interaction row was **not run**.
Silence/OOS rejection tests are synthetic functional tests; the selected
uncalibrated artifact is not assumed to reject all actual silence or speech.

Small batch regression, using existing cache only:

As of 2026-10-15, the calibrated comparison bundle referenced by this
historical command was pruned after its saved metrics failed the promotion
guardrails. Regenerate it with `python -m me2.vcm.calibrate` before including
that third `--model`; the baseline and selected CTC bundles remain available.

```sh
PYTHONPATH=src python scripts/runtime_validation.py \
  --cache-revision /absolute/cache/path/bfca0bbebfe17e00d6a685d1da573615d29ecdce \
  --partitions results/vcm_ctc_partitions.json \
  --model vcm/models/vcm_baseline_model.me2 \
  --model vcm/models/vcm_ctc_model.me2 \
  --model vcm/models/vcm_ctc_calibrated_2026-10-05.me2 \
  --threads 2 --per-label 5 --output results/tmp/new-validation-run.json
```

Requires development dependencies and the **two cached official-train Arrow
shards**, not test/holdout. Missing/corrupt audio fails the run; overlength
commands remain explicit failures. Output refuses an existing filename.
Compare fixed indices, shard/artifact hashes, per-row decisions, accuracy,
F1, coverage, errors and p95; never select thresholds from this regression
sample. Dataset loading does not occur in the normal application or health CLI.

## Privacy-safe diagnostics

Runtime log schema 2 uses UTC ISO timestamps, run ID, component, event name,
Git revision when available, model/grammar/artifact/config identity, utterance
ID, rejection or system-error reason, candidate count and stage timings.
Streaming utterances share IDs across transitions and outcome. INFO does not
write transcripts, slot values, input paths, raw samples or exception messages.
Legacy free-text log calls are represented by a redacted diagnostic event;
bounded typed events carry operational details. Per-window wake debug messages
are not emitted at INFO. Each log file is bounded to 2 MiB plus three backups.

`--debug-sensitive` explicitly enables bounded free-text/trace diagnostics.
Those logs **can contain private text and paths**; keep them local. Older
training/evaluation logs predate this privacy policy and are not safe to share
automatically. Normal inference never stores runtime audio.

For a safe diagnostic bundle, create a new directory and copy only the default
non-debug runtime log and rotations, a fresh `me2-health` JSON, a sanitized
runtime config with local paths removed, the source commit, and artifact hashes.
Inspect files before sharing. Exclude raw audio, models, dataset/cache paths,
debug logs, prior training logs, SSH settings and credentials. Do not archive
the entire repository or home directory.

Log I/O failures close the failing sink and raise `LogWriteError` without the
stdlib fallback that prints original private message arguments. They are fatal,
not retried as microphone faults. Fatal startup errors exit nonzero.
Ordinary rejections are valid non-actionable
results. Lost-sample/device faults invalidate the utterance, release the stream,
discard queued data and retry at 1/2/4 seconds, at most three retries. Exhaustion
stops with an error. Decoder resource failures are system errors, not OOS truth.
The decoder's bounded expansions are not a real-time execution deadline.

## Raspberry Pi 5 release gate — pending actual hardware

No Pi target was connected in this cycle. Do not copy the laptop dependency
versions into an alleged validated Pi lock. The official Raspberry Pi OS
documentation distinguishes 32-bit and 64-bit installations; verify the
installed OS and actual board, not merely the model name on the box:
[Raspberry Pi OS documentation](https://www.raspberrypi.com/documentation/computers/os.html).

On the actual target, record `uname -m`, `getconf LONG_BIT`, `python --version`,
`cat /etc/os-release`, `df -h`, `free -h`, and `arecord -l`. In the candidate
runtime run:

```sh
python -m me2.health --require-pi5 --config /absolute/release/runtime.json \
  --devices --input-device 0 --output /absolute/release/pi-health.json
```

The check requires actual Raspberry Pi 5 identification, 64-bit pointers and
`aarch64`; it reports available governor/temperature/firmware telemetry,
packages and CPU forward results. A listed quantized backend is **not**
operator or accuracy validation. Use `--require-approved-wake` for a release
gate: every current wake artifact fails that gate.

Build the initial dependency candidate in a fresh **target-compatible**
environment with only core project dependencies plus `sounddevice`. Do not
install `[train]` or `[dev]`. After real operator/file/replay/quality checks,
freeze all resolved versions, collect wheels and hashes, and preserve the
exact OS/Python/architecture report. An approved minimal ARM64 lock and
wheelhouse remain a target-dependent deliverable, not something verified here.

An offline install from that approved wheelhouse is:

```sh
python -m pip install --no-index --no-deps --require-hashes \
  --find-links /absolute/release/wheelhouse \
  -r /absolute/release/requirements-pi.lock
python -m pip check
```

Every requirement in the lock must be pinned with its approved wheel hash,
including transitive dependencies. Wheelhouses are platform-specific; see
[pip's repeatable-install guidance](https://pip.pypa.io/en/stable/topics/repeatable-installs/).
Normal ME2 startup does not download/build/train. Install wheels and native
audio libraries during provisioning, not in a service startup hook.

Transfer only source/wheel, explicit artifacts, runtime config, trusted hash
manifest and approved wheelhouse. Exclude dataset caches, training outputs,
raw recordings and laptop environments. Verify each complete artifact via
`me2-health --model ... --sha256 ...` before activation.

For real Pi timing, first run the existing `me2-benchmark`:

```sh
python -m me2.benchmark /absolute/representative.wav \
  --model /absolute/release/command.me2 --threads 2 \
  --warmup 10 --iterations 100 --output /absolute/release/pi-command-benchmark.json
```

Repeat for wake; save source/accent/noise conditions. This benchmark excludes
capture/endpointing and reports post-run RSS, not peak memory. Additionally
collect actual cold process start, sustained peak RSS/CPU/idle cost and
temperature/throttling plus annotated continuous interaction trials using
`docs/PI_EVALUATION_REPORT.md`. Include buffering, endpoint wait and the
completed-command bidirectional lookahead in end-to-end latency. Do not report
the laptop millisecond values as Pi estimates.

Rollback keeps previous source/wheel/environment, runtime config and artifact
hashes in a separate immutable release. Stop the process, select the previous
release/config explicitly, re-run checksum/health checks, then restart. Do not
edit a model's approval flag, overwrite a known-good model, reset the checkout,
or silently fall back to another artifact.
