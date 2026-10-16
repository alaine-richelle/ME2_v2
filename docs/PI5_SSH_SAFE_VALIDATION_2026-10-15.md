# Raspberry Pi 5 validation handoff — 2026-10-15

## Final v5 finalization addendum

This document retains historical Pi activity and its evidence; it is not a
claim that the current target is ready. The final v5 handoff identifies an
Ubuntu 26.04.1 ARM64 CPython 3.14.4 target without an ME2 runtime or model
checkout in the inspected locations. No Pi command was executed during final
v5 finalization, and there are no current Pi inference quality or timing
measurements. Hardware-only capture, ALSA enumeration, host-file replay, and
software playback attempts do not substitute for model validation.

V5 is the final supervised runtime/trial candidate, not a deployment-ready
model: its comparable frozen test result is 2/3 wake events and 4/48 false
accepts, above the 2/48 guardrail. V3 remains the dashboard file-replay
fixture, and CTC remains the command-development bundle. Before any Pi work,
use the frozen config and exact package candidates in
`V5_FINALIZATION_2026-10-15.md`; the configuration and artifact hashes must be
verified on the target.

Only after separate approvals may the next operator perform a read-only
preflight, a disposable detached Git checkout/isolated-venv installation, binary package and
import verification, individual model replay, one supplied-WAV integrated
replay, and a visual `python -m me2.dashboard --demo` review without
`--autostart`. Live microphone, speaker, dashboard listening, Bluetooth,
telemetry, continuous-stream, and deployment-validation activity are deferred
at the operator's request. Physical speaker audibility remains unconfirmed.

## 2026-10-16 Git-based Pi direct-model and demo result

This is a **passed direct file/model and demo-dashboard check**, not a live
recognition, deployment, or release-promotion result. GitHub
`refs/heads/master` resolved to
`050f5c34f5f6930fa0a4d79e13fc9202f04c03a5`. That exact commit was checked
out detached in the new mode-`0700` directory
`/home/rpiadmin/me2-pi5-20261016-050f5c3`; the older
`/home/rpiadmin/me2-pi5-20261015` directory was preserved.

The target was Raspberry Pi 5 Model B Rev 1.0 on Ubuntu 26.04.1 LTS,
`aarch64`, CPython 3.14.4, and glibc 2.43. The isolated venv downloaded only
binary candidates from `config/requirements.pi5-cp314-aarch64.txt`, installed
ME2 with dependency resolution disabled, and passed `pip check`. The audited
runtime versions were NumPy 2.5.4, SciPy 1.18.1, SoundFile 0.14.0, psutil
7.2.2, sounddevice 0.5.6, and CPU-only PyTorch 2.14.1+cpu. Non-audio imports
passed.

The final v5 runtime config hash was
`9bca207c37b6bea8abb102be38ef876f7a8364c5e7fed4db3e95493c8130105e`;
the dual-wake config hash was
`f3b6838d64a0db519032561b58cdc2a541b25cb6eb1b08317a171f9e864d6ff6`.
The v5, v3, and CTC bundle SHA-256 values matched the retained identities in
the finalization record. `me2.health --require-pi5` loaded all three bundles,
reported an actual 64-bit Raspberry Pi 5, and returned
`passed_operator_health_not_deployment_approval`.

Three tracked WAVs were available, all documented cached command-error
examples rather than approved `ateko` clips. Each CTC replay completed and
accepted, respectively, `pause`, `next`, and `color`; this reproduces
command-error-fixture behavior and is not an accuracy claim. V5 and V3 each
rejected every one of the same command-only clips as a wake event. The
individual CLI replays and their structured logs are direct file checks only.

| Gate | Outcome | Interpretation |
| --- | --- | --- |
| V5 and V3 negative wake replay | Passed: all three command-only fixtures rejected | No false wake was observed on these three non-wake fixtures; this is not a wake-recall measurement. |
| CTC command-error replay | Passed execution; observed accepted intents recorded | The supplied files are known error examples, so their decisions are diagnostic only. |
| Positive wake-to-command file replay | Not run | No suitable tracked, approved connected-`ateko` WAV exists in the exact detached revision. |
| V3 fault-only fallback activation | Not run | It requires a V5 load/inference fault in the continuous pipeline; neither a fault injection nor live stream was authorized. |
| Headless `sounddevice` import | Deferred live-audio limitation | PortAudio could not initialize because PulseAudio was unavailable in the noninteractive SSH session. No audio-service change, microphone capture, or workaround was attempted. |

The Pi demo dashboard started without `--autostart`, listened only on
`127.0.0.1:8765`, and returned its local status API with `mode=demo`, no
loaded recognition models, and no microphone prediction. It was made available
only through the normal local tunnel:

```sh
ssh -N -L 8765:127.0.0.1:8765 "$ME2_PI_TARGET"
```

Pi-local evidence is retained under
`/home/rpiadmin/me2-pi5-20261016-050f5c3/results/`,
`logs/`, and `wheel-audit/`, including source identity, binary-wheel manifest,
`pip freeze --all`, import limitation, health reports, direct replay summaries,
redacted JSONL logs, and dashboard status responses. These generated files are
not source changes and were not pushed.

## Wake acknowledgment implementation — pending a new reviewed Pi revision

The direct Pi result above predates the dashboard wake-acknowledgment source
change. It remains valid only for the cited detached commit and does not
verify this new feature. Before a Pi review, commit and push the reviewed
change, then clone that exact new commit into a fresh detached run directory;
do not overwrite the existing evidence directory.

`config/dashboard.pi5-dual-wake.json` is the versioned Pi profile for the
feature. It uses the direct `arecord` ALSA microphone backend to avoid the
headless SSH PulseAudio limitation. Its `wake_acknowledgement` chime is
disabled by default. A valid smoothed v5 wake, or a v3 wake only while v5 has
a load/inference fault, produces the visible **“Yes? Listening for your
command.”** pulse and starts normal command capture. A healthy v5/v3
disagreement, rejected wake, duplicate event, and system failure do not
acknowledge or chime.

When the wired speaker is physically available and an operator separately
approves live playback, run `aplay -l`, select the confirmed wired
`plughw:CARD,DEVICE` entry, then set that exact value plus `"enabled": true`
in the profile. Do not infer the device from card number, use Bluetooth/HDMI,
or enable it as part of the file-only deadline check. The chime is generated
in memory and sent to `aplay` with argument-list execution, a short timeout,
and no saved audio. Playback failure is non-fatal and is logged only as the
redacted `acknowledgement_audio_failed` event.

## Status and release decision

The operator supplied the reachable target and authorized a read-only
preflight. The first strict-host-key attempt failed authentication. After the
operator supplied a password and authorized retry, the read-only inventory
ran. Do not put the supplied target or credentials in this report. Machine-
readable findings are in `results/pi5_readonly_preflight_2026-10-15.json`.

Observed target: Raspberry Pi 5 Model B Rev 1.0, Ubuntu 26.04.1 LTS,
`aarch64`/64-bit, Linux kernel `7.0.0-1020-raspi`, system Python 3.14.4.
The system interpreter reports `me2-v2`, PyTorch, NumPy, SciPy, SoundFile,
psutil and sounddevice as not installed. No ME2 checkout or venv was found at
the two checked conventional home paths. A separate private deployment was
later created under the operator account; system Python remains untouched.

The board has four Cortex-A76 CPUs, 7.7 GiB RAM (7.3 GiB available), no swap,
and 29 GiB root storage (25 GiB free). Thermal sysfs read 41,350 m°C.
`vcgencmd` could not open `/dev/vcio`; throttling is unavailable. After the
operator reported the speaker was powered on, a second read-only check found
kernel ALSA cards for a USB-Audio adapter and two HDMI outputs, `/dev/snd`
nodes, and the `speaker-test` executable. The first `arecord -l` and
`aplay -l` checks reported “no soundcards found.” Inspection showed that the
login account was not in the `audio` group and the nodes were `root:audio`
mode `0660`, consistent with the access failure.

The operator then explicitly authorized the single remote change
`sudo usermod -aG audio rpiadmin`. It completed. A fresh SSH login verified
that the account now has supplementary `audio` membership; `arecord -l` and
`aplay -l` then listed the USB-Audio capture and playback device. The group
membership is the only system-level change. Direct `arecord` microphone
capture was subsequently verified at 48 kHz.

After separate approval for one playback trial, this command was attempted:

```sh
timeout 4s speaker-test -D plughw:0,0 -t sine -f 440 -c 1 -l 1
```

The attempt occurred before the group membership change. It failed before
opening playback with `Cannot get card index for 0` and `Playback open error
-2, No such file or directory`; no tone was emitted. There was no microphone
capture, audio file, or saved trial log.

After the audio-group change was verified in a fresh login, the operator gave
separate approval for one retry of the same command. The retry exited with
status 0, opened `plughw:0,0`, configured 48-kHz mono S16_LE playback, and
streamed a 440-Hz sine to the USB-Audio device. This confirms the ALSA
software playback path. The operator reported not hearing the tone;
Bluetooth playback/pairing was not tested. No audio file or log was created.

At the operator's direction, a USB microphone hardware-only capture was then
run:

```sh
timeout 10s arecord -D plughw:0,0 -f S16_LE -r 48000 -c 1 -d 8 -V mono -t raw /dev/null
```

It completed with status 0 using USB-Audio card 0/device 0 at 48-kHz mono
S16_LE. The live ALSA level meter changed during the eight-second capture.
Captured bytes went directly to `/dev/null`; no raw audio was saved. This
verifies only the tested microphone capture path. The meter is not calibrated
and cannot establish what words were spoken or recognized.

## Isolated runtime deployment and file-only health

After explicit approval, the following bundle was uploaded to the private
directory `/home/rpiadmin/me2-pi5-20261015` (mode `0700`), and its SHA-256 was
verified before extraction:

```text
7826af441305ac951aaf1fae4a970d7f7480865d2810fea904b59e7d1be396fa
```

It contained `src/me2`, `pyproject.toml`, the baseline and Pi candidate
configs, and only the selected wake/CTC artifacts. It contained no datasets,
raw recordings, or Python bytecode caches. An isolated venv was created at
`/home/rpiadmin/me2-pi5-20261015/.venv`; no system package was changed. The
runtime versions installed there are Python 3.14.4, `torch 2.14.1+cpu`,
NumPy 2.5.4, SciPy 1.18.1, SoundFile 0.14.0, psutil 7.2.2, and sounddevice
0.5.6. The default PyPI resolver initially selected CUDA dependencies; that
attempt was canceled before package installation. PyTorch was then explicitly
installed from its CPU-only wheel index.

The artifact-only check passed with status
`passed_operator_health_not_deployment_approval`. It loaded the exact selected
bundles and ran synthetic-zero forwards with two CPU threads, without opening
a microphone or speaker stream:

| Measurement | Raspberry Pi 5 result |
| --- | ---: |
| Wake warm inference | 5.20 / 4.05 ms |
| CTC command-prediction warm path | 295.23 / 298.29 ms |
| Health-process RSS | 433,553,408 bytes |
| Sysfs temperature at check | 44.65°C |

These are brief synthetic-input health timings, not a sustained or live
pipeline benchmark. Firmware throttling remains unavailable because
`vcgencmd` cannot open `/dev/vcio`.

The health run with `--devices` failed while initializing PortAudio:
`PulseAudio_Initialize: Can't connect to server`. This is distinct from the
successful direct ALSA capture: PortAudio's PulseAudio host API cannot connect
to a server in this SSH session. No microphone stream was opened by health.
The current Pi source therefore cannot run the app's default `sounddevice`
capture path.

An explicit `arecord` backend and matching Pi config were implemented locally
and passed the local regression suite (172 passed; one localhost test
deselected because the sandbox blocks loopback binding). After separate
approval, update archive
`2f39869efd78fbefe5f4752760c560d64485a6bda9b69f675bed48f740400b11` was
uploaded, hash-verified on the Pi, and extracted into the isolated deployment.
It contained exactly `src/me2/app.py`, `src/me2/health.py`,
`src/me2/runtime_config.py`, `src/me2/streaming.py`, and
`config/runtime.pi5.natural-pause.json`. No system packages or services were
changed.

The updated health check passed with ALSA device listing and synthetic-zero
model forwards only. It selected backend `arecord`, device `plughw:0,0`, and
found `card 0: Audio [AB13X USB Audio], device 0: USB Audio [USB Audio]`.
`device_format_check` is `not_run_no_capture`: this health check did not open a
microphone stream. The parsed Pi profile is pre-roll 250 ms, onset timeout
3,000 ms, endpoint silence 800 ms, and maximum command duration 5,000 ms.
Wake warm inference measured 4.99/3.95 ms; CTC warm prediction measured
299.25/297.71 ms; health-process RSS was 432,574,464 bytes; sysfs temperature
was 44.1°C. These remain synthetic-input smoke measurements, not live or
sustained performance. The updated report is
[`pi5-health-alsa-20261015.json`](../results/pi5-health-alsa-20261015.json).
The earlier setup and artifact-only reports remain
[`pi5-health-setup-20261015.json`](../results/pi5-health-setup-20261015.json)
and
[`pi5-health-artifacts-only-20261015.json`](../results/pi5-health-artifacts-only-20261015.json).

The initial full-inventory command returned status 255 after the
`vcgencmd` errors; subsequent audio inventory, fresh-login verification, and
health runs completed. The first speaker command failed before playback.
Remote changes were the authorized audio-group membership and isolated
user-home runtime deployment/venv. No system packages, services, or raw audio
files were changed or retained. The approved tone was streamed to ALSA, but
physical audibility is unconfirmed. A bounded microphone capture completed to
`/dev/null`, without ME2 inference.

## Approved live Pi trial 01

After separate approval, one supervised 30-second recognition-only session
ran on the USB-Audio input via the deployed `arecord` backend at 16 kHz mono.
The selected config was `config/runtime.pi5.natural-pause.json` (250-ms
pre-roll, 3,000-ms onset timeout, 800-ms endpoint silence, 5,000-ms maximum);
the retained model artifacts and decoder/rejection policy were unchanged.
The app was allowed to use the explicitly unapproved wake model only for this
controlled trial. No action dispatcher was invoked and no raw audio was
saved.

The stream opened and processed 480,000 samples over 30.13 seconds. There
were 94 wake windows, no callback errors, and no dropped queue chunks. The
highest wake window predicted the `ateko` class at 0.6708. The bundle's
probability threshold was not the blocking factor; its smoothing policy
requires two positive windows out of three, and only one positive window
occurred. Consequently there was no wake event, no command capture, and no
CTC decoder result. The requested pause and `play music` command therefore
were not evaluated. Because raw audio was not retained, the exact spoken
content cannot be independently verified from this log.

Wake-pipeline processing p95 was 6.65 ms and stream-feed p95 was 8.53 ms,
below the configured 250-ms hop for this short run. This is an audio-path
smoke result, not a wake-quality, false-activation-rate, or sustained-
stability claim. The redacted structured log is
[`pi5-live-trial-01.jsonl`](../logs/pi5-live-trial-01.jsonl).

After a separate retry approval, live trial 02 used the same config,
artifacts, thresholds, 16-kHz input, and 30-second bound. An initial SSH
launch attempt disconnected before starting the app; a read-only reconnect
then succeeded, and the actual trial ran normally. It processed 480,000
samples over 30.13 seconds, again with 94 wake windows, zero callback errors,
and zero queue drops. This run had zero `ateko`-class windows; the highest
class score was 0.1074 and was not `ateko`. There was no wake trigger, command
capture, or command result. The spoken content cannot be confirmed because
audio was not saved. Wake-pipeline p95 was 6.17 ms and stream-feed p95 was
7.97 ms. These two live attempts do not validate recognition or the
pause/command handoff. The trial 02 log is
[`pi5-live-trial-02.jsonl`](../logs/pi5-live-trial-02.jsonl).

After the operator reported missing the prior GO cue while switching tabs,
trial 03 repeated the same approved 30-second sequence and frozen settings.
The operator was explicitly told to stay on this chat, then given READY/GO
after the Pi stream opened. The run processed 480,000 samples over 30.13
seconds with 94 wake windows, no callback errors, and no queue drops. There
were zero `ateko`-class predictions; the highest class score was 0.3569, but
the predicted class was not `ateko`. No wake or command event occurred. The
logs cannot verify whether the intended phrase reached the microphone, since
raw audio was not saved. Wake-pipeline p95 was 6.73 ms and stream-feed p95
was 8.19 ms. The result is
[`pi5-live-trial-03.jsonl`](../logs/pi5-live-trial-03.jsonl).

## Approved microphone level-only check

The operator separately approved an eight-second `arecord` level-meter
check at 16 kHz mono to `/dev/null`, without inference or saved audio. ALSA
opened successfully and exited with status 0. The displayed meter stayed
mostly at 0%, with brief 1% and 4% readings. This is low activity on the
meter, but it is not speech recognition or calibrated SPL/RMS measurement.
Because the GO cue was delivered after recording had begun and overlap with
the operator's sentence cannot be confirmed, this attempt is **inconclusive
about the operator's speech level**. Do not infer that the microphone or
model is defective from this check alone.

After the operator approved a staged retry, SSH was established while the
microphone remained inactive; capture started only after the operator replied
READY. GO was then given after ALSA confirmed 16-kHz mono capture had begun.
Over the following interval the meter rose from approximately 0–1% to
15–17%, then returned to 0%. The 15-second capture exited successfully to
`/dev/null`; no inference ran and no audio file was saved. This confirms a
noticeable input signal overlapped the cued interval, but the uncalibrated
meter cannot verify the spoken words or establish model recognition accuracy.

The retained wake models remain `deployment_ready=false`. On the shared small
synthetic test cohort, v3 detected 0/3 wake events with 0/48 false accepts;
v5 detected 2/3 with 4/48 false accepts. The CTC development bundle scored
89.11% intent accuracy / 86.82% macro F1 on the reproduced matched command
test, but made 20/76 OOS false accepts and uses legacy uncalibrated
acceptance. Those figures are not Pi, live-stream, or dashboard-action
evidence. See [the model-selection record](MODEL_SELECTION_2026-10-15.md)
and [capture timing report](CAPTURE_TIMING_2026-10-14.md).

## Approved wake-only Pi trial 04

After the operator replied READY, the isolated Pi deployment was reconnected
over SSH and the selected wake bundle was loaded. A 15-second direct ALSA
capture (`plughw:0,0`, 16-kHz mono S16_LE) completed successfully. The wake
model processed 240,000 samples as 53 overlapping 2-second windows at a
250-ms hop. No raw audio was saved; the CTC model was not loaded, and no
dispatcher ran. This check captures first and then scores the in-memory
windows, so its per-window latency is not a live streaming end-to-end result.

All 53 windows had `non_wake` as the predicted class; there were zero
`ateko`-class windows and zero smoothed wake events. The maximum `ateko`
posterior returned by the wake adapter was 0.0580. Although this exceeds the
bundle's numeric threshold (0.000077435), the production gate also requires
the winning class to be `ateko`; that class condition failed. The configured
2-of-3 smoothing therefore correctly emitted no event. Per-window
preprocessing-plus-model p95 was 4.64 ms. ALSA exited 0, all 15 seconds were
received, and no samples clipped. Because audio was intentionally ephemeral,
the spoken content cannot be confirmed from this result.

This trial does not validate wake recall, pronunciation, false activations
per hour, command capture, or Pi streaming stability. The operator later
confirmed that the GO interval was missed and the target was not attempted.
Therefore classify trial 04 as **missed-cue/inconclusive for recognition**;
do not count its all-`non_wake` output as a wake false rejection. The
machine-readable summary is
[`pi5-wake-only-trial-04-20261015.json`](../results/pi5-wake-only-trial-04-20261015.json).

## Retry attempt: Pi wake-only trial 05 (uncued)

The operator replied READY TO RETRY, but the 30-second capture completed
before a GO cue was delivered. This was an execution/protocol failure on the
assistant side. The user was not prompted to speak, so this is **not** a wake
recognition attempt and must not be scored as a false rejection. The technical
path received 480,000 samples over 30 seconds from the same ALSA device;
no clipping was observed and no raw audio was saved. All 113 scored windows
predicted `non_wake`, with maximum wake posterior 0.0709, and no smoothed
event. Since the capture was uncued, these model outputs do not establish
anything about recognition of the target. The CTC model and dispatcher were
not used. See
[`pi5-wake-only-trial-05-20261015.json`](../results/pi5-wake-only-trial-05-20261015.json).

The one-second runtime profile stays the baseline. The three-second onset
profile is an unpromoted candidate. Do not relax wake/command thresholds or
change the selected model to improve apparent live success.

## Approval gates

Before another remote command, the operator must provide:

- The Pi's reachable hostname or IP address.
- The SSH username (or confirmation that the existing SSH alias contains it).
- Explicit approval for any Pi file transfer or other remote write.

The initially approved bundle transfer, isolated venv installation, and
separately approved ALSA-backend update are complete. No service or
system-package change was made. Request approval separately before each
additional live recognition microphone trial, describe the exact phrase and
capture duration, and provide a clear “GO” cue. Use only a mock/local
dispatcher; do not place calls, send messages, or control physical devices.

Never put credentials, private keys, or copied SSH configuration in the repo.
Use the operator's already-configured SSH agent or host alias. Do not use
`sshpass`, print environment secrets, or disable host-key checking.

## Frozen candidate identity

Before comparing laptop and Pi behavior, select one exact config and record
its SHA-256 plus both artifact hashes. The currently retained runtime example
uses:

| Component | Exact bundle | SHA-256 |
| --- | --- | --- |
| Wake (runtime CLI profile) | `wake-word/models/wake_word_v5_regularized_seed43.me2` | `d7a4898afdc5a5a52fcd305424df4b62582d7bdeea58451d2a90e62646a49222` |
| Command | `vcm/models/vcm_ctc_model.me2` | `b5a358e2815ee2471f29956e63936af87c4359bf28b21c21ab420c21e128eeac` |

`config/runtime.example.json` is the one-second-onset comparison profile.
`config/runtime.onset-3s.json` changes onset only and is a laptop replay
candidate. `config/runtime.pi5.natural-pause.json` is a separate 3-second /
800-ms-endpoint profile and must not be described as the same candidate:
endpointing differs. The baseline can be file-tested now; the natural-pause
profile must first pass an approved live laptop trial using the same model
hashes, decoder, rejection policy, and capture settings intended for Pi.
Freeze the approved config by copying it to a new uniquely named release
config and recording its hash; never overwrite the examples during a test.

The v5 bundle is not approved for unattended microphone use. Its CLI live
override is for supervised engineering trials only and must not be mistaken
for deployment approval.

## Laptop-only checks already available

These checks open no microphone and play no audio. Run from the repository
root; replace log/report names with fresh paths rather than overwriting prior
evidence.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.health \
  --config config/runtime.example.json \
  --output results/health-runtime-example-20261015.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --interactive --config config/runtime.example.json \
  --wake-audio wake-word/data/v2/audition/43fd11c385fc_guide_connected.wav \
  --command-audio vcm/data/test_sample.wav --threads 2 \
  --allow-unverified-wake-model \
  --log-file logs/file-based-dashboard-baseline-20261015.jsonl
```

The second command was already run on 2026-10-15: its file fixtures passed
the wake gate and accepted `play_music`. It is file-based evidence only, not a
microphone or Pi result. The health command is an offline artifact/operator
check; it does not approve the wake model.

## Read-only Pi preflight — only after connection approval

After the operator gives the target and approves, use the exact supplied
destination in place of `USER@HOST`. The initial command only reads host,
OS/runtime, storage, audio-device listing, and optional thermal information:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=5 USER@HOST '
  printf "%s\n" "--- board ---"
  tr -d "\000" </proc/device-tree/model 2>/dev/null || true
  printf "%s\n" "--- architecture ---"
  uname -m
  getconf LONG_BIT
  printf "%s\n" "--- OS ---"
  cat /etc/os-release
  printf "%s\n" "--- Python ---"
  python3 --version
  printf "%s\n" "--- memory and storage ---"
  free -h
  df -h
  printf "%s\n" "--- audio inputs (listing only) ---"
  arecord -l
  printf "%s\n" "--- thermal / throttling ---"
  cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || true
  command -v vcgencmd >/dev/null && vcgencmd measure_temp || true
  command -v vcgencmd >/dev/null && vcgencmd get_throttled || true
'
```

Do not run a microphone-format probe that opens a device until it has been
described to the operator and the relevant capture approval is given. A
device listing is not a capture test.

If the target is not `aarch64`, 64-bit, and identified as Raspberry Pi 5,
stop and report the mismatch. Record package/runtime versions on that target;
do not copy the Mac's Python/Torch results as Pi versions.

## Git checkout and offline validation — separate write approval required

The initial bundle and the specifically approved ALSA update have been
transferred and verified. Any additional remote file change requires its own
explicit approval. Those historical transfer records are provenance only; they
are not the procedure for a new run. A new run must not use a tar/ZIP archive
or a copy of the local working tree.

Before Pi setup, the user must push the reviewed local changes and provide:

- A credential-free Git remote URL. Do not place passwords, personal access
  tokens, private keys, or other secrets in the URL, shell history, logs, or
  this document.
- The exact containing branch or tag, written as a ref such as
  `refs/heads/main` or `refs/tags/v5-final`.
- The exact full commit SHA that the Pi must run.

On the local machine, confirm remote reachability and identity before requesting
the Pi write/package approval. The reference must resolve directly to the
requested commit, or, for an annotated tag, its peeled `^{}` value must resolve
to that commit:

```sh
export ME2_GIT_REMOTE='USER_PROVIDED_CREDENTIAL_FREE_REMOTE_URL'
export ME2_GIT_REF='refs/heads/USER_PROVIDED_BRANCH_OR_TAG'
export ME2_GIT_REV='USER_PROVIDED_FULL_PUSHED_COMMIT_SHA'

git ls-remote --exit-code "$ME2_GIT_REMOTE" \
  "$ME2_GIT_REF" "${ME2_GIT_REF}^{}"
git ls-remote "$ME2_GIT_REMOTE" "$ME2_GIT_REF" "${ME2_GIT_REF}^{}" \
  | awk -v expected="$ME2_GIT_REV" '$1 == expected { found = 1 } END { exit !found }'
```

Stop if either command fails, if the remote string contains credentials, or if
the supplied branch/tag and SHA do not agree. Do not change remotes, push,
force-push, or create a pull request as part of this procedure.

Only after that confirmation and a separate explicit approval for Pi writes
and package installation, create a new run directory and check out the exact
commit detached. The run ID is operator chosen and should not identify a
person:

```sh
export ME2_PI_TARGET='USER@HOST'
export ME2_PI_RUN='/home/USER/me2-v5-final-RUN_ID'
export ME2_RUN_ID='RUN_ID'

ssh "$ME2_PI_TARGET" "set -eu
  git clone --no-checkout '$ME2_GIT_REMOTE' '$ME2_PI_RUN'
  git -C '$ME2_PI_RUN' fetch --depth 1 origin '$ME2_GIT_REV'
  git -C '$ME2_PI_RUN' checkout --detach '$ME2_GIT_REV'
  actual=\$(git -C '$ME2_PI_RUN' rev-parse HEAD)
  test \"\$actual\" = '$ME2_GIT_REV'
  test -z \"\$(git -C '$ME2_PI_RUN' branch --show-current)\"
  git -C '$ME2_PI_RUN' ls-files --error-unmatch \
    config/runtime.v5-final-supervised-20261015.json \
    wake-word/models/wake_word_v5_regularized_seed43.me2 \
    vcm/models/vcm_ctc_model.me2
  mkdir -p '$ME2_PI_RUN/results'
  printf 'remote=%s\\nrequested_ref=%s\\nrequested_commit=%s\\nchecked_out_commit=%s\\ndetached_head=true\\n' \
    '$ME2_GIT_REMOTE' '$ME2_GIT_REF' '$ME2_GIT_REV' \"\$actual\" \
    > '$ME2_PI_RUN/results/pi5-source-identity-$ME2_RUN_ID.txt'
"
```

This source-identity record may contain only the credential-free remote,
requested branch/tag, and commit identity. If the selected config, v5/CTC
bundles, or approved replay fixtures are not present and hash-verifiable from
the reviewed revision (or an explicitly approved published artifact source),
stop. Do not fill the gap by copying an untracked local file. Do not transfer
`.venv`, pickles, full datasets, raw recordings, SSH files, or the experiment
archive.

After the detached checkout, verify artifact hashes on the target, confirm the
candidate config hashes and resolved paths before inference, and run
health/artifact smoke and file replay before live capture. Normal inference
startup must not fetch models or datasets.

The target-side offline checks are:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.health \
  --require-pi5 --config config/runtime.example.json \
  --output results/pi5-health-20261015.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --interactive --config config/runtime.example.json \
  --wake-audio fixtures/verified-connected-wake.wav \
  --command-audio fixtures/supported-command.wav --threads 2 \
  --allow-unverified-wake-model \
  --log-file logs/pi5-file-replay-20261015.jsonl
```

Use actual independently reviewed fixtures on the Pi; the laptop example
fixtures are not a substitute. Confirm the app reports the expected resolved
artifact paths/hashes. File replay does not exercise a microphone, dashboard
speaker, continuous wake smoothing, command onset, or endpoint latency.

## Approved Pi venv, deterministic replays, and dashboard tunnel

The following commands are an **expected future procedure**, not a record of
commands run during this finalization. Run them only after the separate Pi
write/package approval, remote/ref/SHA confirmation, detached checkout, and
source-identity record in the preceding section. They do not authorize a
microphone, speaker, Bluetooth, live dashboard listener, or external action.

From the detached Pi checkout, create an isolated venv. Download only binary
packages using the audited CPython-3.14/Linux-ARM64 candidate record, install
them from that local wheel directory, then install the checkout without asking
the package resolver for additional dependencies:

```sh
cd "$ME2_PI_RUN"
mkdir -p wheel-audit logs results
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip download --only-binary=:all: \
  --dest wheel-audit -r config/requirements.pi5-cp314-aarch64.txt
.venv/bin/python -m pip install --no-index --find-links wheel-audit \
  -r config/requirements.pi5-cp314-aarch64.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python -c \
  'import numpy, psutil, scipy, sounddevice, soundfile, torch; print(torch.__version__)'
.venv/bin/python -m pip freeze --all > "results/pi5-pip-freeze-$ME2_RUN_ID.txt"
```

Stop on a missing ARM64 CPython-3.14 wheel, a source-build attempt, failed
import, failed `pip check`, or an incompatible target runtime. The
requirements record is a direct-candidate record, not a complete transitive
lock; retain the wheel filenames, package output, and `pip freeze --all` in
the approved run evidence.

Before a deterministic replay, obtain independently reviewed WAV paths that
are present in the detached checkout or in an explicitly approved published
artifact source. The current repository tracks the frozen V5 config and v5/CTC
bundles, but it does not track an approved replay WAV. Do not substitute a
file from the local development working tree.

```sh
export ME2_WAKE_WAV='APPROVED_CONNECTED_WAKE_WAV'
export ME2_COMMAND_WAV='APPROVED_SUPPORTED_COMMAND_WAV'

test -f "$ME2_WAKE_WAV"
test -f "$ME2_COMMAND_WAV"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.health \
  --require-pi5 --config config/runtime.v5-final-supervised-20261015.json \
  --output "results/pi5-health-$ME2_RUN_ID.json"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --wake-word --config config/runtime.v5-final-supervised-20261015.json \
  --audio "$ME2_WAKE_WAV" --allow-unverified-wake-model \
  --log-file "logs/pi5-v5-wake-replay-$ME2_RUN_ID.jsonl"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --vcm --config config/runtime.v5-final-supervised-20261015.json \
  --audio "$ME2_COMMAND_WAV" \
  --log-file "logs/pi5-vcm-replay-$ME2_RUN_ID.jsonl"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --interactive --config config/runtime.v5-final-supervised-20261015.json \
  --wake-audio "$ME2_WAKE_WAV" --command-audio "$ME2_COMMAND_WAV" \
  --threads 2 --allow-unverified-wake-model \
  --log-file "logs/pi5-integrated-replay-$ME2_RUN_ID.jsonl"
```

The V5 override permits a controlled engineering replay only; it does not
change its `deployment_ready=false` status. Stop and preserve the explicit
rejection/error output when a replay fails. A successful supplied-WAV replay
does not establish microphone, speaker, continuous-stream, or deployment
behavior.

For the visual dashboard check, run the demo server on the Pi only and leave
it bound to Pi loopback. In a separate local terminal, the operator may open
the SSH tunnel after approving the connection:

```sh
# On the Pi, from $ME2_PI_RUN:
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.dashboard \
  --demo --port 8765 --threads 2 \
  --log-file "logs/pi5-dashboard-demo-$ME2_RUN_ID.log"

# On the operator's local machine, in a separate terminal:
ssh -N -L 8765:127.0.0.1:8765 "$ME2_PI_TARGET"
```

Open `http://127.0.0.1:8765` only after the tunnel is established. Demo mode
uses synthetic state, does not load the microphone or recognition models, and
must not be started with `--autostart`. Stop the Pi demo and the tunnel with
`Ctrl-C`. The dashboard's call, message, device, and playback controls remain
local simulations during this review.

## Approved live run and telemetry

The operator must approve each live Pi microphone session. State the device,
phrase, pause/command condition, mock-only outcome, log destination and stop
method first. Wait for approval, then give a visible/audible “GO” cue before
the operator speaks. Never record raw microphone audio unless separately
requested and consented; default logs contain only allowlisted summaries.

Run one session at a time with the frozen config and selected hashes. Use the
CLI interactive path for recognition-only trials; it does not dispatch
dashboard actions. Do not use a dashboard speaker action without separate
speaker approval. Capture baseline and candidate as separate run IDs/logs.

While the already-approved process is running, collect process and safe-log
telemetry in a new output file:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  scripts/pi5_collect_telemetry.py \
  --pid PID --config config/runtime.example.json \
  --log logs/pi5-live-20261015.jsonl \
  --duration-seconds 300 --interval-seconds 1 \
  --output results/pi5-live-telemetry-20261015.json
```

This monitor observes an existing process only: it does not open a microphone,
start inference, play sound, contact weather, or dispatch. It records process
CPU/RSS, board temperature/throttling when available, wake-window end-to-end
processing time, command decision and post-endpoint time, queue delay/depth,
callback overflows/status, clipping counts and artifact/config identity.
False activations/hour deliberately remains unavailable unless a separate
annotated continuous-negative event manifest is supplied. `me2-health` and
telemetry are diagnostic gates, not model-quality approval.

## Acceptance table and stop conditions

Keep separate rows for file inference, laptop microphone and Pi microphone.
For each, retain config/artifact hashes, test IDs, input device/format,
trial counts, negative-audio duration, wake and command outcomes, capture end
reason, valid feature/output frames, terminal/prefix diagnostic counts,
dispatch status, p50/p95 latency, CPU, peak RSS, queue/drop counters and
temperature/throttling where available.

Stop and report a failed or inconclusive gate if any of these occur:

- The current wake artifact cannot pass its readiness guard without a
  supervised override; a successful override test does not change that fact.
- A live command is truncated, exceeds the model's artifact/CTC frame
  contract, has no supported grammar terminal, or is accepted incorrectly.
- Any ordinary/OOS speech or rejected/system-error result dispatches an
  action.
- The stream drops samples, reports callback status/overflow, shows sustained
  queue growth, or takes longer than its configured wake hop/chunk budget.
- The target/runtime/artifact hash differs from the frozen candidate.
- There is not enough annotated negative audio to calculate false
  activations/hour. Do not turn zero observed triggers over a short session
  into a zero-rate claim.

No current candidate passes the overall release gate. Pi readiness requires
independent connected-pronunciation wake evidence, matched wake/OOS tests,
supported long-command trials, explicit rejection tests, sustained clean
streaming, and measured target resource/latency results. Until then the Pi
result is **not run / not deployment-ready**, not “passed.”
