# Deadline plan progress — 2026-10-15

## Decision

The safe dashboard demonstration and offline verification are ready to show.
The live wake-to-command path is **not deployment-ready**. For the final v5
record, v5 is the final supervised runtime/trial candidate, v3 remains the
dashboard file-replay fixture and diagnostic/fault-only fallback, and the
retained CTC bundle is the command-development model. A healthy v5 non-wake
remains non-wake if v3 disagrees. V5 has 2/3 comparable test wake events but
4/48 false accepts, exceeding the 2/48 guardrail; it is not a deployment
promotion.

The current Pi handoff identifies an Ubuntu 26.04.1 ARM64 CPython 3.14.4
target with no ME2 runtime or model checkout in the inspected locations. No Pi
command was run in this finalization. Historical host-file, hardware-only,
ALSA, or earlier trial records below are retained for provenance, but do not
constitute current Pi inference validation. Physical speaker audibility is
unconfirmed, Bluetooth was not tested, and no Pi inference quality or timing
measurement exists for this finalization. The next approved scope is
deterministic file replay and a non-listening dashboard `--demo` review only.
Live microphone, speaker, live dashboard listening, Bluetooth, telemetry,
continuous streaming, and deployment validation are deferred.

The 19 canonical dashboard actions are implemented as local demonstrations.
They do not place calls, send messages, or control physical devices. Weather
is optional and configured-city; automated tests mock the service. Rejected,
unknown, disabled, duplicate, missing/unsupported-slot, and system-error
results fail closed.

The authoritative selection, frozen supervised profile, exact package
candidate audit, cleanup disposition, and approval-separated Pi gates are in
`V5_FINALIZATION_2026-10-15.md`.
The separate supervised laptop plan is in `LAPTOP_OPERATOR_GUIDE_2026-10-16.md`;
it does not change the Pi direct-model/demo-only scope.

## Exact local evidence

- Repository: current ME2_v2 checkout, branch `master`, HEAD
  `20f793d43b664c232f1b25f33a87cf6479c069aa`.
- The checkout had extensive uncommitted and untracked prior work. Nothing was
  staged or committed.
- Runtime CLI file-based profile:
  `config/runtime.example.json` (1,000 ms onset).
- Wake artifact:
  `wake-word/models/wake_word_v5_regularized_seed43.me2`,
  SHA-256 `d7a4898afdc5a5a52fcd305424df4b62582d7bdeea58451d2a90e62646a49222`,
  `deployment_ready=false`.
- Command artifact:
  `vcm/models/vcm_ctc_model.me2`,
  SHA-256 `b5a358e2815ee2471f29956e63936af87c4359bf28b21c21ab420c21e128eeac`,
  legacy grammar, no supported slot mappings, uncalibrated acceptance.
- File-based combined replay used the existing connected wake fixture and
  `vcm/data/test_sample.wav`: wake accepted at 0.9543; command accepted as
  `play_music`; 117 valid CTC output frames; command decision 17.25 ms in the
  latest run. This is one fixture pair and host timing, not live recognition.
- New run log:
  `logs/file-based-dashboard-postchange-confirmed-20261015.jsonl`.
- Offline health report:
  `results/health-runtime-example-20261015.json`. Status is
  `passed_operator_health_not_deployment_approval`; host is macOS `arm64`,
  `measured_pi5=false`. It confirms the same artifact hashes and reports the
  wake readiness and legacy grammar/provenance warnings.
- A read-only SSH preflight was attempted after the operator supplied the
  target and authorized that scope. The first strict-host-key attempt failed
  authentication; the operator then supplied a password and authorized a
  retry. The preflight returned the inventory below.
- The target identified as Raspberry Pi 5 Model B Rev 1.0, Ubuntu 26.04.1 LTS,
  `aarch64`/64-bit, kernel `7.0.0-1020-raspi`, system Python 3.14.4. The
  system interpreter has none of the listed ME2 runtime packages. No checkout
  or `.venv` was found in the two checked conventional home-directory
  locations; this does not rule out a checkout elsewhere.
- CPU: four Cortex-A76 cores, 1.5–2.4 GHz range. Memory: 7.7 GiB total, 7.3
  GiB available, no swap. Root storage: 29 GiB total, 25 GiB available.
  Thermal sysfs reported 41.35°C. `vcgencmd` could not access `/dev/vcio`, so
  throttling status is unavailable.
- After the operator reported the speaker was powered on, a second read-only
  check found kernel ALSA cards for a USB-Audio adapter and two HDMI outputs,
  plus `/dev/snd` device nodes; `speaker-test` is installed. `arecord -l` and
  `aplay -l` initially reported “no soundcards found.” Inspection found that
  the login account lacked `audio` group membership and `/dev/snd` nodes are
  `root:audio` mode `0660`, consistent with the access failure.
- After explicit approval, the one-time speaker command
  `timeout 4s speaker-test -D plughw:0,0 -t sine -f 440 -c 1 -l 1` was
  attempted. ALSA failed before opening playback (`Cannot get card index for
  0`, `Playback open error -2`), so no tone was emitted. No microphone capture
  or audio file was created; terminal output was not saved to a log.
- The operator then explicitly authorized `sudo usermod -aG audio rpiadmin`;
  it completed. A fresh SSH login confirmed supplementary `audio` membership,
  and `arecord -l`/`aplay -l` listed the USB-Audio capture/playback device.
  This was the only system-level change.
- After a separate approval, the same four-second-capped 440-Hz speaker test
  was retried. ALSA opened `plughw:0,0`, configured 48-kHz mono playback, and
  the command exited 0 after streaming the tone. This confirms the software
  playback path; whether it was physically audible at the speaker still needs
  operator confirmation; the operator reported not hearing it. Bluetooth
  playback/pairing was not tested. No audio file was created.
- At the operator's direction, an eight-second USB microphone hardware
  capture then ran at 48-kHz mono S16_LE to `/dev/null`, with a ten-second
  outer timeout. It exited 0 and the live level meter varied. No raw audio was
  retained. This verifies capture at that format, not wake or command
  recognition.
- After explicit approval, a 13-MB bundle of runtime source, configs, and the
  two selected `.me2` artifacts was transferred to a new mode-0700 user-home
  directory and hash-verified before extraction. An isolated Python 3.14.4
  venv was installed there with `torch 2.14.1+cpu` plus NumPy, SciPy,
  SoundFile, psutil, and sounddevice. No system package/service changed. An
  initial default resolver attempt that selected CUDA dependencies was
  canceled before package installation.
- The artifact-only Pi health check passed with synthetic zero audio and two
  threads: wake warm path 5.20/4.05 ms; CTC command-prediction warm path
  295.23/298.29 ms; process RSS about 434 MB; sysfs temperature 44.65°C.
  These are health-smoke values, not live recognition or sustained
  benchmarks. A separate device-enumeration health run failed at PortAudio
  initialization because it could not connect to a PulseAudio server. Direct
  ALSA capture had already succeeded at 48 kHz.
- I implemented an explicit `arecord` backend and Pi config locally; focused
  regression suite: 172 passed, one localhost HTTP test deselected because
  sandbox socket binding is denied. After the operator separately approved
  the transfer, the update archive was SHA-256 verified locally and remotely
  (`2f39869efd78fbefe5f4752760c560d64485a6bda9b69f675bed48f740400b11`) and
  extracted only into the isolated Pi deployment. The five approved files
  were `app.py`, `health.py`, `runtime_config.py`, `streaming.py`, and
  `config/runtime.pi5.natural-pause.json`; no system package or service was
  changed.
- Updated Pi health passed with `audio_backend=arecord`; ALSA listed the
  USB-Audio input at `plughw:0,0`. Runtime config parsing confirmed
  pre-roll 250 ms, onset timeout 3,000 ms, endpoint silence 800 ms, and
  maximum command duration 5,000 ms. Health explicitly reports
  `device_format_check=not_run_no_capture`; no microphone stream or live ME2
  recognition was started as part of this update.
- Updated synthetic-input Pi smoke: wake warm inference 4.99/3.95 ms, CTC
  warm prediction 299.25/297.71 ms, health RSS 432,574,464 bytes, and
  one-time sysfs temperature 44.1°C. This is not an end-to-end, sustained,
  audio-device, or deployment-readiness result. Report:
  `results/pi5-health-alsa-20261015.json`.
- After separate approval, one 30-second live recognition-only Pi session
  opened the USB-Audio `arecord` stream at 16 kHz mono and processed 480,000
  samples. It produced 94 wake windows, zero callback errors and zero queue
  drops. Exactly one window classified as `ateko` (score 0.6708); the wake
  event policy requires two positive hits in three windows, so there was no
  activation. The probability threshold was not the blocker. No command
  capture, CTC decode, or dispatch occurred, so the requested 1.5-second
  pause and `play music` command were not evaluated. Raw audio was not saved;
  exact spoken content cannot be confirmed from this redacted log. Wake
  processing p95 was 6.65 ms and stream-feed p95 8.53 ms, below the 250-ms
  hop during this brief run only. Log:
  `logs/pi5-live-trial-01.jsonl`.
- A separately approved second 30-second run used the same frozen setup. A
  first SSH launch disconnected before the application started; after a
  read-only connectivity check, the actual trial ran normally. It also
  processed 480,000 samples/94 windows with no callback errors or queue
  drops, but had zero `ateko`-class predictions (maximum class score 0.1074),
  zero wake activations, and zero command results. It cannot confirm the
  phrase spoken because raw audio was not retained. Wake p95 was 6.17 ms and
  stream-feed p95 7.97 ms. Log:
  `logs/pi5-live-trial-02.jsonl`.
- After the operator reported missing the GO cue while switching tabs, a
  third approved 30-second run repeated the exact settings. The operator was
  told to remain on this chat and received READY/GO after stream startup. It
  processed 480,000 samples/94 windows with zero errors or drops, but no
  windows classified as `ateko`; the highest class score was 0.3569 and its
  predicted class was not `ateko`. No wake or command event occurred. Exact
  spoken content is unverified because raw audio was not saved. Wake p95 was
  6.73 ms and feed p95 8.19 ms. Log:
  `logs/pi5-live-trial-03.jsonl`.
- A separately approved 8-second 16-kHz mono ALSA meter check wrote to
  `/dev/null` and exited 0. Its meter was mostly 0%, briefly 1% and 4%.
  Because the GO cue arrived after capture had begun, overlap with the
  operator's sentence is unverified; this is inconclusive about speech level
  and not a calibrated level measurement. No raw audio was saved.
- A staged, separately approved 15-second retry established SSH while the
  mic remained idle, then started capture only after the operator replied
  READY. GO followed ALSA's recording-start message. The meter rose from
  approximately 0–1% to 15–17% during the cued interval and later returned
  near zero. ALSA exited successfully with output directed to `/dev/null`;
  no inference or file recording occurred. This confirms noticeable input
  signal during the cued interval, not the words spoken or recognition.
- The initial SSH inventory command returned status 255 after the `vcgencmd`
  errors; subsequent audio inventory and fresh-login verification completed
  with status 0. The isolated user-home deployment and venv are the only
  remote file/package changes; no system packages or services changed and no
  raw audio was retained. The first playback attempt failed before emitting
  audio; the approved post-fix retry streamed the tone. No weather service was contacted. Findings are saved in
  [`results/pi5_readonly_preflight_2026-10-15.json`](../results/pi5_readonly_preflight_2026-10-15.json).

## Implementation and verification

- The safe structured `command_result` now logs the accepted canonical intent
  alongside decoder, model, grammar, artifact and timing information; the
  existing redaction allowlist still excludes transcript and slot text.
- Continuous wake-window timing now includes frontend preparation, model
  inference, and wake smoothing. The dashboard's existing `inference_ms`
  display retains its narrower model-inference meaning.
- The Pi telemetry sidecar observes an already-running process only. It
  summarizes CPU, peak RSS, thermal/throttling signals, wake-window and
  command timing, queue delay/depth, callback drops/status, and clipping.
  Absolute local paths and transcripts are omitted. It deliberately does not
  calculate false activations/hour without an annotated continuous-negative
  trial set.
- All 19 accepted intents traverse the structured pipeline event, dispatcher,
  and visible-state assertions. Added checks cover slot misuse, system-error
  non-dispatch, action failure without visible state mutation, safe telemetry
  summaries, and full wake-window timing.
- Full automated regression command:

  ```sh
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/pytest -q \
    -k 'not http_assets_and_control_require_local_host_and_token'
  ```

  Result: **168 passed, 1 deselected**. The deselected localhost HTTP test
  could not bind its loopback socket in this sandbox (`PermissionError`); it
  was not counted as passed.
- Focused dashboard/capture/decoder/weather/telemetry run: **80 passed, 1
  deselected** before adding one additional wake-timing test. That added test
  and the telemetry/dashboard focused subset subsequently passed (**24
  passed**).
- Ruff and `git diff --check`: passed.

## Capture and model decision

The known baseline is a one-second command-onset deadline. A separate
three-second onset-only replay candidate exists and supports a 1–2 second
pause in deterministic replay, but it has not passed a new approved live
laptop trial. Keep `config/runtime.onset-3s.json` a candidate. The
`config/runtime.pi5.natural-pause.json` profile also changes endpoint silence,
so it is not the same controlled onset-only comparison.

The previous longer-capture replay recorded an OOS false accept, and prior
live trials include a wake trial with no positive windows. The candidate
settings therefore are not promoted and no acceptance threshold was relaxed.
The CTC artifact has a five-second input contract and 126 maximum output
frames; the audit identifies a longest legacy phrase requiring 128 frames.
Longer capture alone cannot make that path feasible. The grammar's 577
terminals do not supply the unsupported slot semantics requested by some
dashboard actions. These are artifact/grammar boundaries, not proof that
onset timing caused every `no terminal candidate`.

No retraining was warranted in this pass: the tested long-phrase evidence
does not establish a training-data/model failure after grammar and frame
contract inspection, and changing acceptance would compromise the known
rejection guard. Preserve the model and collect exact supported-phrase live
trials after capture is approved.

## Remaining gates

1. The isolated ME2 runtime, artifacts, and approved `arecord` backend/config
   are installed; file-only health and ALSA device listing pass. The
   sounddevice/PortAudio input probe still fails because no PulseAudio server
   is available in the SSH session, so the Pi config selects direct ALSA.
2. Obtain separate approval before each additional live microphone/speaker
   trial. Three earlier supervised recognition trials emitted no wake event;
   command capture was not exercised. Two later wake-only captures are
   excluded from recognition assessment: in trial 04 the operator missed GO;
   in retry 05 the capture ended before the assistant delivered GO. Both only
   confirm technical capture, not wake recall.
   Prior callback/drop results apply only to the runs that used the streaming
   app; the latest buffered wake-only check did not measure live queue health.
   The tone reached ALSA, but the operator did not hear it and Bluetooth
   remains untested.
3. Before broader Pi evaluation, freeze one laptop-passing config and record its hash;
   use the identical config, model hashes, decoder, and rejection settings on
   Pi. Current wake bundles still fail the readiness gate.
4. Validate valid wake + natural pause + supported long command, immediate
   handoff, ordinary/OOS speech, invalid wake, silence and repeated wake on
   the target with a mock-only dispatcher. Record negative-audio duration and
   event matching before claiming false activations/hour.
5. Report ARM64 OS/Python/runtime, wake-hop and chunk p50/p95, command
   post-endpoint p50/p95, sustained CPU, peak RSS, queue/drop status,
   temperature/throttling, wake recall, command errors and rejection results.

The SSH-safe staged runbook is
[`PI5_SSH_SAFE_VALIDATION_2026-10-15.md`](PI5_SSH_SAFE_VALIDATION_2026-10-15.md).
