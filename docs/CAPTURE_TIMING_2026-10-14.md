# Command capture investigation — 2026-10-14

Status: **implementation candidate; not promoted and not committed**. Offline
timer fixes are reproduced. Approved live laptop runs through T13 are logged
across the Haylou earbuds and wired external microphone. The 100-row capture
replay exposes a rejection regression, and wired live trials have not yet
produced a stable wake event, so neither this profile nor the selected model
is declared safe for deployment. No new training, threshold tuning, grammar
changes, or artifact replacement occurred.

## Baseline and provenance

Repository: `ME2_v2`, branch `master`, baseline
`e3aa9f8bedf25aec64a7eeb033bd506ae133a46e`, origin
`alaine-richelle/ME2_v2`. Tracked and staged files were initially clean.
Existing untracked datasets, models, navigation skills, build output, reports
and logs are preserved. The original runtime was exported with `git archive`
into a separate temporary source tree for before/after execution.

Read the hardening, final-selection and runtime-validation reports:
`HARDENING_2026-10-13.md`, `IMPROVEMENT_CYCLE_2026-10-06.md`,
`RUNTIME_VALIDATION.md`. Previous official-test results have already influenced
development; they are not fresh final evidence. This investigation uses only
the saved official-train validation selection and deterministic fixtures.

Settings resolve defaults < JSON < explicit CLI flags. Relative JSON paths
resolve against the JSON's directory (`runtime_config.py::load_runtime_defaults`),
not the current directory. Both the original and candidate profiles resolve:

| Artifact | SHA-256 |
|---|---|
| `wake-word/models/wake_word_v5_regularized_seed43.me2` | `d7a4898afdc5a5a52fcd305424df4b62582d7bdeea58451d2a90e62646a49222` |
| `vcm/models/vcm_ctc_model.me2` | `b5a358e2815ee2471f29956e63936af87c4359bf28b21c21ab420c21e128eeac` |
| Retained `vcm/models/vcm_baseline_model.me2` | `85ebe9751f6ad7107936615ac9da975b16a16034f77900cefb0d1ac8d43069f9` |

Host: macOS ARM64, project `.venv/bin/python` 3.14.3, Torch 2.14.1,
two Torch threads. Exact environment and artifact details are in the local
machine-readable evidence. The October 13 interactive tests used Haylou GT7
Neo, Bluetooth mono 16 kHz, then device index 2. On October 14 native metadata
inspection lists only the built-in microphone and speakers: the earbuds are
not connected. Do not reuse numeric device 2 without checking it.

The original live command was:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.app \
  --interactive --config config/runtime.example.json --input-device 2 \
  --threads 2 --allow-unverified-wake-model --log-file /dev/null
```

Those historical live outputs are chat evidence only. `/dev/null` left no
recoverable waveform or persisted diagnostic log. We cannot infer actual word
boundaries, user timing, or the cause of the additional wake from those outputs.

## Confirmed mechanisms and unresolved hypotheses

Baseline line numbers below refer to commit `e3aa9f8`, not the edited tree.

1. **Onset budget is shortened by pre-roll.** `streaming.py:142–167` initializes
   `_command_samples` with 250 ms pre-roll and checks it against the 1,000 ms
   onset timeout. At 100 ms chunks, a no-speech capture rejects 800 ms after
   wake. An actual 1.5-second pause cannot work with that configuration.
2. **Three clocks share one counter.** `streaming.py:155–175` charges pre-roll,
   onset waiting, speech, internal pauses and endpoint silence against the
   same five-second maximum. The maximum test precedes the endpoint test.
   A controlled 4.3-second speech signal is rejected at 4.8 seconds after wake
   while confirming its endpoint. This is safe discard, not a decoder failure.
3. **Minimum speech is not speech duration.** The same lines compare total
   buffered samples with the 250 ms minimum. A 100 ms noise burst plus silence
   reaches the decoder. Correcting this exposes a trade-off for quiet commands.
4. **Energy onset can lose quiet speech.** The 0.008 RMS gate is not a verified
   word boundary. A rolling 250 ms pre-roll can discard low-level speech before
   onset is recognized. Validation row 10015 has speech-like signal below the
   threshold for approximately 1.4 seconds. With the longer wait it reaches
   the selected uncalibrated model and is incorrectly accepted as `play_music`.
   This is an unresolved capture/rejection risk, now explicitly instrumented.
5. **A generic no-terminal message hides distinct failures.** Baseline
   `ctc.py:594–607` returns `no_terminal_candidate` without frame or surviving
   prefix counts. New diagnostic categories describe observed conditions,
   without asserting that pruning or unsupported wording caused every failure.

Findings that do **not** support the suspected historical frame-crop bug:

- Selected architecture is `CharacterAudioCNN`, 3,652,572 parameters, 5 s at
  16 kHz; 40 log-power Mel bands, FFT 400, hop 160. No fixed 128-feature-frame
  slice exists in the inspected runtime.
- Maximum frontend length is 501; convolution geometry yields
  `ceil(ceil(501/2)/2) = 126` output frames. Packed recurrent lengths and
  decoder input lengths exclude padded outputs. BiGRU processes the completed
  utterance; it is not causal streaming ASR.
- `audio.py::prepare_audio_sequence` rejects overlength input rather than
  silently cropping; its duration exception is now categorized.
- CTC decoder consumes every valid frame, retains repeated-symbol self-loops,
  requires a blank to append a repeated character, and merges with log-sum-exp.
  Brute-force probability and prefix-terminal tests still pass. No recurrence,
  beam width, or acceptance threshold changed.

Remaining model/grammar boundaries:

- Alphabet is `abcdefghijklmnopqrstuvwxyz `; blank index 27. Grammar has 578
  legacy training-transcript terminals, no verified slot mappings, and lacks
  full training-dataset provenance. Runtime normalizer preserves digits, but
  the legacy alphabet and grammar cannot provide arbitrary numeric values.
- One 123-character grammar terminal requires 128 CTC steps (including repeats),
  beyond this artifact's 126-step maximum. Extending only capture cannot make
  that phrase feasible. The offline report identifies it explicitly.
- The selected acceptance policy is `legacy-uncalibrated`: effectively no
  calibrated acoustic/margin threshold. Beam posterior is not calibrated
  correctness. Improving capture can expose pre-existing false accepts.
- Several phrases previously suggested in chat are absent from this artifact,
  including “set a timer”, “set an alarm”, “increase brightness”, “change the
  color”, and “create a reminder”. Use exact verified terminals for testing;
  do not interpret those failures as duration failures.
- No new evidence establishes wireless packet loss or clipping as a root cause.
  Future logs include RMS/peak/clipping counts, queue delay, callback status,
  ADC timestamps when available, and recovery events. Device transport loss
  invisible to PortAudio cannot be measured by those counters.

## Implemented candidate

`streaming.py` now maintains independent onset, speech-activity, speech-span
and trailing-silence counters. Waiting retains bounded rolling pre-roll.
Maximum speech span includes internal pauses; endpoint confirmation gets its
own allowance. Complete captured samples, including quiet endings, reach
artifact preprocessing. Audio that exceeds the artifact contract is discarded
explicitly, with no cropped decode.

| Setting | Original example | Separate candidate |
|---|---:|---:|
| Onset timeout | 1,000 ms (effectively 800 ms previously) | 3,000 ms from wake |
| Pre-roll | 250 ms | 250 ms |
| Minimum energy-positive speech | 250 ms, previously measured incorrectly | 250 ms |
| Endpoint silence | 500 ms | 800 ms |
| Maximum command speech span | 5,000 ms, previously shared with waiting | 5,000 ms |
| Cooldown | 1,000 ms | 1,000 ms |
| Chunk / requested wake hop | 100 / 250 ms | 100 / 250 ms |
| RMS threshold / queue capacity | 0.008 / 32 chunks | unchanged |
| Decoder beam / limits / policy | 16 / existing budgets / legacy policy | unchanged |

Profiles: `config/runtime.natural-pause.json` and
`config/runtime.pi5.natural-pause.json`. The original example JSON remains
unchanged. Missing `maximum_command_ms` inherits the artifact duration.
An explicit maximum above that duration fails before microphone startup.
Older configuration files continue to load, but corrected timer/minimum-speech
semantics are deliberately different.

Three seconds covers the requested 1–2-second pause with a finite margin.
Eight hundred milliseconds preserves the full controlled utterance containing
a 600 ms internal pause. These are candidate settings from explicit timing
requirements and replay tests, not measured live-optimal values. An extra
300 ms endpoint allowance adds approximately 300 ms to decisions after speech
ends. Actual chunk scheduling/driver latency must be measured.

Capture storage remains bounded: onset waiting stores only pre-roll; active
capture retains at most maximum span + pre-roll + endpoint + one chunk.
At this profile that is at most about 394 KB of float32 audio, plus the
205 KB queue, wake buffer, Python objects and model runtime. These calculated
payload bounds are not process RSS or Pi measurements.

`CommandResult.diagnostics` and rotating allowlisted JSON logs retain sample/
feature/output lengths, deadline/speech evidence, decoder prefix/terminal/
pruning counts, capture end reason and actionable reason codes. They do not
retain waveforms or arbitrary transcript/contact fields. Pre-onset dropped
samples/energy are reported and `onset_boundary_verified=false` is explicit.
Unknown phrase support remains unknown without a reference; the scripted
reference checker does not alter grammar or acceptance.

`--session-seconds` adds cooperative bounded live sessions. The separate trial
runner also supervises the audio process, terminates stalled children, uses
exclusive new log paths and requires `--microphone-approved`. A watchdog
timeout is a system failure, not a successful trial.

## Newly reproduced results

Local evidence directory: `results/capture_timing_2026-10-14/`.
No audio is written there.

| Offline scenario | Original | Candidate |
|---|---|---|
| Immediate 600 ms command signal | decoded | decoded |
| 1.5 s / 2 s pause then short signal | onset timeout | decoded |
| 4.3 s speech signal | duration rejection | complete speech decoded |
| 2 s pause then 4.6 s speech | onset timeout | complete speech decoded |
| 600 ms internal pause | prematurely decodes first 600 ms speech | retains both 600 ms speech portions |
| 100 ms noise burst | reaches decoder | `command_too_short` |
| 5.1 s speech | duration rejection | duration rejection |
| No command | onset timeout | onset timeout |
| Existing prepared file fixture, immediate | `play_music` | `play_music` |
| Same fixture after 1.5 s pause | onset timeout | `play_music` |

Synthetic signal rows use mock wake/command predictors to verify mechanics,
not recognition. File-fixture rows use a mock wake edge and the real selected
command model. Removing existing fixture padding uses the shared frontend;
these rows are not natural human continuous recordings.

The same 100 official-train validation rows (five per label, 23 source
identities, pinned HF revision `bfca0bbebfe17e00d6a685d1da573615d29ecdce`)
were used for model-only and full capture comparisons. No test/holdout data,
normalization fitting, model training or threshold selection occurred.
The legacy model's training ancestry is incomplete, so independence from
its original training cannot be established.

**Model-only:** all 100 accepted flags, candidate texts, intents, rejection
reasons and scores match exactly. Accuracy .78; macro F1 .802782 over 20
labels; OOS FA 1/5; in-scope rejects 14/95; wrong accepted 7/95.
Laptop p95 12.47 → 13.56 ms in one measured pass (about +8.7%).
These timings exclude capture and endpointing.

**Full capture replay with a mocked wake, raw validation audio:**

| Pause | Version | Intent accuracy | Macro F1 (20 labels) | OOS FA / 5 | In-scope rejects / 95 | Wrong accepted / 95 |
|---|---|---:|---:|---:|---:|---:|
| 0 | original | .75 | .78035 | 0 | 20 | 5 |
| 0 | candidate | .75 | .78245 | 1 | 19 | 5 |
| 1.5 s | original | .05 | .00476 | 0 | 95 | 0 |
| 1.5 s | candidate | .75 | .78245 | 1 | 19 | 5 |

These are paired development diagnostics, not deployment prevalence estimates
or live wake recall. Rejections/no-response remain in denominators. The
unanswered original delayed trials cannot be counted as superior recognition
because they never reach decoding.

Regression analysis: row 10015 is annotated OOS ordinary speech and becomes a
wrong `play_music` acceptance. Its initial audio remains below the fixed onset
energy threshold for about 1.4 s. Row 16 is labeled play_music (automatic
transcript “Play.”) and has only two 100 ms energy-positive chunks; the corrected
250 ms minimum rejects it. No listening review was claimed, labels were not
changed, and no row-specific exceptions were added. Both regressions block
promotion. Timing repair alone does not solve onset boundary uncertainty or
OOS calibration. Lowering thresholds to conceal these failures is not done.

## Approved live microphone trials

All live runs used a dry-run dispatcher and saved metrics-only JSONL logs; no
raw microphone audio was retained. Historical Haylou trials T01–T04 are
summarized in the trial evidence above, with the limitations noted there.
T05–T13 used the wired external microphone, so results are not directly
comparable across devices.

| Trial | Condition | Outcome |
|---|---|---|
| T05 | Wired mic, wake-only | No wake event; no command capture |
| T06 | Wired mic, slower wake-only | One isolated wake-class window; no smoothed event |
| T07 | Wired mic, two attempted connected wake utterances about 0.5 s apart, then silence; 19.9 s | 60 windows; 5 isolated `ateko` class windows (maximum score 0.8791), zero emitted wake events; command stage not entered |
| T08 | Wired mic, one attempted connected wake, ~1.5 s pause, then attempted `play music`; 20.0 s | 60 windows; 3 isolated `ateko` class windows (maximum score 0.8796), zero emitted wake events; command stage not entered |
| T09 | Wired mic, wake-only, then silence; 20.0 s | 61 windows; 1 isolated `ateko` class window (score 0.8004), zero emitted wake events; command stage not entered |
| T10 | Wired mic ~10 cm away, wake-only, then silence; 20.0 s | 61 windows; 1 isolated `ateko` class window (score 0.8881), zero emitted wake events; command stage not entered |
| T11 | Wired mic, input profile checked, wake-only, then silence; 19.9 s | 60 windows; 1 isolated `ateko` class window (score 0.9046), zero emitted wake events; command stage not entered |
| T12 | Wired mic, 8-second temporary waveform diagnostic | 25 windows; no `ateko` classifications or wake events; command stage not entered |
| T13 | Wired mic, stream opened before explicit GO cue, one attempted connected wake then silence; 8.0 s | 25 windows; zero `ateko` classifications or wake events; strongest wake probability 0.1730, still labeled `non_wake` |

T07 had zero clipped samples and PortAudio reported 0.61675 s input-stream
latency. The five positive-class windows were separated over the trial rather
than satisfying the configured 2-of-3-window smoother. Thus there was no
handoff, onset timer, or command-decoder result to assess, and no interaction
was dispatched. Without retained audio, the log cannot verify whether each
spoken attempt was acoustically complete or identify which attempt caused
each positive window. Do not infer pronunciation correctness from class scores.

T07 log: `logs/laptop-long-command-20261014-T07-double-wake-230621.jsonl`.
T08 log: `logs/laptop-long-command-20261014-T08-wired-pause-play-music.jsonl`.
T09 log: `logs/laptop-long-command-20261014-T09-wired-wake-isolation.jsonl`.
T10 log: `logs/laptop-long-command-20261014-T10-wired-close-level-wake-only.jsonl`.
T11 log: `logs/laptop-long-command-20261014-T11-wired-input-level-verified-wake.jsonl`.
T12 log: `logs/laptop-long-command-20261014-T12-wake-waveform-diagnostic.jsonl`.
T13 log: `logs/laptop-long-command-20261014-T13-wired-ready-cued-wake-diagnostic.jsonl`.
T08's final state was STOPPED after the supervised 20-second session, with no
utterance ID, no onset wait, and no decoder result. The wired-mic wake event
remains the blocking step before meaningful pause/long-command laptop trials.
Threshold and smoothing settings remain unchanged because the available
continuous validation data are insufficient to recalibrate them safely.

T09 likewise ended in STOPPED with no utterance ID or onset wait. It had zero
clipped samples; the largest 100 ms chunk had RMS 0.0271 and peak amplitude
0.1261, with only three chunks above RMS 0.02. The log therefore confirms one
positive-class model window but cannot establish the spoken content or exact
mic-to-mouth distance. Low input level or pronunciation/window alignment are
possible contributors, not confirmed causes.

T10 also ended in STOPPED with no utterance ID or onset wait. It had zero
clipped samples; the largest 100 ms chunk had RMS 0.0218 and peak amplitude
0.0628, with two chunks above RMS 0.02. The single positive-class model window
had score 0.8881. Moving closer did not produce the two positive windows
needed for an event in this run. Aggregate levels cannot distinguish
microphone gain, utterance timing, pronunciation, or model/window mismatch.

For T11, the user confirmed the macOS input profile before the trial. Peak
100 ms chunk RMS rose to 0.1216 (peak amplitude 0.3649; seven chunks above
RMS 0.02), with zero clipping, yet only one of 60 windows classified as wake.
This makes low input level alone a less sufficient explanation, but without
retained audio the exact spoken content and alignment remain unverified. No
wake event, onset wait, or command decode occurred.

T12 captured 8 seconds from the wired external microphone to a temporary WAV
under `/private/tmp`, ran offline wake scoring, and removed the WAV in a
`finally` cleanup; a follow-up filesystem check found no remaining
`me2-t12-wake-*` WAV. It produced 25 windows, all predicted `non_wake`; the
largest wake probability was 0.00952 (window starting at 750 ms), with zero
wake hits and zero emitted events. Whole-capture RMS was 0.00468, peak
amplitude 0.03267, and clipping count zero. The structured log records this as
`whole_capture_rms` and includes the selected wake artifact SHA-256. The WAV
was not auditioned, so the actual phrase and delivery cannot be verified from
this diagnostic.

Correction to the earlier runtime-window estimate: the selected wake artifact
sets a **2.0-second** input duration. `app.py` derives the streaming window
from that artifact duration and the candidate runtime hop is 250 ms. The
1.5-second value is only the `StreamingConfig` fallback, not this loaded
artifact's actual window.

For T13 the input stream was confirmed active before the assistant issued GO,
and the user was explicitly told to speak only after that cue. The temporary
WAV was scored offline and removed; no matching file remained in `/private/tmp`.
All 25 two-second windows predicted `non_wake`; the strongest wake probability
was 0.1730 in the window starting at 1,000 ms. Maximum 100 ms RMS was 0.0259,
whole-capture RMS 0.00736, peak amplitude 0.0916, and clipping count zero.
Thus the 2-of-3 smoother was not the only barrier in this run: the classifier
did not label any window as `ateko`. Since the waveform was removed without
audition, the log still cannot prove what exact words were captured.

## Verification and change inventory

Final regression run on October 14: **127 tests passed in 9.01 seconds**.
This includes the existing brute-force CTC probability tests, streaming
recovery/overflow tests, and new capture-clock, endpoint, duration, diagnostic,
configuration, privacy and supervised-trial tests. Device behavior is mocked
in automated tests; this result does not constitute microphone validation.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts/capture_timing_validation.py scripts/laptop_capture_trial.py
git diff --check
```

The complete suite required permission for its local-only HTTP server test.
No microphone or speaker was opened. Ruff and whitespace checks passed.
All three retained artifact checksums above were rechecked and unchanged.
No files are staged, and no commit has been created.

| Files | Candidate change |
|---|---|
| `src/me2/streaming.py` | Independent capture clocks, speech minimum, bounded context, sample-loss and capture diagnostics |
| `src/me2/audio.py`, `src/me2/bundle.py` | Typed duration/input failures and actual usable/feature/output lengths |
| `src/me2/vcm/ctc.py` | Decoder diagnostics and reference-phrase support inspection; unchanged decisions |
| `src/me2/app.py`, `src/me2/runtime_config.py` | Validated duration control, timed sessions, actionable result messages |
| `src/me2/logging_utils.py` | Allowlisted diagnostic fields and implementation fingerprint |
| `config/runtime.natural-pause.json`, `config/runtime.pi5.natural-pause.json` | Separate provisional profiles; original example preserved |
| `scripts/capture_timing_validation.py`, `scripts/laptop_capture_trial.py` | Paired offline replay and individually approved, supervised live trials |
| `tests/test_capture_timing.py`, `tests/test_streaming.py` | Focused new tests and corrected old clock expectations |
| This report and `docs/RUNTIME_VALIDATION.md` | Evidence, approval workflow, limitations and target runbook |

Local generated evidence under `results/capture_timing_2026-10-14/` remains
untracked and separate from models. Existing untracked recordings, artifacts,
logs, reports and navigation resources were not staged or removed.

## Reproduction and approval-gated live trials

Create a baseline source snapshot in a new temporary directory:

```sh
baseline_dir=$(mktemp -d /tmp/me2-capture-baseline.XXXXXX)
git archive e3aa9f8bedf25aec64a7eeb033bd506ae133a46e src config \
  | tar -x -C "$baseline_dir"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/capture_timing_validation.py \
  --baseline-root "$baseline_dir" --output results/<new-run>/offline.json
```

To also replay the persisted development subset, pass `--cache-revision`
with the pinned official-train Arrow cache, `--selection` with the recorded
`model_before.json`, and `--partitions results/vcm_ctc_partitions.json`.
The runner checks dataset revision, saved assignment hash and subset membership.
Its two children run the baseline and candidate source separately.

Run a single live trial **only after explicit microphone/log approval**:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/laptop_capture_trial.py \
  --config config/runtime.natural-pause.json \
  --device "Haylou GT7 Neo" --trial-id T03-after --seconds 20 \
  --expected-text "play music" \
  --log logs/laptop-long-command-<timestamp>-T03-after.jsonl \
  --allow-unverified-wake-model --microphone-approved
```

For the corresponding before trial, supply `--source-root "$baseline_dir"`,
`--config config/runtime.example.json`, and a fresh `T03-before` ID/log.
Both use the unchanged real artifact paths resolved against the original
configuration files. The runner prints READY after model/device validation.
Each trial lasts 20 s, supports Ctrl+C, and has a parent watchdog. Keep raw
audio retention disabled. The script prints accepted intent to the terminal;
the persistent trial log contains counts/codes, not private transcript text.

| Trial | Scripted condition | Before live | After live |
|---|---|---|---|
| T01 | Quiet, then incomplete “a teh” | 0 wakes, 0 command decisions; audio-free log cannot verify utterance timing | pending |
| T02 | Connected wake + “play music” | Baseline: 0 wakes. Candidate repeat: 1 wake and 1 accepted terminal; intent not persisted | Pending a baseline pause trial and a candidate trial with intent logging |
| T03 | Connected wake, 1–2 s pause, “play music” | 0 wakes; pause/command stage not reached | 0 wakes; pause/command stage not reached |
| T04 | Connected wake + “play music” on wired mic | not run | 0 wakes; six `ateko` windows were too far apart for smoothing; command stage not reached |
| T04 | Wake, pause, “can you read me my next reminders” | pending | pending |
| T05 | Wake immediately followed by same longer phrase | pending | pending |
| T05-wake-only | One connected wake on wired mic, then silence | not run | 0 wake events; no `ateko` class window; command stage not reached |
| T06-wake-only | One slower, connected wake on wired mic, then silence | not run | 0 wake events; one `ateko` window; command stage not reached |
| T06 | Wake then ordinary unrelated speech | pending | pending |
| T07 | Wake then unsupported “translate this sentence into french” | pending | pending |
| T08 | Repeated wake; check duplicate events | pending | pending |
| T09 | Wake + brief “stop” / softer supported speech | pending | pending |

The longer T04/T05 phrase is an exact grammar terminal, intent
`list_reminders`, requiring 33 CTC frames. Its presence is not evidence of
acoustic recognition quality. No speaker playback is needed for these trials.
Each trial needs separate confirmation, timing instructions, result review
and before/after event/capture/decoder/latency comparison.

## Pi runbook and remaining gates

The Pi profile has identical capture/decoder settings; it is a pending test
configuration, not an approved deployment. Keep the float artifact and baseline.
Use a target-compatible, pinned ARM64 environment and an offline wheelhouse.
Do not substitute laptop Python package versions for a measured Pi lock.

1. Check board, architecture, artifact hashes and operators with
   `python -m me2.health --require-pi5 --config config/runtime.pi5.natural-pause.json`.
2. Select the actual Pi microphone by name and validate 16 kHz mono input.
3. Run the same separately approved trial matrix with new logs and raw audio
   disabled. Use the supervised script, replacing the Bluetooth device name.
4. Run `python -m me2.benchmark <fixture.wav> --model <artifact.me2>
   --threads 2 --iterations 100 --warmup 10 --output <new-report.json>`.
5. Measure idle/active CPU, peak process RSS, frontend/model/decoder latency,
   sample-end-to-decision p50/p95 (including 800 ms endpoint), callback losses,
   retry counts, sustained temperature and throttling over at least 30 minutes.

Engineering gates: no silent truncation, duplicate dispatch, unbounded capture,
or recovery error converted to success; valid pause/long scenarios must improve
without newly observed wrong-command/OOS accepts at comparable coverage;
unaffected short cases must remain accepted. Predeclare an operating FA/hour
target and collect enough independent continuous negative hours for an upper
confidence bound. The short prior laptop negative trial cannot satisfy it.
As provisional processing budgets, require wake-path p95 below the actual hop
and command compute p95 below 500 ms on the Pi; report endpoint latency
separately. These budgets are requirements, not measured Pi results.

No Pi is available. Two approved baseline laptop trials are recorded below;
candidate live trials remain pending. No final configuration/model is selected;
no commit is made before
relevant live tests and the rejection guardrail pass. Retraining is deferred:
the demonstrated timer errors do not require it. Better onset/endpoint evidence,
independent real wake recordings and rejection calibration are required first.
Commands beyond the artifact's complete five-second input contract require a
separate duration/model/grammar study and versioned artifact.

## Approved live laptop trials

Eight 20-second trials were run through the supervised runner: three baseline
trials and five candidate trials. The first four used the Haylou GT7 Neo; later
candidate trials used the wired “External Microphone, Core Audio” input. The
runner confirmed mono 16 kHz input, used a dry-run dispatcher, and stopped
cleanly after 19.9–20.0 seconds. No trial retained audio.

| Trial | Runtime | Intended condition | Observed result | Limitation |
|---|---|---|---|---|
| T01 | Baseline | Quiet, then two incomplete “a teh” attempts | 0 wake events, 0 command decisions, 0 dispatches; no clipped samples | Level bursts occurred at several times; without audio the exact words and timing cannot be verified |
| T02-before | Baseline | Connected “a-teh-ko” immediately followed by supported “play music” | 0 wake events, 0 command decisions, 0 dispatches | 35/199 chunks contained clipping (398 clipped samples total); peak chunk RMS .5023; level-only data cannot isolate the target phrase or verify timing |
| T02-after | Natural-pause candidate | Same immediate wake + command after the input level was reduced | 1 wake, 1 accepted terminal, 0 rejected; no clipped samples | The older trial logger did not persist the accepted intent. Do not count this as verified `play_music` accuracy |
| T03-before | Baseline | Connected wake, 1.5-second pause, then supported “play music” | 0 wake events, 0 command decisions, 0 dispatches | 4/200 chunks contained 18 clipped samples; peak chunk RMS .3823; no wake-window scores in baseline log |
| T03-after | Natural-pause candidate, wired external mic | Same 1.5-second pause scenario | 0 wake events, 0 command decisions, 0 dispatches | 61 windows; two `ateko` classifications 6.3 s apart; two chunks contained 127 out-of-range samples; reported stream latency .617 s |
| T04-wired-immediate | Natural-pause candidate, wired external mic | Connected wake immediately followed by “play music” | 0 wake events, 0 command decisions, 0 dispatches; no clipping | Six `ateko` windows scored .584–.971, separated by 3.3–3.6 s; stream latency .617 s |
| T05-wake-only | Natural-pause candidate, wired external mic | One wake attempt, then silence | 0 wake events, 0 command decisions; no clipped samples | 61 windows, no `ateko` classification, maximum wake probability .135; stream latency .617 s |
| T06-wake-only | Natural-pause candidate, wired external mic | One slower connected wake, then silence | 0 wake events, 0 command decisions; no clipped samples | 60 windows; exactly one `ateko` classification (.954); stream latency .617 s |

The T02 candidate log records 45 wake windows, two classified as `ateko`, one
smoothed wake event, and a maximum wake probability of .7226. It measured
500 ms from wake handoff to energy onset, 700 ms of energy-positive speech,
800 ms of endpoint silence, and 43 valid CTC output frames. Its decoder retained
one terminal and accepted it, with no truncation. “Play music” is confirmed as
a legal phrase for this grammar, but this run's accepted intent was not written
to the persistent log and its terminal cannot be identified now. The log does
not establish that the connected wake and command were spoken exactly as
scripted.

The baseline T02 did not enter command decoding, and its input clipped. The
candidate T02 did decode, with no clipping, but its accepted intent is
unavailable. These runs are useful capture evidence, not a matched accuracy
comparison and not proof of a safe command decision. The supervised trial
logger now persists only the accepted canonical intent (never free-form
decoded text or slot values) for future reviewed trials.

The later baseline T03 run likewise emitted no wake event. It therefore cannot
confirm the suspected effective onset deadline: the command-onset clock never
started. Its 20-second log contains several high-level input bursts and 18
clipped samples across four chunks, but the baseline source does not log
wake-window scores, and no waveform was retained for pronunciation or timing
review. The candidate T02 run shows the selected mic and wake artifact can
produce an event in at least one run; more controlled trials remain necessary.

Candidate T03 also never reached command capture. Its two windows labeled
`ateko` scored .6629 and .7479 at 6.3 seconds apart. Both exceed this bundle's
stored .0000774 wake threshold, but the smoothing policy requires two hits
within three adjacent wake windows, so the separated hits did not emit an
event. In candidate T02 on the Haylou, the two `ateko` hits that emitted a wake
event were about .30 seconds apart. This identifies why the T03 trace did not
handoff under the current smoother, but the retained level data cannot establish
which spoken sounds produced either classification. No threshold or smoothing
setting was changed based on these few trials.

The wired input reports .617 seconds of stream latency and peaks near 3.0
before shared preprocessing clips samples to the model range. Only 127 samples
across two chunks were out of range. This is a device-path confound and should
be measured with the microphone gain and placement reviewed; it does not by
itself explain the absent wake event.

Candidate T04-wired-immediate used the same wired input with no clipped samples and a peak
chunk RMS of .0704. Across 61 wake windows, six were classified as `ateko`,
with probabilities .584–.971. They were separated by roughly 3.3–3.6 seconds,
and no wake event was emitted. The current smoother needs two positive windows
among three adjacent windows. Thus smoothing explains why these isolated
classifications did not hand off. Because the microphone audio was not stored,
the trace cannot show whether they corresponded to repeated wake attempts,
other speech, or background sound. No acceptance threshold or smoothing
setting was relaxed.

Candidate T05-wake-only used the wired input with no clipped samples and a peak
chunk RMS of .239 during the first half-second. The model produced 61 wake
windows but labeled none as `ateko` (maximum wake probability .135); there was
no wake event or command capture. This isolated run did not activate. Because
no waveform was retained, the log cannot confirm the utterance or its
pronunciation; this remains an end-to-end missed activation without a verified
acoustic cause.

Candidate T06-wake-only also had unclipped input. The model classified exactly
one of 60 windows as `ateko`, with probability .954; the other windows did not
produce another hit within the three-window smoothing span. No wake event or
command capture occurred. This shows the slower trial produced one strong
wake-class window, but the event policy did not trigger from a single hit.
Since the source waveform is not retained, pronunciation remains unverified.

The saved v5 validation report contains 51 clips, only three labeled as wake,
one independent speaker, and zero seconds of continuous negative audio. It
reports one clip-level false rejection and no clip-level false accepts at the
stored threshold, but it cannot validate continuous false activations or
support changing the two-of-three smoothing rule. The stored probability
threshold is 0.0000774; no threshold or smoothing value was changed during
these laptop trials.

Level-only logs:
`logs/laptop-long-command-20261014-T01-before-live-221724.jsonl` and
`logs/laptop-long-command-20261014-T02-before-immediate-221900.jsonl`;
candidate repeat:
`logs/laptop-long-command-20261014-T02-after-222358.jsonl`; paused baseline:
`logs/laptop-long-command-20261014-T03-before-1500ms.jsonl`; wired candidate:
`logs/laptop-long-command-20261014-T03-after-wired-223631.jsonl` and
`logs/laptop-long-command-20261014-T04-wired-immediate-223900.jsonl`.
Wake-only candidate:
`logs/laptop-long-command-20261014-T05-wake-only-225506.jsonl`;
slower connected wake:
`logs/laptop-long-command-20261014-T06-wake-only-230123.jsonl`.
Prior saved clip-level validation evidence:
`results/wake_v5_regularized_seed43_validation/metrics.json`.
