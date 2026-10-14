# Deadline implementation report — 2026-10-14

## Outcome

Dashboard action mapping and fail-closed local demonstrations are implemented.
The baseline dashboard configuration is preserved at a one-second onset
deadline; a separate three-second onset-only candidate is available for
controlled validation. The wake and CTC weights, vocabulary, grammar,
acceptance thresholds, smoothing, and cooldown were not changed.

This is not a deployment approval: live microphone trials were not repeated,
the Pi host name did not resolve from this workstation, the selected wake
bundles remain marked `deployment_ready=false`, and the selected command
bundle has a legacy uncalibrated rejection policy with no slot mappings.

## Baseline and artifacts

- Checkout: the local `ME2_v2` project workspace
- Branch/HEAD at start: `master`, `e3aa9f8bedf25aec64a7eeb033bd506ae133a46e`
- The worktree was already dirty with prior capture/runtime changes and
  numerous local experimental artifacts. Those unrelated changes and files
  were not staged or replaced.
- `config/dashboard.json` resolved to
  `wake-word/models/wake_word_v3_binary_seed43_r1.me2` and
  `vcm/models/vcm_ctc_model.me2`. Its old one-second onset configuration was
  retained. `config/runtime.example.json` separately points to wake v5, so
  the dashboard and CLI examples are not artifact-identical.
- Dashboard wake artifact SHA-256:
  `35ec56088979aea4d16f547dfcda047fac969b8f9d46a60176ee1703d853170b`.
  It is a 2,858-parameter TinyAudioCNN and explicitly not deployment-ready.
- Selected CTC artifact SHA-256:
  `b5a358e2815ee2471f29956e63936af87c4359bf28b21c21ab420c21e128eeac`.
  It is a 3,652,572-parameter five-second `CharacterAudioCNN` with 126
  output frames, legacy grammar and uncalibrated acceptance settings. The
  loaded grammar contains no semantic slot values.
- Before changes, file-based inference accepted the reviewed connected wake
  fixture at wake score `0.780` and accepted `vcm/data/test_sample.wav` as
  `play_music`. The combined file gate accepted the same pair. These are
  single-fixture checks, not aggregate quality results.

## Implemented

- Added one canonical action for each of the 19 intents. The dashboard brand
  is “Atecco”; “a-teh-ko” remains the wake pronunciation guide.
- Rejected/system-error results, unknown or disabled mappings, duplicate
  utterance IDs, and missing or unsupported required slots fail closed.
  Legacy saved `display` mappings are upgraded in memory to canonical actions;
  the saved file is not rewritten until the operator saves.
- Implemented local preview behavior for light, player, volume, brightness,
  color, thermostat, time, reminders, timers, alarms, and explicitly simulated
  call/message actions. No physical appliance, call, or message action exists.
- Added an original generated sample melody and simulated ringtone through a
  lazy optional audio-output backend. It opens only after an accepted live
  action; the synthetic demo and tests do not play audio.
- Added configured-city weather lookup with bounded requests, short-lived
  in-process cache, loading/ready/stale/error UI states, and no repository
  credential. Network requests are mocked in tests.
- Added the implementation handoff prompt at
  [`AI231 ME2/ME2_Voice_Command_Dashboard_Mapping_and_Final_Model_Prompt.md`](../AI231%20ME2/ME2_Voice_Command_Dashboard_Mapping_and_Final_Model_Prompt.md).
- Added `config/dashboard.natural-pause.json` and
  `config/runtime.onset-3s.json`. The former changes only onset timeout
  (1000→3000 ms) relative to the dashboard baseline. The existing combined
  natural-pause/endpoint/max-duration candidate remains unpromoted.

## Measurements and checks

The new onset-only replay is saved at
[`results/deadline_capture_timing_onset_only_20261014.json`](../results/deadline_capture_timing_onset_only_20261014.json).
It uses mocked wake decisions and synthetic speech spans except for the
existing `play_music` WAV fixture; it does not model real microphone
pronunciation or ambient false activations.

| Offline replay scenario | 1 s baseline | 3 s onset-only candidate |
| --- | --- | --- |
| Immediate short command | Accepted (`play_music`) | Accepted (`play_music`) |
| 1.5 s pause, short command | Onset timeout | Accepted (`play_music`) |
| 2.0 s pause, short command | Onset timeout | Accepted (`play_music`) |
| 2.0 s pause, 4.6 s synthetic speech span | Onset timeout | Reaches mock command predictor |
| 4.3 s synthetic speech span, no pause | Reaches mock command predictor | Reaches mock command predictor |
| 600 ms internal pause | Accepted | Accepted |
| Short noise / 5.1 s overlong span | Rejected (`command_too_short` / `command_too_long`) | Same explicit rejections |
| No command | Onset timeout | Onset timeout |
| Real `play_music` WAV after 1.5 s inserted pause | Onset timeout | Accepted (`play_music`) |

The 4.3/4.6-second cases are only capture-path synthetic tests; their mock
predictor returns an accepted result when called. They do not show that the
CTC model recognizes a long spoken utterance. The separate prior
100-row replay for the combined 3 s onset + 800 ms endpoint + 5 s maximum
candidate reported one OOS false accept among five OOS examples versus zero
for the baseline. That is not the onset-only candidate, but it is sufficient
reason not to promote the combined profile.

Reproduction from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python scripts/capture_timing_validation.py \
  --baseline-root "$PWD" --source-root "$PWD" \
  --baseline-config "$PWD/config/runtime.example.json" \
  --config "$PWD/config/runtime.onset-3s.json" \
  --audio "$PWD/vcm/data/test_sample.wav" \
  --output "$PWD/results/deadline_capture_timing_onset_only_20261014.json"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/pytest -q
.venv/bin/ruff check src tests
```

Offline dashboard demo:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m me2.dashboard \
  --demo --autostart --log-file logs/atecco-synthetic-demo.jsonl
```

The demo uses synthetic model events and fixed example slots. It does not
open the microphone or speakers. For controlled live testing, the dashboard
candidate is `config/dashboard.natural-pause.json`; do not start its
microphone until the operator has approved the specific trial.

Automated results:

- Full suite: **163 passed**.
- Ruff: **all checks passed**.
- `git diff --check` and Python compile checks passed.
- Weather HTTP, cache/error, player stream, mapping, slot, duplicate,
  rejection, streaming, CTC, and dashboard localhost security tests ran.
  The localhost test was run with local loopback permission; no external
  service was called.
- No model training, threshold calibration, real microphone capture, or
  actual audio playback was performed in this implementation turn.
- The weather service was not contacted; its client and dashboard cache/error
  paths were tested with mocked responses only.

## Raspberry Pi 5 status and next action

The workstation identifies as macOS ARM64, not a Pi. A read-only SSH check to
the documented target `ateko-pi.local` failed with “Could not resolve
hostname”; no Pi performance numbers can be reported. The configurations and
runbook are prepared, but laptop results do not establish Pi performance.

To continue target validation, provide the Pi host name or IP reachable from
this Mac (and confirm the SSH username if it is not `me2`). First run
read-only OS/runtime and artifact-hash checks; then transfer only the reviewed
source/config subset and run file-based tests. Ask separately before each
live microphone or speaker trial, with a visible “GO” cue. Keep the natural-
pause profile a candidate until both OOS rejection and supported long-command
behavior pass on the same validation protocol.
