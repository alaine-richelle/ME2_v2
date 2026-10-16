# AI231 ME2 voice-command dashboard mapping and final-model implementation prompt

Act as a senior speech/ML systems engineer responsible for integrating the
ME2 wake-word and CTC voice-command system with the Atecco dashboard and
validating it on a Raspberry Pi 5.

## Current checkout status — 2026-10-15

The 19-intent canonical action map and safe local demonstration behaviors are
implemented in `src/me2/dashboard.py`; mapping, rejection, slot, duplicate,
weather, and local audio behavior have automated coverage. Do not replace
these with a second dispatcher or mutate a user's saved mappings while
auditing them. The selected CTC bundle is a development fixture, not a
deployment approval. The retained wake artifacts are not approved for
unattended use; see `docs/MODEL_SELECTION_2026-10-15.md`.

The current one-second onset profile is the baseline. A three-second onset
profile has replay evidence but has not passed a new approved live laptop
trial. Keep it a candidate until matched evidence supports promotion. The
exact Pi target has not yet been contacted. Before any SSH connection, obtain
the operator's reachable host/IP, SSH username, and explicit approval for
that connection. Start with read-only inventory only. Request separate
approval before transferring files or starting live microphone/speaker
trials. Never request or place SSH passwords, private keys, or tokens in this
repository.

## Objective

Implement and test the path from an accepted voice command to one safe,
visible dashboard action. The dashboard brand is **Atecco**. The spoken wake
word remains the connected pronunciation “a-teh-ko”; the application label
may remain `ateko`.

Inspect the current code, model bundles, configuration, grammar, reports, and
worktree before changing anything. Preserve existing user changes and baseline
artifacts. Treat `--demo` as a synthetic state-flow demonstration, not model
evidence.

## Acceptance and dispatch contract

- Use one canonical mapping for exactly these 19 intents:

  | Intent | Canonical local action |
  | --- | --- |
  | `play_music` | Play the locally generated sample track |
  | `weather` | Fetch current weather for the configured city |
  | `time` | Display host-local time |
  | `light_on`, `light_off` | Update simulated light state |
  | `pause`, `stop`, `next` | Control the sample player |
  | `volume_up`, `volume_down` | Adjust bounded demo-player volume |
  | `call` | Show/play a simulated ringtone; never place a call |
  | `message` | Display a demonstration message; never send it |
  | `list_reminders`, `create_reminder` | Display or update demo reminders |
  | `timer`, `alarm` | Create demo UI entries only |
  | `temperature` | Update simulated thermostat state |
  | `brightness`, `color` | Update supported dashboard values |

- Dispatch only a result that is accepted, actionable, mapped, enabled, and
  free of system errors. Rejections, unknown intents, disabled actions,
  duplicates, unsupported/missing slots, and action failures must not cause
  unintended dashboard or physical-device changes.
- Validate a required slot against values explicitly supported by the loaded
  grammar. Never infer a contact, number, time, color, or reminder text from a
  transcript. The retained `legacy-character-audio-cnn-v1` artifact has no
  semantic slot values; report that limitation and fail closed for those
  live actions.
- Track recognition acceptance separately from action status. Keep simulated
  `call` and `message` behavior visibly labeled as demonstrations.
- Preserve the current model as the active model unless a candidate improves
  results on identical permitted partitions, decoder, and acceptance policy.
  Keep one runnable baseline in a separate reference location; do not delete
  or overwrite experimental user artifacts.

## Dashboard behaviors and failure handling

- Implement a local generated sample melody with play, pause, stop, next, and
  bounded volume controls. Open the output device only after an accepted
  action; handle missing devices as an action failure. Do not play audio at
  dashboard startup or in the synthetic demo.
- Show current time, simulated light state, brightness, color, and
  thermostat value. Show reminders, timers, and alarms as local UI data only.
- Use a configured city for weather. Display city, observation time, loading
  state, and a clear service error or cached/stale observation if the network
  is unavailable. Keep credentials out of the repository; prefer a provider
  that needs no credential for this demo. Bound network timeouts and cache
  only in process memory.
- Log safe structured summaries for rejected, disabled, duplicate, unknown,
  and failed actions. Do not log raw audio, credentials, or private
  transcripts by default.
- Retain offline speech inference: normal startup must not download models or
  datasets. Weather is optional network functionality, not an inference
  dependency.

## Capture and model handling

- Preserve the existing wake detector and CTC/trie command path. Do not
  lower wake or command acceptance thresholds to hide the observed
  `no terminal candidate` or missed-wake failures.
- Compare a natural-pause setting by changing onset timeout alone first;
  preserve pre-roll, minimum speech, endpoint silence, maximum artifact
  duration, wake smoothing, decoder, and rejection settings for that
  comparison. The current one-second onset profile remains the baseline; the
  three-second profile is a candidate until evidence supports promotion.
- Verify command capture, endpointing, duration cap, feature/output lengths,
  grammar terminals, decoder diagnostics, and complete CTC frames before
  considering retraining. A duration limit must reject explicitly, never
  silently crop.
- Keep the command artifact's frontend, character vocabulary, blank index,
  grammar, normalization, and acceptance policy coupled to its weights.
  Never claim a command is correctly recognized just because it is a
  grammar terminal.

## Required tests

- Exercise each of the 19 accepted intents from structured result through
  dispatcher to visible dashboard state.
- Verify rejected speech, unknown intent, system error, disabled mapping,
  duplicate utterance ID, absent slot, unsupported slot, weather timeout,
  missing audio output, and shutdown all fail safely.
- Test weather success, loading, timeout/error, cached/stale display, and
  bounded response parsing with mocked HTTP; tests must not require internet.
- Test exact current wake/command model loading and file-based inference
  before interactive trials.
- Replay silence, invalid wake, short command, natural pause, long supported
  command, immediate command, ordinary speech, unsupported speech, and
  repeated wake. Record capture end reason, frames, decoder category, result,
  and action status.
- Test Pi configuration validation, offline model startup, audio-device
  faults, bounded recovery, and clean shutdown. Use only a mock/local
  dispatcher.

## Raspberry Pi 5 release gate

Validate the exact frozen laptop candidate on the actual Pi 5. Record ARM64
OS, Python/runtime versions, artifact hashes, CPU, peak RSS, p50/p95 wake-hop
processing and post-endpoint command latency, temperature/throttling, stream
duration, overflows/dropped samples, wake recall, false activations per hour,
command acceptance/rejection, and duplicate events. Wake-hop processing must
keep pace with the configured hop with no sustained queue growth. Do not call
the system Pi-ready unless live wake, long-command, OOS rejection, and
continuous-stream checks pass.

Obtain operator approval before each real microphone capture or speaker
playback trial. Give the exact phrase/condition and an audible/visible “GO”
cue, then wait for “DONE”. Never execute real calls, messages, GPIO actions,
or shell commands.

Do not assume a host name from an old runbook is the operator's current Pi.
When Pi access details or permission are missing, finish file-based tests,
prepare the versioned Pi runbook and report the exact remaining gate instead
of attempting SSH.

## Deliverables

Provide a matched baseline/candidate report; per-intent mapping and slot
coverage; files/commands/hashes; focused and full test results; laptop and
Pi runbooks; Pi measurements or a precise statement that hardware validation
is pending; and remaining data/model limitations. Keep generated output and
logs separate from final artifacts. Review and stage only task-related
changes. Commit only after the relevant tests pass and the staged diff has
been checked for private data, raw recordings, credentials, and unrelated
user work.
