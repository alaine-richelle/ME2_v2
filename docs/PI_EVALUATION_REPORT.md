# Raspberry Pi held-out evaluation report

`me2-pi-report` converts a fixed ground-truth manifest, ordered structured Pi
events, and Pi telemetry into the required six-section report. It never uses
development-host benchmarks to fill missing Pi fields.

## Versioned command schema

`config/command_schema_v1.json` records the 19 intent names and the 93 unique,
non-empty variation strings in the pinned official test and holdout splits of
`airimonda/ai231-me2-voice-commands` revision
`bfca0bbebfe17e00d6a685d1da573615d29ecdce`.

The pinned train split contains 98 intent/variation pairs because five slotted
intents also have an empty variation label. Empty labels are not observable
command identities and are excluded from the 93-command reporting schema.
Intent correctness therefore cannot be reused as 93-command correctness.

## Inputs

The ground-truth CSV requires one row per trial:

```text
clip_id,expected_intent,expected_command,expected_slots,out_of_scope,speaker_id,source,wake_eligible,command_end_ms
```

- `expected_command` is the exact variation ID from the schema.
- `expected_slots` is a JSON object such as `{"value":"20 percent"}`.
- OOS may use `out_of_scope=true`; its expected class becomes `REJECT`.
- `command_end_ms` must use the same clock as event `timestamp_ms`. An event may
  instead provide `relative_ms`, measured from the annotated command end.

The Pi event log is JSONL. Every event requires `clip_id`, `event_type`, and a
common-clock `timestamp_ms` or command-end-relative `relative_ms`.

```json
{"clip_id":"test-001","event_type":"wake_detected","relative_ms":-350}
{"clip_id":"test-001","event_type":"command","relative_ms":182,"intent":"brightness","variation":"Brightness 20 percent","slots":{"value":"20 percent"},"infer_ms":34.2,"audio_ms":1900}
{"clip_id":"test-002","event_type":"reject","relative_ms":240,"accepted":false}
```

Supported decision event types are `command`, `accepted`, `reject`, `rejected`,
and `decision`. A `decision` uses its `accepted` field. Debug messages and wake
acknowledgments are not command decisions. Unknown command names are retained
and scored wrong.

The scorer uses the first structured decision from command end through the
configured deadline. It never selects a later answer because it matches the
reference. No decision becomes `NO_RESPONSE`; classification maps that to
`REJECT` while pipeline diagnostics retain the distinction. Every emitted
command, including later, malformed, premature, and late commands, remains
available for fire and misfire accounting.

Telemetry CSV accepts these sample columns:

```text
timestamp,cpu_temperature_c,cpu_clock_mhz,load_1m,load_5m,load_15m,system_cpu_percent,assistant_cpu_percent,system_ram_used_mib,assistant_rss_mib,throttling_flags
```

Run metadata is JSON and records the run ID, student fields, test time/timezone,
connection types, holdout provenance, code/model/grammar hashes or versions,
acceptance settings, Pi specifications, telemetry method, response window, and
aggregate CPU/test-duration values. `config/pi_report_pending.json` is an
explicit not-run example; it is not a test result.

## Generate

```sh
python -m me2.pi_report \
  --metadata pi-run/run_metadata.json \
  --schema config/command_schema_v1.json \
  --manifest pi-run/ground_truth.csv \
  --events pi-run/events.jsonl \
  --telemetry pi-run/telemetry.csv \
  --output reports/<run_id>
```

The output includes:

- `report.md` with exactly Header, Classification, Slot values, Raspberry Pi,
  Most frequent confusions, and Per-intent scores, in that order.
- `metrics.json` and normalized `predictions.csv`.
- Raw normalized `telemetry.csv`, derived `telemetry_metrics.csv`, classification,
  slot, confusion, per-intent, and source-breakdown CSV tables.
- `input_references.json` with absolute paths and SHA-256 hashes rather than
  redundant copies of the source event log.

The scorer reports Wilson 95% intervals for clip accuracy and OOS false
acceptance. When at least two non-empty speaker IDs are present, it also reports
a deterministic speaker-grouped bootstrap interval. Macro scores average
per-class metrics over ground-truth classes represented in the run. F2 is
computed per class before averaging.

Slot reporting is conditional on a correct primary intent and separately
reports joint intent/slot accuracy. Alarm errors use circular minutes over a
24-hour clock; timer errors use seconds; brightness uses percent; temperature
retains the dataset's degree scale. Categorical slots do not receive fabricated
numeric distances. Phonetic distance remains unavailable until a versioned,
language-appropriate phonemizer is selected.

## Current authoritative artifact

`reports/pi-evaluation-pending-2026-10-06/report.md` is deliberately marked
`not_run`. No Raspberry Pi 5, held-out Pi trial log, ground-truth trial manifest,
or Pi telemetry was available in this workspace. Host benchmark files under
`results/` are not substituted for those missing measurements.
