# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.10+ `src/`-layout package for Raspberry Pi-targeted wake-word
and voice-command recognition. Shared audio, feature extraction, model,
streaming, bundle, and CLI code lives in `src/me2/`. Task-specific pipelines are
in `src/me2/wake_word/` and `src/me2/vcm/`; follow shared callers when changing
audio contracts, feature shapes, labels, or bundle metadata. Tests are in
`tests/`, configuration in `config/`, and operational documentation in `docs/`.

`wake-word/` and `vcm/` hold local data and versioned model artifacts.
`results/`, `logs/`, and `reports/` are experiment evidence or generated output,
not implementation source. Avoid broad searches of datasets, binary `.me2` or
legacy `.pkl` artifacts, and `build/`.

## Build, Test, and Development Commands

Run commands from the repository root. Create a local environment and install
development dependencies with:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
source .venv/bin/activate
```

Use `pytest -q` for the complete regression suite, or target a file while
iterating: `pytest -q tests/test_streaming.py`. Run
`ruff check src tests` before submitting Python changes. For uninstalled
module commands, use `PYTHONPATH=src python -m me2.vcm.audit`; editable
installation provides the `me2-*` console commands.

## Coding Style & Naming Conventions

Use four-space indentation, type annotations where they clarify interfaces, and
Ruff's configured 100-character line limit. Keep modules and functions in
`snake_case`, classes in `PascalCase`, and tests named
`test_<behavior>.py` / `test_<behavior>()`. Prefer small deterministic helpers
and explicit failure handling: inference and decoding must fail closed rather
than silently inventing a result.

## Testing & Model Safety

Add focused pytest coverage beside related tests for behavior changes,
especially audio boundaries, model loading, streaming recovery, privacy, and
rejection paths. Do not tune on official test or holdout data. Keep training,
validation, calibration, and test partitions separate; calibrate thresholds
without altering acoustic weights. Normal loading must use safe `.me2` bundles;
only migrate trusted legacy `.pkl` files with the explicit migration command.

## Model Artifact Promotion & Cleanup

Before retaining, replacing, or deleting a model artifact, inspect the saved
metrics and reports for that candidate and the current baseline/selected model.
Record which split, data revision, acceptance policy, and promotion guardrails
the comparison uses; do not choose by raw accuracy alone. Train and evaluate
candidates at separate versioned paths. Overwrite the canonical selected-model
path only after a candidate demonstrably improves the comparable evaluation
and meets the documented safety, rejection, compatibility, and deployment
guardrails; a higher value on one metric is insufficient if rejection or other
promotion metrics regress. On promotion, replace the canonical selected bundle
atomically and retain the prior selected bundle as a recoverable baseline or
archive, together with the promoted artifact's provenance and comparison
metrics. Otherwise keep the incumbent and leave the candidate unpromoted.

After experiments, remove only exact trial artifacts that are demonstrably
unused or superseded, and only when cleanup is authorized. Preserve the
reproducible baseline, selected/runtime fixtures, anything used for protected
live-microphone evidence, and historical metrics, predictions, logs, reports,
and provenance cited by documentation. A stale experiment bundle may be
pruned while its evidence remains if its metrics fail selection, its binary is
not needed for a protected replay, and the cleanup record explains how to
regenerate it. Before cleanup, trace each candidate through live-microphone
trials, runtime-validation reports, documentation, result metadata,
configurations, and code references. Check Git status and the repository's
current remote/default-branch baseline; never remove that baseline
implementation. Preserve user changes. For ambiguous or user-owned files,
present the exact candidate paths and rationale and get direction before
moving or deleting them. Record original paths or deletion disposition,
checksums, reasons, metric evidence, and references checked in the cleanup
ledger.

## Commits & Pull Requests

Use short, imperative commit subjects. Existing history favors Conventional
Commit-style prefixes such as `feat:`, `fix:`, and `docs:`. Keep a commit
focused; do not mix generated logs, results, datasets, or model replacements
with unrelated code. Pull requests should state the behavior change, tests run,
configuration or artifact compatibility implications, and include measured
results or screenshots when dashboard/runtime behavior changes.
