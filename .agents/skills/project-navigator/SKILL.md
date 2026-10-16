---
name: project-navigator
description: Navigate the ME2_v2 repository, locate the correct implementation and artifact paths, and trace changes across shared audio code, wake-word recognition, voice-command models, tests, configuration, training, evaluation, and inference. Use when exploring this project, finding where behavior lives, assessing change impact, or deciding which files and commands are relevant.
---

# Project Navigator

Navigate from repository intent to the smallest relevant set of files. Treat
`src/me2/` as the implementation source and the top-level `wake-word/`, `vcm/`,
`results/`, and `logs/` directories primarily as datasets, model bundles, and
generated artifacts.

## Start Here

1. Resolve the repository root with `git rev-parse --show-toplevel`.
2. Inspect `git status --short` before assuming files are pristine. Preserve all
   existing user changes and untracked experiment outputs.
3. Read [references/project-map.md](references/project-map.md) for ownership,
   workflow boundaries, and common request-to-file mappings.
4. When the tree may have changed, run:

   ```sh
   python .agents/skills/project-navigator/scripts/project_map.py
   ```

5. Search with `rg` or `rg --files` before opening broad file sets. Exclude
   `.venv/`, caches, binary model bundles, raw audio, and generated results
   unless the task specifically concerns them.

## Trace Work by Behavior

- Start from the CLI entry points in `pyproject.toml` or `src/me2/app.py`.
- Follow imports into shared preprocessing, features, models, bundles, metrics,
  and training code before changing a task-specific module.
- For wake-word work, trace through `src/me2/wake_word/`.
- For voice-command work, distinguish the default intent classifier from the
  opt-in CTC path under `src/me2/vcm/ctc*.py`.
- Check `config/default.json`, relevant tests, README commands, and saved bundle
  compatibility whenever behavior or interfaces change.
- Verify generated metrics from files under `results/`; do not infer observed
  performance from code or documentation.

## Navigation Commands

Prefer focused commands such as:

```sh
rg --files src tests config
rg -n '^(class|def) ' src tests
rg -n 'symbol_or_term' src tests README.md pyproject.toml config
pytest -q
```

Use narrower test selections while iterating, then run the relevant full suite
when the change is complete. Dataset downloads, training runs, microphone
capture, and model replacement can be expensive or externally stateful; run
them only when the user request requires them.
