# Local model inventory and retention decision — 2026-10-15

## Result

The checkout intentionally assigns distinct roles rather than globally
promoting one wake artifact. V3 remains the dashboard file-replay fixture; v5
is the final **supervised runtime/trial candidate** because the protected
T01–T13 microphone logs used it and it has higher comparable wake recall. V3
is also scored side by side as a diagnostic and may be a fault-only fallback
after a v5 load/inference fault; a healthy v5 non-wake remains non-wake. The
CTC command bundle remains the command-development model in every
dashboard/runtime profile. Neither wake artifact is deployment-ready.

No candidate meets the wake promotion guardrail. On the shared small synthetic
test cohort, v3 operationally detected 0/3 wake events with 0/48 false accepts;
v5 detected 2/3 but had 4/48 false accepts. The command CTC bundle is retained
as the best measured command implementation, not as a Pi-approved model. The
finalization scope, frozen v5 profile, package audit, and deferred Pi gates are
recorded in `V5_FINALIZATION_2026-10-15.md`.

## Repository and exact resolved paths

- Local branch: `master`, HEAD `20f793d43b664c232f1b25f33a87cf6479c069aa`.
- `origin/master` and remote default were verified at
  `52b38f7fdaf35a0f5b5a8dbbd239f4bfb7700e24`; local `master` is six commits
  ahead. The worktree also contains unrelated prior local changes; this report
  does not stage or commit them.
- Config-relative paths resolve from the config file's directory.

| Entry point/config | Resolved wake artifact | Resolved VCM artifact |
|---|---|---|
| `config/dashboard.json` | `wake-word/models/wake_word_v3_binary_seed43_r1.me2` | `vcm/models/vcm_ctc_model.me2` |
| `config/dashboard.natural-pause.json` | `wake-word/models/wake_word_v3_binary_seed43_r1.me2` | `vcm/models/vcm_ctc_model.me2` |
| `config/runtime.example.json`, `runtime.natural-pause.json`, `runtime.onset-3s.json`, `runtime.pi5.natural-pause.json`, `runtime.v5-final-supervised-20261015.json` | `wake-word/models/wake_word_v5_regularized_seed43.me2` | `vcm/models/vcm_ctc_model.me2` |
| `config/runtime.laptop-dual-wake.json` | v5 primary plus v3 diagnostic/fault fallback | `vcm/models/vcm_ctc_model.me2` |
| `python -m me2.app` without explicit model/config overrides | `wake-word/models/wake_word_model.me2` (legacy five-class fallback) | `vcm/models/vcm_ctc_model.me2` |

The laptop command used a runtime config and explicitly permitted the
unverified wake bundle, so its wake artifact was v5. No Pi inference was run.

| Retained bundle | SHA-256 | Bundle facts |
|---|---|---|
| `wake-word/models/wake_word_v3_binary_seed43_r1.me2` | `35ec56088979aea4d16f547dfda047fac969b8f9d46a60176ee1703d853170b` | `TinyAudioCNN`, 2,858 parameters, `deployment_ready=false`; dashboard file/integration fixture |
| `wake-word/models/wake_word_v5_regularized_seed43.me2` | `d7a4898afdc5a5a52fcd305424df4b62582d7bdeea58451d2a90e62646a49222` | `TinyAudioCNN`, 2,858 parameters, `deployment_ready=false`; T01–T13 protected laptop evidence |
| `vcm/models/vcm_ctc_model.me2` | `b5a358e2815ee2471f29956e63936af87c4359bf28b21c21ab420c21e128eeac` | `CharacterAudioCNN`, 3,652,572 parameters, 13,550,852 bytes; legacy transcript grammar and uncalibrated acceptance policy |

## Measured comparison

Wake rows report accuracy/macro F1 from saved reports, plus the operational
wake-event recall and false accepts. These are tiny clip cohorts, not
false-activations-per-hour measurements. Wake v2 used a different older test
cohort than v3/v5. v3/v5 share the 3-positive/48-negative frozen test rows;
the corpus is synthetic and has only one independent speaker.

| Task / artifact | Evaluation report and split | Accuracy / macro F1 | Wake event recall; false accepts / false rejects | Decision |
|---|---|---:|---:|---|
| Wake dashboard fixture: v3 binary seed43-r1 | `results/wake_v3_binary_seed43_r1_test/metrics.json`, test | 0.9216 / 0.6458 | 0/3; 0/48 / 3/3 | Retain as dashboard fixture only; misses all test wake events |
| Wake protected live-trial model: v5 regularized seed43 | `results/wake_v5_regularized_seed43_test/metrics.json`, test | 0.9020 / 0.6953 | 2/3; 4/48 / 1/3 | Keep artifact and T01–T13 logs; exceeds 2/48 false-accept guardrail, do not promote |
| Wake v2 B corrected-basic | `results/wake_v2_ablation_B_corrected_basic/metrics.json`, old test | 0.7222 / 0.1677 | 0/6; 0/48 / 6/6 | Prune bundle; no wakes detected |
| Wake v2 C source-augmented | `results/wake_v2_ablation_C_source_aug/metrics.json`, old test | 0.7222 / 0.3140 | 0/6; 0/48 / 6/6 | Prune bundle; no wakes detected |
| Wake v2 D regularized | `results/wake_v2_ablation_D_regularized/metrics.json`, old test | 0.6852 / 0.4024 | 0/6; 3/48 / 6/6 | Prune bundle; no wakes and more false accepts |
| Wake v2 E binary | `results/wake_v2_ablation_E_binary/metrics.json`, old test | 0.7222 / 0.5857 | 2/6; 4/48 / 4/6 | Prune bundle; inferior/superseded and not used by protected microphone trials |
| Wake v3 binary seed42 / seed44 | `results/wake_v3_binary_seed42_validation/metrics.json`; `...seed44_validation/metrics.json`, validation | 0.9216 / 0.6458; 0.9020 / 0.6953 | 0/3; 0/48 / 3/3; 1/3; 1/48 / 2/3 | Prune bundles; neither provides a better guarded alternative to the retained dashboard fixture |
| Wake v3 five-class seed42 | `results/wake_v3_five_class_seed42_test/metrics.json`, test | 0.7059 / 0.4165 | 1/3; 3/48 / 2/3 | Prune bundle; worse and less safe than the binary fixture |
| Wake v4 expanded seeds42/43/44 | `results/wake_v4_expanded_seed4{2,3,4}_validation/metrics.json`, validation | accuracy 0.9412–0.9608 / macro F1 0.4848–0.7699 | 0/3; 0/48 / 3/3 for all seeds | Prune bundles; high accuracy is explained by missing every positive |
| Wake v5 command-negative seed42 | `results/wake_v5_command_negatives_seed42_validation/metrics.json`, validation | 0.8824 / 0.7167 | 0/3; 0/48 / 3/3 | Prune bundle; missed every wake |
| Wake v5 width16 seed43 | `results/wake_v5_width16_regularized_seed43_validation/metrics.json`, validation | 0.9804 / 0.9233 | 0/3; 0/48 / 3/3 | Prune bundle; impressive aggregate scores still mean all wakes were rejected |
| VCM GitHub classifier baseline | `results/reproduced_baseline_test_2026-10-05/metrics.json`, official test, 4,443 rows | 0.5350 / 0.4062 | OOS false accepts 56/76; in-scope false rejects 107/4,367 | Preserve as baseline; 8,052 parameters |
| VCM selected CTC + corrected trie | `results/reproduced_legacy_ctc_corrected_decoder_with_greedy_test_2026-10-05/metrics.json`, same official test | 0.8911 / 0.8682 | OOS false accepts 20/76; in-scope false rejects 316/4,367 | Retain selected CTC fixture; better intent metrics, but OOS acceptance and slot grammar remain limitations |
| VCM CTC calibrated-policy candidate | `results/reproduced_legacy_ctc_calibrated_test_2026-10-05/metrics.json`, same official test | 0.8397 / 0.8414 | OOS false accepts 7/76; in-scope false rejects 655/4,367 | Prune bundle; safety improves, but fails predeclared F1/coverage guardrails and rejects 339 more in-scope rows |
| VCM compact CTC preliminary | `results/vcm_compact_ctc_preliminary_2026-10-05_test/metrics.json`, same official test | 0.0171 / 0.0017 | OOS false accepts 0/76; in-scope false rejects 4,367/4,367 | Prune bundle; reject-all failure, not a usable model |

The CTC baseline comparison is not capacity-matched: the selected CTC has
3,652,572 parameters versus 8,052 for the classifier. The VCM table uses the
reproduced report with explicit corrected-trie and acceptance metadata. A
separate older `results/vcm_ctc_test/metrics.json` reports 0.9356 accuracy,
0.9095 macro F1, 22/76 OOS false accepts and 140/4,367 in-scope rejects, but
does not bind the result to a model hash, decoder version, or acceptance policy.
It is retained as historical evidence and is not substituted into the matched
comparison.

## Bundle pruning and preserved evidence

The following exact untracked candidate bundles are removed from the model
directories: wake v2 B/C/D/E; v3 seed42, seed44 and five-class seed42; all
three v4 expanded seeds; v5 command-negative seed42 and width16 seed43; VCM
compact preliminary, calibrated-policy candidate, and CTC smoke bundle.
The saved metrics, prediction tables, reports, epoch logs, training/data
partitions, calibration outputs, host benchmarks and historical experiment
reports remain in place. Old reproduction commands may need to regenerate a
pruned bundle before they can load it.

Retained files include:

- GitHub baseline pickles `wake-word/models/wake_word_model.pkl` and
  `vcm/models/vcm_model.pkl` (both tracked at the upstream baseline revision).
- Safe migrated/fallback baseline bundles `wake-word/models/wake_word_model.me2`
  and `vcm/models/vcm_baseline_model.me2`.
- Trusted CTC migration input `vcm/models/vcm_ctc_model.pkl`.
- Dashboard fixture v3, protected live-trial model v5, and selected CTC model.
- All referenced laptop logs and all metric/report evidence listed above.

The extra baseline files are deliberate: `.pkl` preserves the exact upstream
GitHub baseline for provenance, while `.me2` is the safe-load runtime form;
the old CTC pickle is retained only as a trusted migration input. No pickle is
loaded in normal inference.

The VCM calibrated binary is removed, but the result remains auditable from
the official-test metrics/predictions and train-only calibration report. Its
rejection tradeoff is insufficient reason to keep a second active candidate
bundle. The smoke artifact had no quality standing and its benchmark record
remains.

## Promotion and deployment status

The only retained dashboard wake fixture is v3, but its zero event recall
means it is not a reliable live wake detector. V5 has better test wake recall
but violates the false-accept guardrail and is the final supervised
runtime/trial candidate, not a promotion. Do not promote either on this
evidence, relax thresholds, or describe the system as deployment-ready.

The selected CTC model remains the command-development fixture because its
matched official-test metrics materially exceed the preserved classifier
baseline. Its 20/76 OOS false accepts, legacy grammar without slot metadata,
and unverified Pi performance are explicit release blockers. Raspberry Pi 5
performance and live long-command acceptance remain unmeasured.
