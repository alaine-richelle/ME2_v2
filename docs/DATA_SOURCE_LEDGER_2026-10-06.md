# Speech data source ledger — 2026-10-06

This ledger records sources considered during the improvement cycle. Only the
pinned ME2 Hugging Face command dataset was incorporated. No source is treated
as permission for unrestricted redistribution.

| Source | Revision / access | Terms noted | Intended use | Disposition |
|---|---|---|---|---|
| `airimonda/ai231-me2-voice-commands` | `bfca0bbebfe17e00d6a685d1da573615d29ecdce` from the local HF cache | Research/education release with component-specific provenance; consult upstream `LICENSE.md` | Command training/evaluation and command-without-wake negatives | 320 official-train clips, at most two per source speaker, added to wake v5 as pending-review train-only negatives |
| Google Speech Commands | Current TensorFlow Datasets source card | CC BY 4.0 on the source card | Generic keyword and near-word negatives | Evaluated, not downloaded; vocabulary is not a verified source of connected `a-teh-ko` positives |
| MUSAN / OpenSLR 17 | Current OpenSLR source card | CC BY 4.0 | Background speech/music/noise mixtures | Evaluated, not downloaded; current failure was unseen-voice recall and calibration, so adding a large noise corpus was not the first controlled change |
| RIR and Noise Database / OpenSLR 28 | Current OpenSLR source card | Apache 2.0 | Room responses and stationary noise | Evaluated, not downloaded; reserved for a later real-microphone augmentation experiment |
| Mozilla Common Voice | Current Mozilla Data Collective source card | CC0 aggregate card; release-specific review still required | Independent ordinary-speech negatives | Evaluated, not separately downloaded; the pinned ME2 dataset already contains provenance-linked Common Voice examples |

Source pages:

- https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands
- https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands/blob/main/LICENSE.md
- https://www.tensorflow.org/datasets/catalog/speech_commands
- https://openslr.org/17/
- https://www.openslr.org/28/
- https://datacollective.mozillafoundation.org/datasets/cmj4o0n2k00lo10qzuaj7u19p

The generated v4/v5 TTS clips use local macOS voices. They remain pending
per-voice listening review. The single previously approved Samantha/rate-175
connected clip remains the only user-approved acoustic anchor.
