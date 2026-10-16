# Voice-command misunderstanding analysis

- Rows: 4443
- Final semantic accuracy: 0.8910645959936979
- OOS false acceptance: 20/76 (0.2631578947368421)
- In-scope rejection: 316/4367 (0.07236088848179528)
- Supported-slot joint accuracy: 0/2538 (0.0)

## Most frequent true-command → outcome errors

- time → __rejected__: 35/141 (0.2482)
- color → __rejected__: 35/423 (0.0827)
- alarm → __rejected__: 30/423 (0.0709)
- weather → __rejected__: 28/141 (0.1986)
- volume_up → __rejected__: 28/141 (0.1986)
- brightness → __rejected__: 22/423 (0.0520)
- light_off → light_on: 17/141 (0.1206)
- call → __rejected__: 17/141 (0.1206)
- light_off → __rejected__: 15/141 (0.1064)
- timer → __rejected__: 15/423 (0.0355)
- color → light_on: 15/423 (0.0355)
- list_reminders → __rejected__: 13/141 (0.0922)
- play_music → __rejected__: 12/141 (0.0851)
- stop → __rejected__: 12/141 (0.0851)
- message → __rejected__: 11/137 (0.0803)

Greedy and trie semantic metrics are reported only when those fields exist in the source predictions. Decoder scores are diagnostic and are not interpreted as calibrated correctness probabilities.
