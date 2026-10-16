# Voice-command misunderstanding analysis

- Rows: 4443
- Final semantic accuracy: 0.8397479180733739
- OOS false acceptance: 7/76 (0.09210526315789473)
- In-scope rejection: 655/4367 (0.14998855049232884)
- Supported-slot joint accuracy: 0/2538 (0.0)

## Most frequent true-command → outcome errors

- color → __rejected__: 93/423 (0.2199)
- message → __rejected__: 66/137 (0.4818)
- time → __rejected__: 49/141 (0.3475)
- volume_up → __rejected__: 44/141 (0.3121)
- weather → __rejected__: 43/141 (0.3050)
- alarm → __rejected__: 42/423 (0.0993)
- brightness → __rejected__: 42/423 (0.0993)
- light_off → __rejected__: 41/141 (0.2908)
- call → __rejected__: 29/141 (0.2057)
- timer → __rejected__: 29/423 (0.0686)
- temperature → __rejected__: 27/423 (0.0638)
- list_reminders → __rejected__: 25/141 (0.1773)
- create_reminder → __rejected__: 24/423 (0.0567)
- volume_down → __rejected__: 23/141 (0.1631)
- play_music → __rejected__: 18/141 (0.1277)

Greedy and trie semantic metrics are reported only when those fields exist in the source predictions. Decoder scores are diagnostic and are not interpreted as calibrated correctness probabilities.
