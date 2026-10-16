# Voice-command misunderstanding analysis

- Rows: 4443
- Final semantic accuracy: 0.5349988746342561
- OOS false acceptance: 56/76 (0.7368421052631579)
- In-scope rejection: 107/4367 (0.0245019464163041)
- Supported-slot joint accuracy: 0/2538 (None)

## Most frequent true-command → outcome errors

- color → play_music: 98/423 (0.2317)
- alarm → list_reminders: 94/423 (0.2222)
- alarm → temperature: 67/423 (0.1584)
- alarm → create_reminder: 57/423 (0.1348)
- light_off → next: 54/141 (0.3830)
- timer → brightness: 43/423 (0.1017)
- stop → play_music: 42/141 (0.2979)
- timer → create_reminder: 42/423 (0.0993)
- call → pause: 41/141 (0.2908)
- stop → next: 36/141 (0.2553)
- alarm → timer: 33/423 (0.0780)
- message → create_reminder: 30/137 (0.2190)
- time → pause: 29/141 (0.2057)
- light_on → call: 28/141 (0.1986)
- light_on → list_reminders: 28/141 (0.1986)

Greedy and trie semantic metrics are reported only when those fields exist in the source predictions. Decoder scores are diagnostic and are not interpreted as calibrated correctness probabilities.
