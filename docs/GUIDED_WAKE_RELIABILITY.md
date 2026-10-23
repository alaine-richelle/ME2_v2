# Guided Wake-to-Command Reliability Upgrade

This is an engineering validation workflow, not a deployment approval. Normal
microphone runtime stays automatic. `--guided-terminal` is fixture-only: it
requires both approved WAV inputs and pauses at start, wake initialization,
read, detection, acknowledgement, command initialization/read/detection,
execution, confirmation, and end.

## Private five-sample calibration

On the Pi, keep recordings and manifests under
`/home/rpiadmin/me2-validation-evidence/wake-calibration/`, never in Git.
Run `me2-wake-private-calibrate --model MODEL --sample WAV` five times and
`--output` under that same root. The tool checks format, duration, silence and
clipping, then uses the active model preprocessing and decision threshold. It
stores only per-sample index/validity/decision plus aggregate counts. Passing
requires five valid samples and at least four complete wake detections. It
never authorizes training; do not tune a model on five samples.

## Feedback and capture safety

Dashboard feedback is local only: wake accepted is 880 Hz, command accepted is
1046 Hz, and rejection/error is 440 Hz. There is no TTS. Playback failure is
nonfatal and cannot make a rejected command actionable. The 150 ms post-wake
guard discards only newly arriving samples after wake; pre-roll remains intact.

## Promotion and collection gates

Collect a consented, speaker-disjoint corpus with a provenance/consent manifest
outside Git. Split speakers before training, reserve held-out speakers for
evaluation, and retain continuous negative sessions separately. Promotion
requires documented wake recall, negative-session false-trigger, endpoint, and
connected-command targets on the same candidate/baseline policy. Fine-tune VCM
only if evaluation of correctly segmented commands demonstrates a need.

## Pi regression matrix

For muted playback then approved normal-speaker trials, record only aggregate
event/endpoint/clipping outcomes. Separately exercise dashboard feedback. For
each baseline, reboot, and hub re-enumeration condition: require clean SHA,
non-numeric device selection, input/output health, supervised session, then
conditional supplied-WAV Step 7. A live wake rejection, device fault, timeout,
clipping, or ambiguity is reported as such and never dispatched as an action.
