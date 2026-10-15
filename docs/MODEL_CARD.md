# Model card — VoxShield synthetic speech detector

## Model

Spectrogram CNN, 913,121 trainable parameters. Input: one-channel 128×128
normalised log-Mel spectrogram of a 3.0 s, 16 kHz window. Output: one logit,
squashed to a score in [0, 1] where higher means more consistent with
machine-generated speech.

Four convolutional blocks (32 → 64 → 128 → 256 channels, 3×3 kernels, batch
normalisation, ReLU, max-pooling, rising dropout), then adaptive average
pooling to 4×4 and a two-layer dense head.

## Intended use

A **supporting signal** inside a layered identity-assurance process for any
organisation that accepts voice as an input channel. It answers one narrow
question: does this audio carry the traces of a speech-synthesis or
voice-conversion system?

## Out of scope

- **It is not an authentication mechanism.** It does not identify a speaker,
  and a low score is not proof of a genuine caller.
- **It must not be the sole basis of an adverse decision** about a person.
- It does not transcribe, and it does not detect what was said.
- It is not evidence in a legal proceeding without expert corroboration.

## Training data

Read `model/saved_models/model_meta.json` for the loaded checkpoint. The field
`training_data_kind` is authoritative:

| Value | Meaning |
|---|---|
| `real_corpus` | Trained on recorded and synthesised speech. |
| `synthetic_demo` | Trained on generated oscillator tones. **Not a detector.** |
| absent | Unknown provenance. Treated as `synthetic_demo`. |

The checkpoint distributed with this repository is trained on ASVspoof 2019
Logical Access (training attacks A01-A06) plus LibriSpeech real and vocoded
speech (`training_data_kind: "real_corpus"`). It is a research prototype and
reaches about 21.9% EER on unseen attacks A07-A19.

Training used 34,204 clips in total:

- ASVspoof 2019 LA training set: 25,380 utterances (2,580 bonafide, 22,800
  spoof, attacks A01-A06).
- LibriSpeech: 8,824 clips (4,412 real, 4,412 vocoded), split by speaker
  (59 train, 14 validation, 73 test).

Evaluation used 71,237 held-out ASVspoof 2019 LA utterances (7,355 bonafide,
63,882 spoof) containing attacks A07-A19 that never appear in training.

## Metrics

Equal Error Rate and ROC-AUC, both threshold-free and both robust to the
corpus's roughly 1:9 class imbalance. Raw accuracy is reported only with an
explicit caveat, because a constant "spoof" answer scores about 90% on this
data while detecting nothing.

The official ASVspoof metric, minimum t-DCF, is **not** computed: it requires
the tandem speaker-verification scores distributed with the challenge.

The decision threshold is the Equal Error Rate operating point measured on the
development partition and then held fixed. It is recorded in the metadata and
read by the serving runtime.

## Known limitations

1. **Domain shift.** Trained on studio-quality read speech. Telephony codecs,
   packet loss and room acoustics degrade performance, sometimes sharply.
2. **Unseen synthesis systems.** Generalisation to synthesisers newer than the
   training corpus is the field's central open problem. Retraining against
   current attacks is an operational requirement.
3. **Language coverage.** ASVspoof LA is English. Behaviour on Sinhala, Tamil
   and code-mixed speech is unmeasured — this is the project's stated research
   gap, not an oversight.
4. **Adversarial evasion.** Deliberate perturbation, added noise or artefact
   masking will reduce detection.
5. **Uncalibrated scores.** The output is a score, not a probability. An 85%
   score does not mean 85 of 100 such calls are spoofed.
6. **Short audio.** Clips under about one second carry too little evidence;
   they are padded, and the result should be treated as low-information.

## Fairness

Detection performance is likely to vary across accents, ages, genders and
recording conditions, because the training corpus does not represent them
equally. This has a direct fairness consequence: a group that is
disproportionately false-flagged faces disproportionate friction in getting
service. **Before any real deployment, measure the error rates per demographic
group and publish them.** This has not been measured here.

## Privacy

Voice is biometric data. The service is stateless: audio is written to a
temporary file only because the decoders require a seekable path, deleted in a
`finally` block, and never logged. No audio, transcript or identifier is
persisted. Deployments should run inside the data controller's own boundary.

## Security of the model artefact

Checkpoints load with `weights_only=True`, so a hostile `.pth` cannot execute
code during deserialisation. Treat any checkpoint from an untrusted source as
untrusted input.

## Maintenance

The threat model moves. A detector trained today degrades as new synthesis
systems appear. Plan for periodic retraining against current attacks, and
monitor the live score distribution — a drift in the distribution of scores on
genuine traffic is the earliest signal that the model has gone stale.
