CODE 1 — PREPROCESSING MODULE

(Audio & Visual Data Preparation)

1. Purpose of This Code

This module prepares raw video data from the MELD dataset for multimodal deep learning.
Since neural networks cannot directly operate on raw video streams, this step converts videos into structured audio and visual inputs aligned at the utterance level.

2. What This Code Does
a) Folder Setup

The code automatically creates a standardized directory structure for:

Audio files

Visual image frames

These are organized into:

Training

Development (Validation)

Test splits

This ensures dataset consistency, reproducibility, and prevents data leakage.

b) Audio Extraction

For each utterance video:

Speech audio is extracted

Converted to mono channel

Resampled to 16 kHz

Saved in .wav format

This format is optimal for Wav2Vec2.0-based speech representation learning.

c) Visual Frame Extraction

One representative frame is extracted per video

Saved as a .jpg image

Captures facial expressions and visual context

Using a single frame per utterance is:

Computationally efficient

Consistent with standard MELD-based research practices

d) Dataset Coverage

The preprocessing pipeline is applied independently to:

Training set

Development set

Test set

This ensures strict experimental integrity.

e) Verification

After preprocessing, the code:

Counts generated audio and image files

Displays sample filenames

This confirms successful and complete preprocessing.

3. Why This Step Is Important

Converts raw videos into model-ready inputs

Ensures multimodal alignment at the utterance level

Enables scalable training

Prevents runtime errors during training


4. Output of Code 1

After execution, the system contains:

One .wav audio file per utterance

One .jpg visual frame per utterance

A clean, structured dataset ready for multimodal learning





CODE 2 — MODEL TRAINING MODULE

(Unified Multimodal Emotion & Sentiment Learning)

1. Purpose of This Code

This module implements a unified multimodal deep learning framework for:

Emotion recognition

Sentiment classification

The model learns from text, audio, and visual modalities, while remaining robust to missing or noisy inputs and producing interpretable predictions.

2. Model Inputs

For each conversational utterance:

Text

Dialogue transcript

Tokenized using DeBERTa tokenizer

Encoded via DeBERTa encoder

CLS embedding projected for multimodal fusion

Audio

Speech waveform

Encoded using Wav2Vec2.0

Visual

Extracted image frame

Encoded using ResNet-50

All modalities are synchronized at the utterance level.

3. Modality-Specific Feature Extraction

Each modality is processed using a specialized encoder:

Text Processing (DeBERTa)

Captures semantic meaning and contextual dependencies

Handles complex conversational utterances

Audio Processing (Wav2Vec2.0)

Extracts speech characteristics such as:

Tone

Intonation

Emotional cues

Visual Processing (ResNet-50)

Extracts facial and visual features

Encodes emotional expressions from static frames

4. Modality-Invariant Shared Latent Space (MISLS)

Extracted modality-specific features are projected into a Modality-Invariant Shared Latent Space (MISLS).

This ensures:

Cross-modal feature alignment

Comparable feature scales

Robust learning under missing or incomplete modalities

The model can operate even when:

Audio is unavailable

Visual input is corrupted or missing

5. Emotion-Specific Attention Gate (ESAG)

An Emotion-Specific Attention Gate (ESAG) dynamically regulates the contribution of each modality by:

Conditioning attention on emotional context

Suppressing unreliable or missing modalities

Learning cross-modal emotional dependencies

This mechanism also provides intrinsic interpretability, highlighting modality importance per prediction.

6. Temporal Context Modeling

A Bidirectional GRU (Bi-GRU) is applied to the fused representations to:

Capture conversational flow

Model emotional transitions across utterances

Incorporate contextual dependencies

Predictions are context-aware rather than isolated.

7. Dual-Phase Knowledge Distillation

To improve robustness under missing modalities, the model employs a Dual-Phase Knowledge Distillation strategy:

A full-modality teacher model guides learning

A student model learns to perform reliably with partial inputs

This enables stable inference in real-world multimodal scenarios.

8. Multitask Learning Strategy

The system jointly performs:

Emotion Recognition

Predicts one of seven emotions:

Anger

Disgust

Fear

Joy

Neutral

Sadness

Surprise

Sentiment Classification

Predicts:

Positive

Neutral

Negative

Joint learning improves generalization and stability.

9. Training Optimization Techniques

The training pipeline includes:

Class-balanced loss to address label imbalance

Modality dropout for robustness

Encoder freezing during early epochs

Extended training for convergence

Self-supervised pretraining to improve generalization and reduce labeled data dependency

10. Explainability

Final predictions are interpreted using SHAP (SHapley Additive exPlanations), enabling:

Transparent decision analysis

Identification of influential modalities and features

Trustworthy and interpretable predictions

11. Evaluation Metrics

The system is evaluated using standard metrics:

Macro F1-Score

Unweighted Average Recall (UAR)

Sentiment Macro F1-Score

These metrics provide balanced and realistic performance assessment.

12. Expected Outcomes

Typical performance on the MELD dataset includes:

Strong balanced emotion recognition

High sentiment classification reliability

Stable convergence across epochs

Results are reproducible and suitable for publication.

13. Overall System Flow

Preprocessing converts raw videos into aligned audio and visual inputs

Multimodal encoders extract modality-specific features

MISLS aligns representations

ESAG performs emotion-aware fusion

Bi-GRU captures temporal context

Dual-phase knowledge distillation ensures robustness

SHAP enables explainable predictions
