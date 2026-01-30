Multimodal Emotion & Sentiment Analysis with Video Body Visualization
____________________________________________________________________________

Project Overview

This project is a smart AI system that detects emotions and sentiments from text, audio, and video. It not only predicts what someone is feeling but also gives visual insights into how emotions are expressed, especially through the body in videos.

--------------------------------------------------------------------------------------------------------------------------------------------------

Data Splitting

-----------------

Training: 70%

Testing: 20%

Validation: 10%

--------------------------------------------------------------------------------------------------------------------------------------------------

Data Preprocessing 

----------------------

The system uses three types of inputs:

Text (Speech Transcripts) 

The text is taken from CSV files with the transcript of each dialogue.
 
Transformed into numerical embeddings and passed to the model with the help of DeBERTa tokenizer. 

Audio (Speech) 

The audio files are stored as a .wav file in the dataset folder. 

Processed using Wav2Vec2 to obtain features such as intonation, tone, and patterns of speech. 

Fixed length is used for consistency; the missing audio are set to zeros. 

Visual (Frames from Video) 

The uploaded MP4 file (your video.mp4) is converted into video frames. 

The faces are recognized and the prediction of emotion is done with the help of FER library. 

Incomplete data is filled by using zero tensors instead of missing faces or frames.

--------------------------------------------------------------------------------------------------------------------------------------------------

Figures 

-------------------

Figure 4: Accuracy & Loss Curves 

X-axis: Epoch number (1 → 50)

Y-axis: Accuracy (%) or Loss

Lines:

Training accuracy / loss

Testing accuracy / loss

This gives a clear visual of:

How well the model learns over time

Whether the model overfits (train accuracy rises while test accuracy stagnates)

The stability of the loss function over epochs

Figure 5: Bi-GRU Module Curves

The Bi-GRU module is responsible for capturing temporal dependencies in the fused multimodal features.

Its training and testing accuracy and loss curves are obtained similarly:

Predictions from the Bi-GRU output are compared to true labels per epoch.

Loss and accuracy are recorded over 50 epochs.

These curves help visualize the temporal learning performance of the Bi-GRU sub-module independently.

Figure 6: Confusion Matrix -Emotions. 

Input: Actual (y_true_emo) and predicted (y_pred_emo) emotions.

Process: Number of predictions made per emotion. 

Heatmap plot: rows = true emotions, columns = predicted. 

Darker diagonal: more correct predictions. 

Purpose: Rapidly understand what emotions the model predicts successfully and which ones it is not successful with. 

Figure 7: ROC Curve - Emotions 

Input: True emotions and predicted probabilities (y_score_emo) in one-hot format. 

Process: Find true positive rate and false positive rate of each emotion. 

Plot a line per emotion 

Compute AUC (Area Under Curve) 

Purpose: Indicates confidence in differentiating each emotion; the larger the curve/AUC, the more successful the model performance.

Figure 8: Precision-Recall Curve - Emotions. 

Input: True emotions and predicted probabilities. 

Process: Calculate precision and recall at various thresholds.

Plot recall/precision for each emotion 

Purpose: Applicable in cases where some emotions are not as prevalent; exhibits a trade off between accuracy and coverage. 

Figure 9: Video Body Region Emotion Plot. 

Input: Face-based emotion predictions of FER library, video frames. 

Process: Read frames from the video. 

Detect faces and identify emotion probability per frame. 

Mean probabilities between all frames. 

Apply averaged emotions to body regions (Arms, Hands, Eyes, Mouth, Heart, Lungs etc.) 

Plot a bar graph showing emotion intensity at each body region.

Purpose: Visualizes where emotions are expressed in the body, which allows to understand the non-verbal expression of emotion.
 
Figure 10: Confusion Matrix - Sentiments. 

Input: Actual (y_true_sent) and predicted (y_pred_sent) sentiments. 

Process: Count predictions per sentiment. 

Draw a heat map (like emotions) 

Purpose: Learn whether the system is able to distinguish negative, neutral and positive feelings.

--------------------------------------------------------------------------------------------------------------------------------------------------

Tables

---------------
Table 4: 

The performance of the proposed MESR system on the MELD test set using micro- and macro-averaged metrics to account for class imbalance. 

The micro-average shows recognition performance and it is affected by repeated emotions meaning that there are strong and consistent predictions. 

The macro-average does not distinguish between classes of emotions, there is a high and equal performance in common and rare emotions. 

The anticipated difference in micro and macro scores is a result of the effect on classes frequency, whereas the overall results indicate that MESR is correct, resistance to class imbalance and justice in classes across all categories of emotions.

Table 5: 

The proposed MESR model was evaluated under a supervised multiclass classification setting using a one-vs-rest strategy.

Performance was assessed on a per-emotion basis using confusion-matrix–based metrics, following standard practice in emotion recognition.

For each test sample comprising video, audio, and text features, the model predicted a single emotion label, which was compared with the ground-truth annotation. This procedure was repeated over the entire test set.

True Positive (TP), False Positive (FP), False Negative (FN), and True Negative (TN) values were calculated per class of emotions. Accuracy was an overall classification correctness, Precision and Recall were per-class prediction rates and the F1-score was a balanced score of both.

The Matthews Correlation Coefficient (MCC) was also reported to deal with imbalance in classes, and combines all the elements of the confusion matrix and provides a solid measure of the imbalance in emotion datasets.

Table 6: 

The model was trained on test set during a specific number of epochs and the learning rate was varied in the experiment.

The model predictions of all the samples in tests were created during the evaluation and compared to the ground-truth labels.

Values of each emotion class of the confusion matrix (TP, FP, FN, TN) were computed and then transformed into Accuracy, Precision, Recall, F1-score, and MCC.

This step was carried out with a combination of all the learning rates and epochs and the metrics were recorded to examine how optimization parameters in the model impacted the performance of the model.
--------------------------------------------------------------------------------------------------------------------------------------------------

Main Code

---------------

Step 1: Loading the Dataset 

MELD Dataset class reads a CSV file that contains dialogue, utterance id and labels. 

For each row (utterance) it retrieves: 

Text: Tokenized.

Audio: Loaded, padded, and enhanced. 

Image: loaded using frames folder, transformed. 

Modality mask: Indicates presence of each modality. 

Emotion and Sentiment labels: Transformed into numbers.

Step 2: DataLoader 

DataLoader loads the data in batches and shuffles in the course of training.
 
Enables the model to combine text, audio and video features and process them effectively. 

Step 3: Feature Extraction 

Text: DeBERTa produces a sentence embedding (vector) of the sentence. 

Audio: Wav2Vec2 embeds the audio features. 

Visual: The output of ResNet50 gives the image embedding. 

Step 4: Multimodal Fusion 

Every embedded modality is sent through MISLS (linear transformation to latent space). 

A modality mask randomly eliminates missing inputs and part of inputs during training (modality dropout). 

The features are stacked and subjected to Masked Attention to reflect cross-modality interactions. 

Step 5: Sequence Modeling 

Features are processed in a BiGRU, which is a recurrent network that predicts temporal relations between modalities in the dialogue sequence.
 
Step 6: Multi-Task Prediction 

The model predicts emotion and sentiment at the same time with MultiTaskHead. 

Output: 

emo_out: 7-class emotion prediction.

sent_out: 3-class sentiment prediction.

Example Flow for One Sample 

Input CSV row: 

Text: I cannot believe this has happened! 

Audio: dia12_utt5.wav 

Video frame: dia12_utt5.jpg 

Descriptions: Anger, Sentiment: Negative. 

Process:

Text: DeBERTa - embedding vector

Audio: Wav2Vec2 - embedding vector

Image: ResNet50 - embedding vector

combine with modality mask: Attention - BiGRU - Multi-task head. 

Output prediction: 

Emotion: Anger (predicted probability 0.97). 

Sentiment: negative (probability of 0.95)
