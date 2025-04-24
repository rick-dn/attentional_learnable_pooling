# Attentional Learn-able Pooling for Human Activity Recognition
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FICRA48506.2021.9561347-blue)](https://doi.org/10.1109/ICRA48506.2021.9561347)

**Summary**

This repository contains the code for our paper "Attentional Learn-able Pooling for Human Activity Recognition" by Debnath et al.

In this paper, we tackle the problem of human activity recognition from RGB videos, which is crucial for enabling natural human-robot interaction. [cite: 1, 2, 3] We address the challenges in this field, such as variations in pose, appearance, background, and lighting, which make it difficult to accurately recognize activities. [cite: 14, 15]

To overcome these challenges, we propose a novel attention-based learn-able pooling mechanism. [cite: 3, 35, 36, 37] Our approach uses a Convolutional Neural Network (CNN) to extract features from video frames and then employs an attention mechanism to focus on the most important features for activity discrimination. [cite: 8, 35, 36, 37] We also introduce a learn-able pooling mechanism that extracts activity-aware spatio-temporal cues using bidirectional Long Short-Term Memory (bi-LSTM) networks and Fisher Vectors (FVs). [cite: 8, 9, 10]

Our learn-able pooling mechanism learns structural information from the hidden states of a bi-LSTM, allowing us to capture long-term temporal dependencies in videos. [cite: 314, 315, 316, 317, 318, 319] We demonstrate that our model achieves state-of-the-art results on challenging human activity recognition datasets. [cite: 322, 323]

**Key Contributions**

* We introduce a novel learn-able Fisher Vector (FV) with activity-aware pooling mechanism. [cite: 35, 36, 37, 49, 50, 51] This mechanism learns structural information from the hidden states of a Long Short-Term Memory (LSTM) network, which improves temporal learning. [cite: 35, 36, 37, 49, 50, 51]
   
* We propose a sequential self-attention-based end-to-end trainable human activity recognition model. [cite: 322, 323] This model integrates our learn-able LSTM FV pooling and achieves state-of-the-art results on two challenging datasets. [cite: 322, 323]

**Important Notes**

* This repository provides the code for our research.
   
* For detailed information about the model architecture, experimental setup, and results, please refer to the original paper.
