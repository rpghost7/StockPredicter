This project builds and trains a dual-branch LSTM neural network to predict stock price movement (up = 1, down = 0) using both numerical time-series features and event embeddings (text-based signals).

The model combines structured financial indicators with unstructured event data to learn whether a stock is likely to go up or down in the next time step.

🚀 Project Overview

Goal: Binary classification of stock impact (1 = Up, 0 = Down).

Approach:

Numerical branch → 30-day sequences of scaled financial indicators.

Event branch → 30-day sequences of event embeddings (e.g., news/sentiment vectors).

Outputs merged and passed through dense layers for prediction.

Output: Probability between 0–1 (thresholded at 0.5).

🧠 Model Architecture

Numerical branch: LSTM(256) → Dropout(0.1)

Event branch: LSTM(64) → Dropout(0.1)

Concatenation: Merge both branches

Dense layers:

Dense(128, ReLU) → Dropout(0.3)

Dense(64, ReLU) → Dropout(0.3)

Output layer: Dense(1, Sigmoid) → binary classification

Loss: binary_crossentropy
Optimizer: Adam
Metrics: Accuracy (with option to add Precision/Recall/AUC)
