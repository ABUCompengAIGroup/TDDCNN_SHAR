# TDDCNN_SHAR
This project presents a Time-Distributed Dilated CNN with projection shortcut-based deep learning model for suspicious human activity recognition (SHAR).
# Deep Learning-Based Human Activity Recognition System using Time-Distributed Dilated Convolution and Projection Shortcut Residual Learning

## 📌 Overview

This project presents a deep learning-based Human Activity Recognition (HAR) system designed for intelligent surveillance applications. The model detects suspicious and abnormal human activities from video sequences using a hybrid spatiotemporal deep learning framework.

The proposed architecture integrates Time-Distributed Dilated Convolution for spatial feature extraction and Projection Shortcut Residual Learning combined with Bidirectional Long Short-Term Memory (BiLSTM) networks for temporal dependency modeling.

---

## 📊 Dataset

This study uses the *Suspicious Human Activity Recognition* dataset introduced by Mohamed Zaidi et al. (2024) in IEEE Access.

**Reference:**
M. Mohamed Zaidi et al., “Suspicious Human Activity Recognition From Surveillance Videos Using Deep Learning,” *IEEE Access*, vol. 12, pp. 105497–105510, 2024.
https://doi.org/10.1109/ACCESS.2024.3436653

The dataset consists of labeled surveillance video clips representing different human activity classes. Each video is processed into fixed-length frame sequences for training and evaluation.

---

## 🧠 Methodology

The proposed model combines the following components:

* **Time-Distributed Convolutional Neural Networks (CNNs)** for spatial feature extraction from video frames
* **Dilated Convolution** to expand the receptive field efficiently
* **Projection Shortcut Residual Learning** to improve gradient flow and training stability
* **Bidirectional LSTM (BiLSTM)** to capture forward and backward temporal dependencies

This hybrid architecture enables robust learning of both spatial and temporal patterns in video sequences.

---

## ⚙️ Training Configuration

* Input shape: (20 frames, 64×64 resolution, RGB)
* Optimizer: Adam
* Learning rate: 0.0001
* Loss function: Categorical Crossentropy
* Batch size: 16
* Epochs: 100 (with Early Stopping)
* Validation split: 20%

---

## 📈 Results

The proposed model achieved the following performance:

* **Accuracy:** 96.50%
* **Precision:** 97.00%
* **Recall:** 96.00%
* **F1-score:** 96.00%

The confusion matrix and training curves are available in the `results/` directory.

---

## 📁 Repository Structure

```
TDD-PRL-BiLSTM-Project/
│
├── src/
│   ├── model.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   └── final_model.keras
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_curves.png
│
├── notebooks/
│   └── HAR_training.ipynb
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train model

```bash
python src/train.py
```

### Evaluate model

```bash
python src/evaluate.py
```

---

## 💾 Pretrained Model

The trained model is stored in:

```
models/final_model.keras
```

This allows direct evaluation without retraining.

---

## 🔁 Reproducibility Note

Due to randomness in deep learning training (initialization, data augmentation, and hardware differences), results may slightly vary across runs. The provided pretrained model ensures reproducible evaluation of reported results.
