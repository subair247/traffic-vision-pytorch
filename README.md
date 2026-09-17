# 🚦 TrafficVision: Traffic Sign Classifier

A deep learning-based traffic sign recognition system built from scratch using **PyTorch**. The model classifies 32x32 RGB traffic sign images into 10 distinct categories with high validation accuracy.

---

## 📋 Project Overview
Building an automated traffic sign classifier for smart city applications. This project utilizes a custom Convolutional Neural Network (CNN) architecture trained on pre-loaded dataset tensors, applying data scaling, tensor permutation, and multi-class cross-entropy optimization.

---

## 🛠️ Tech Stack & Libraries
* **Python**
* **PyTorch** (`torch`, `torch.nn`, `torch.optim`, `torch.utils.data`)
* **NumPy**

---

## 🏗️ Model Architecture
The custom CNN (`TrafficCNN`) consists of:
1. **Feature Extraction Block:**
   * 2x Convolutional Layers (`nn.Conv2d`) with ReLU activation and Padding.
   * 2x Max Pooling Layers (`nn.MaxPool2d`) for spatial dimension reduction.
2. **Classification Block:**
   * Flatten layer.
   * Fully Connected (`nn.Linear`) hidden layer with ReLU activation.
   * Final Linear output layer producing **10 classes**.

---

## 🚀 Getting Started & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/traffic-vision-pytorch.git]
   cd traffic-vision-pytorch
   ```
Install dependencies:
  ```
  pip install -r requirements.txt
  ```

Run the notebook or script:

  Ensure your traffic_signs.npz dataset is in the working directory and execute the script/Jupyter notebook.

📊 Results
Loss Function: nn.CrossEntropyLoss

Optimizer: optim.Adam

Validation Accuracy: Achieved ~99% accuracy on the validation split.
