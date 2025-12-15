# 🧠 Deep Learning-Based Brain Tumor Classification Using MRI Images

This project applies deep learning techniques to classify brain tumor MRI images into four categories using Convolutional Neural Networks (CNNs). The goal is to evaluate the effectiveness of transfer learning models—particularly ResNet-50—and compare their performance against a simple baseline CNN for automated medical image analysis.

## 📌 Project Overview

Accurate diagnosis of brain tumors relies on expert interpretation of MRI scans, a process that is time-consuming and requires years of medical training. This project explores how deep learning models can assist radiologists by automatically classifying MRI images into tumor and non-tumor categories, potentially improving efficiency and consistency in diagnosis.

## 📂 Datasets

Two publicly available datasets were used:

### 1️⃣ Kaggle Brain Tumor MRI Dataset (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

* Used for training and validation.

* Total images: 7023

* Train / Validation split: 80% / 20%

* Training: 5712 images

* Validation: 1311 images

* Classes:

  * glioma

  * meningioma

  * pituitary

  * notumor

### 2️⃣ Roboflow MRI Dataset (https://www.google.com/url?q=https%3A%2F%2Funiverse.roboflow.com%2Fali-rostami%2Flabeled-mri-brain-tumor-dataset) 

* Used only for testing.

* Total images: 1695

* Same four classes as Kaggle

* Ensures evaluation on completely unseen data

## 🧠 Models Implemented

### ✅ Baseline CNN

* Built from scratch

* 3 convolutional layers + fully connected layers

* Used as a performance benchmark

### ✅ ResNet-50 (Best Model)

* Pretrained on ImageNet (transfer learning)

* Final fully connected layer fine-tuned

* Optimized using SGD with momentum

* Hyperparameter tuning applied

## ⚙️ Training Details

* Optimizer: SGD

* Learning Rate: 0.01

* Momentum: 0.9

* Weight Decay: 1e-4

* Loss Function: Cross-Entropy Loss

* Epochs: 10

* Input Size: 224 × 224

## 📊 Results

### 🔹 Accuracy Comparison

* ResNet-50	95.10%
* Baseline CNN	86.84%
  
### 🔹 Test Performance (ResNet-50)

* Test Accuracy: 94.69%

* Test Loss: 0.1604

* Strong generalization across all four classes

* Minimal confusion between visually similar tumor types

Confusion matrices, loss curves, accuracy curves, and example predictions are included in the notebooks.

## 🧪 Model Evaluation & Visualization

* The project includes:

* Training & validation loss/accuracy plots

* Confusion matrices

* Random examples of correct and incorrect predictions

* Class distribution histograms for all datasets

## 🧪 Demo (Inference)

You can test the trained model on a new image:

1. Upload resnet50_best_model.pth

2. Upload a test image (e.g., demo_image.jpg)

3. Run the demo cell

The output will display the image and the predicted tumor class.

## 🔮 Future Work

* Fine-tune deeper layers of ResNet-50

* Train for more epochs

* Experiment with advanced architectures (Xception, EfficientNet)

* Apply data augmentation

* Include additional evaluation metrics (F1-score, ROC-AUC)

* Explore multi-sequence MRI data

## 📌 Technologies Used

* Python

* PyTorch

* Torchvision

* NumPy

* Matplotlib

* Scikit-learn

* Google Colab


  
