# 🧠 Deep Learning-Based Brain Tumor Classification Using MRI Images

This project applies deep learning techniques to classify brain tumor MRI images into four categories using Convolutional Neural Networks (CNNs). The goal is to evaluate the effectiveness of transfer learning models—particularly ResNet-50—and compare their performance against a simple baseline CNN for automated medical image analysis.

This work was completed as the final project for MIE1517: Introduction to Deep Learning at the University of Toronto. The project was carried out by a team of four members and includes a progress report, final report, and project presentation documenting the development process, results, and analysis.

## 📌 Project Overview

Accurate diagnosis of brain tumors relies on expert interpretation of MRI scans, a process that is time-consuming and requires years of medical training. This project explores how deep learning models can assist radiologists by automatically classifying MRI images into tumor and non-tumor categories, potentially improving efficiency and consistency in diagnosis.

## 📖 Blog Post
For a detailed walkthrough of the methodology, analysis, and results, see the accompanying Medium article:

[![Medium](https://img.shields.io/badge/Medium-Read%20Article-black?logo=medium)](https://medium.com/@bskky001/deep-learning-based-brain-tumor-classification-using-mri-images-48e2a4643cfe) 

## 📂 Datasets

Two publicly available datasets were used:

### 1️⃣ Kaggle Brain Tumor MRI Dataset 
(https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

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
 
<img width="1933" height="350" alt="image" src="https://github.com/user-attachments/assets/8d5aaf8e-7675-4158-815f-15ae3072a78d" />

### 2️⃣ Roboflow MRI Dataset
(https://www.google.com/url?q=https%3A%2F%2Funiverse.roboflow.com%2Fali-rostami%2Flabeled-mri-brain-tumor-dataset) 

* Used only for testing.

* Total images: 1695

* Same four classes as Kaggle

* Ensures evaluation on completely unseen data

## 🧠 Models Implemented

### ✅ Baseline CNN

* Built from scratch

* 3 convolutional layers + fully connected layers

* Used as a performance benchmark

<img width="961" height="280" alt="image" src="https://github.com/user-attachments/assets/5df62ed4-cb31-4750-968c-f7d3bb0d9751" />


### ✅ ResNet-50 (Best Model)

* Pretrained on ImageNet (transfer learning)

* Final fully connected layer fine-tuned

* Optimized using SGD with momentum

* Hyperparameter tuning applied

  <img width="968" height="364" alt="image" src="https://github.com/user-attachments/assets/de5e26a1-8235-48dc-9c60-66507c834fcc" />


## ⚙️ Training Details

* Optimizer: SGD

* Learning Rate: 0.01

* Momentum: 0.9

* Weight Decay: 1e-4

* Loss Function: Cross-Entropy Loss

* Epochs: 10

* Input Size: 224 × 224

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/c3862b50-d2fa-4168-9805-ba4739b5c037" />

## 📊 Results

### 🔹 Accuracy Comparison

* ResNet-50	95.10%
* Baseline CNN	86.84%
  
### 🔹 Test Performance (ResNet-50)

<img width="513" height="470" alt="image" src="https://github.com/user-attachments/assets/cb063610-3ffe-4e7a-8c14-c7931bb31886" />

* Test Accuracy: 0.9510

* Test Loss: 0.1429

* Strong generalization across all four classes

* Minimal confusion between visually similar tumor types

Confusion matrices, loss curves, accuracy curves, and example predictions are included in the notebooks.

## 🧪 Model Evaluation & Visualization

The project includes:

* Training & validation loss/accuracy plots

* Confusion matrices

* Random examples of correct and incorrect predictions

* Class distribution histograms for all datasets

## ▶️ Demo 

You can watch the demo video: https://drive.google.com/file/d/1HPsSqxOowycgL-8DYcT4narRqJvrNbwK/view?usp=drive_link

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

## 🛠️ Technologies Used

* Python

* PyTorch

* Torchvision

* NumPy

* Matplotlib

* Scikit-learn

* Google Colab

## 📚 Project Structure
Deep-Learning-Based-Brain-Tumor-Classification-Using-MRI-Images/

├── FinalReport_Team2.ipynb

├── MIE1517 Final Project.pptx

├── README.md

## 📝 Conclusion

This project shows how deep learning can be effectively applied to brain tumor MRI classification. By comparing a baseline CNN with a tuned ResNet-50 model, we demonstrate the clear benefits of transfer learning in achieving high accuracy and strong generalization. The results highlight the potential of deep learning models as reliable tools to support automated medical image analysis and provide a strong foundation for future improvements.

## 👤 Authors

This project was completed by a team of four members.

Amine Mazouzi, Basak Kaya, Calise Moldawa, Haisu Wang
