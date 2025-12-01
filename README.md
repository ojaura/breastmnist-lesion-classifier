# BreastMNIST Lesion Classifier
Breast ultrasound lesion classifier using PyTorch & the BreastMNIST dataset.

🔍 Overview

This project trains a convolutional neural network (CNN) to classify malignant vs benign breast ultrasound images.
It uses the BreastMNIST subset from the MedMNIST medical imaging benchmark.

🧠 Machine Learning Pipeline: 
    •    BreastMNIST loading + preprocessing
    •    Train/validation/test split
    •    CNN training with Early Stopping
    •    Performance evaluation (recall, F1-score, confusion matrix)
    •    Saved trained model (.pth)

📁 Repository Structure includes:
 • breast-mnist-lesion-classifier
 • Breast-Lesion-Detection.ipynb
 • models/breastmnist_cnn.pth
 • requirements.txt

🚀 Key Results: 
    •    High lesion recall (important for cancer screening)
    •    Clean PyTorch training + evaluation pipeline
