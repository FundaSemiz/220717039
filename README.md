
-  # 🌸 Flower Classification using CNNs and Transfer Learning

This project was completed as part of the **SE3508 Introduction to Artificial Intelligence** course, instructed by **Dr. Selim Yılmaz**, Department of Software Engineering at **Muğla Sıtkı Koçman University**, 2025.

> ⚠️ Note: This repository must not be used by students in the same faculty in future years—whether partially or fully—as their own submission. Any form of code reuse without proper modification and original contribution will be considered a violation of academic integrity policies.


---

## 📌 Project Overview

This project demonstrates multi-class image classification using:
- A **Custom CNN**
- **VGG16 as a Feature Extractor**
- **VGG16 with Fine-Tuning**

We also apply **feature visualization** strategies inspired by **Zeiler & Fergus (2014)** and evaluate all models using standard classification metrics.

---

## 📂 Dataset

**Source**: [Flowers Dataset on Kaggle](https://www.kaggle.com/datasets/imsparsh/flowers-dataset/data)

- **Classes**: Daisy, Dandelion, Rose, Sunflower, Tulip
- **Preprocessing**:
  - Resize to 224×224
  - Normalize using ImageNet mean and std
  - Augmentations: horizontal flip, rotation, etc.

---

## 🧠 Models

### 🏗️ Model 1: Custom CNN
- Implemented using PyTorch from scratch.
- Architecture:
  - Conv → ReLU → MaxPool (several layers)
  - Fully connected layers → Softmax
  - the images were set as 128,128
- **Visualizations**:
  - Conv Layer 1: Edge detectors
  - Conv Layer 3: Texture & pattern features
  - Conv Layer 5: Class-specific complex features

### 🧊 Model 2: VGG16 - Feature Extractor
- Pretrained on **ImageNet**
- All convolutional layers **frozen**
- Final classifier replaced for 5-class prediction
- **Only the classifier is trained**
- **No visualizations**, since base features remain unchanged

### 🔧 Model 3: VGG16 - Fine-Tuned
- Pretrained VGG16 with:
  - First convolutional block **frozen**
  - Remaining layers **fine-tuned**
- Final classifier retrained
- **Visualizations**:
  - Conv Layer 1: General features
  - Conv Layer 3: Mid-level floral patterns
  - Conv Layer 5: Complex class-specific patterns

---

## 📊 Results and Comparison

| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Custom CNN             | 73.64%   | 73.60%    | 73.60% | 74%      |
| VGG16 Feature Extractor| 86.65%   | 86.60%    | 83.65% | 87%      |
| VGG16 Fine-Tuned       | 91.82%   | 91.60%    | 91.82% | 92%      |


---

## 🖼️ Visualizations

We use Zeiler & Fergus-style visualization to understand what each layer learns:

- **Model 1 (Custom CNN)**: Layers 1, 3, 5
- **Model 3 (VGG16 Fine-Tuned)**: Layers 1, 3, 5

You can find the outputs in the `Visualizations/` directory.

---

## 📁 Project Structure


