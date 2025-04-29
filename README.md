# 📑 Brain Tumor Classification System

[![Python 3.10.17](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow 2.12](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://www.tensorflow.org/)
[![IEEE Format](https://img.shields.io/badge/Format-IEEE-blueviolet.svg)](https://ieeeauthorcenter.ieee.org/)

Deep learning system for automated brain tumor classification from MRI scans, implementing state-of-the-art CNN architectures with clinical validation metrics.

## 📅 Project

Developed as part of a study for the automated classification of brain tumors in MRI images. The following models were implemented and compared:

- Custom Models: Basic CNN and Enhanced CNN
- Pre-trained Models: VGG16 and MobileNetV2 (Transfer Learning)

Main results:

- Best model: Pre-trained VGG16
- Achieved accuracy: 91% on independent test data

## 🧠 Project Structure

```
📂 brain-tumor-classification
├── data/
│   ├── raw/
│   │   ├── training/
│   │   ├── testing/
│
├── models
│   ├── best_model_*.keras
│
├── reports
│   ├── Deep CNNs for Automated BTC using MRI Imaging.pdf
│
├── src/
|   ├── utils/
│   │   ├── augmentation.py
│   │   ├── metrics.py
│   │   ├── models.py
│   │   ├── preprocessing.py
│   │   ├── visualization.py
│   ├── evaluate.py
│   ├── predictions.py
│   ├── train.py
├── .gitignore
├── README.md
├── requirements.txt
```


## 🔍 Key Features

- Multi-model Architecture: MobileNetV2, VGG16, Basic CNN, and Enhanced CNN
- Data Augmentation: Rotations, shifts, zoom, shear, horizontal flipping
- Transfer Learning: Fine-tuning the last 4 convolutional blocks
- Comprehensive Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- GUI Application: Brain MRI classifier with graphical interface (Tkinter)

## 📚 Dataset

- Source: Brain Tumor MRI Dataset - Kaggle
- Total images: 7,023
- Classes:
   - Glioma
   - Meningioma
   - Pituitary
   - No Tumor

- Preprocessing:
   - Resized to 150x150 pixels
   - Normalization to [0, 1]
   - Augmentation during training

## 💻 Installation
```bash
# Clone repository
git clone https://github.com/yourusername/brain-tumor-classifier.git
cd brain-tumor-classifier

# Install dependencies
pip install -r requirements.txt
```

## 💡 Usage

### 1. Data Preparation & Augmentation
```bash
python src/utils/augmentation.py
```
### 2. Model Training

Select the model to train (basic_cnn, enhanced_cnn, vgg16, mobilenet) in train.py
```bash
python src/train.py
```
### 3. Model Evaluation

Evaluate the trained model on the test set
```bash
python src/evaluate.py
```
### 4. Predict on New Images

Launch the GUI to classify MRI scans:
```bash
python src/predictions.py
```

## 📊 Methodology

- Data Augmentation: Rotation 10°, width/height shift 20%, zoom 30%, shear 30%, horizontal flip

- Models:
   - Basic CNN: Simple convolution + Dense layers
   - Enhanced CNN: Additional convolutional layers
   - VGG16: Pre-trained on ImageNet + Fine-tuning
   - MobileNetV2: Lightweight model + Fine-tuning
- Optimizer: Adam (lr = 1e-5)
- Callbacks: Early Stopping and ReduceLROnPlateau for better convergence
- Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

## 📝 Results

| Model                  | Accuracy | Precision | Recall | F1-Score |
|-------------------------|----------|-----------|--------|----------|
| Basic CNN               | 56%      | 62%       | 56%    | 53%      |
| Enhanced CNN            | 60%      | 64%       | 60%    | 57%      |
| VGG16 (Transfer)        | **91%**  | **91%**   | **91%**| **90%**  |
| MobileNetV2 (Transfer)  | 79%      | 80%       | 79%    | 78%      |

## 👨‍💼 Author
- **Diego Vega Camacho**
- Email: A01704492@tec.mx
- **Tecnológico de Monterrey (ITESM)**
