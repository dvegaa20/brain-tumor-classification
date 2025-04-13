# Brain Tumor Classification System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow 2.12](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://www.tensorflow.org/)
[![IEEE Format](https://img.shields.io/badge/Format-IEEE-blueviolet.svg)](https://ieeeauthorcenter.ieee.org/)

Deep learning system for automated brain tumor classification from MRI scans, implementing state-of-the-art CNN architectures with clinical validation metrics.

## Project Structure

```
📂 brain-tumor-classification
├── data/
│   ├── raw/
│   │   ├── training/
│   │   ├── testing/
│
├── models
│   ├── model.h5
│
├── reports
│   ├── model.h5
│
├── src/
|   ├── utils/
│   │   ├── preprocessing.py
│   │   ├── augmentation.py
│   │   ├── models.py
│   │   ├── metrics.py
│   │   ├── visualization.py
│   ├── train.py
│   ├── evaluate.py
│
├── requirements.txt
├── README.md
```


## Key Features
- **Multi-model Architecture**: Implements MobileNetV2, VGG16, and custom CNNs

## Installation
```bash
# Clone repository
git clone https://github.com/yourusername/brain-tumor-classifier.git
cd brain-tumor-classifier
```

# Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation & Augmentation
```sh
python augmentation.py
```

### 2. Model Trainning
Select the model you want to train, after that, execute the script
```sh
python train.py
```

### 3. Model Evaluation
Select the trained model outputed from the last step, after that, execute the script
```sh
python evaluate.py
```

## Version Control

* 1.0.0
    * First version with preprocessing and data augmentation
