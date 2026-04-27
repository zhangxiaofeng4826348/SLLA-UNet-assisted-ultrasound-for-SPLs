# SLLA-UNet: AI-Assisted Ultrasound Diagnosis of Subpleural Pulmonary Lesions

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An interactive and interpretable AI-assisted system for ultrasound diagnosis of subpleural pulmonary lesions (SPLs) based on the SLLA-UNet architecture.

## Overview

This repository implements a deep learning-based system for assisting ultrasound diagnosis of subpleural pulmonary lesions. The core model is built upon the SLLA-UNet architecture, supporting both semi-supervised learning and fine-tuning strategies. Grad-CAM is integrated to provide interpretable visual explanations for clinical decision support.

## Model Architecture
<img width="1041" height="753" alt="mode structurel" src="https://github.com/user-attachments/assets/ac99c2c7-eeff-44c4-8d00-2de2a7dd8219" />

## Repository Structure
├── config/ # Configuration files
├── data/ # Dataset and data loaders
├── losses/ # Loss functions
├── models/ # Model definitions
├── scripts/ # Training and testing scripts
├── utils/ # Utility functions
├── images/ # Model architecture figure
├── .gitignore
├── LICENSE
└── README.md

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Other dependencies can be installed via:

```bash
pip install -r requirements.txt

# Semi-supervised learning
python scripts/train_ssl.py --config config/config_ssl.py

# Fine-tuning
python scripts/train_fine.py

# SSL without Swin Transformer
python scripts/train_non_swim_ssl.py

# Fine-tuning without Swin Transformer
python scripts/train_non_swim_fine.py
python scripts/test_finetune.py
python utils/grad_cam.py --image_path data/Ext_test1_images/sample.jpg
```
