# Pneumonia Detection using Deep Learning with Data Augmentation

This repository contains the implementation of a pneumonia detection system using deep learning and data augmentation techniques. The model is based on AlexNet architecture and demonstrates the effectiveness of data augmentation in improving detection accuracy from chest X-ray images.

## Project Overview
This project investigates the impact of data augmentation on pneumonia detection performance. We achieved:
- 97% accuracy with data augmentation
- 93% accuracy without data augmentation
- Significant improvement in normal case precision (84% to 95%)

## Dataset
The chest X-ray dataset is not included in this repository due to size constraints. You can download it from:
[Chest X-Ray Images (Pneumonia)](https://data.mendeley.com/datasets/rscbjbr9sj/2)

Dataset statistics:
- Total images: 2,929
- Training set: 2,342 images
- Validation set: 293 images
- Test set: 294 images
- Class distribution: Normal (27.0%), Pneumonia (73.0%)

### Detailed Dataset Distribution
Training Set:
- NORMAL: 633 images
- PNEUMONIA: 1,709 images

Validation Set:
- NORMAL: 79 images
- PNEUMONIA: 214 images

Test Set:
- NORMAL: 80 images
- PNEUMONIA: 214 images

## Project Structure
```
├── model.py            # Model architecture implementation
├── train.py            # Training script without augmentation
├── train_with_aug.py   # Training script with augmentation
├── test.py           # Evaluation script
└── data/             # Dataset directory (not included)
```


### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

## Usage

### Training

1. Train without augmentation:
```bash
python train.py
```

2. Train with augmentation:
```bash
python train_with_aug.py
```

### Testing
```bash
python test.py
```



