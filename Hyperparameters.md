# Research Hyperparameters and Augmentations

This document provides detailed information about the hyperparameters and data augmentations used in our research, as referenced in the paper "Trust Prediction in Assistive Robotics using Multi-Modal Video Transformers".

## Model Training Hyperparameters

The following table lists the key hyperparameters used for training our models:

| Component | Parameter | Value |
|-----------|-----------|-------|
| **General** | Batch Size | 12 |
| | Training Epochs | 35 |
| | Sequence Length | 200 |
| | Stride | 100 |
| | Downsample Factor | 2 |
| **Optimizer** | Type | AdamW |
| | Initial Learning Rate | 0.001 |
| | Weight Decay | 0.01 |
| | Betas | (0.9, 0.999) |
| | Eps | 1e-08 |
| **Learning Rate Scheduler** | Scheduler Type | CosineAnnealingLR |
| | T max | 35 |
| | Eta min | 0.00001 |
| | Warmup Steps | 500 |
| **Vision Model (ViT)** | Pretrained | TRUE |
| | Image Size Input | 224 |
| | Number of Channels | 3 |
| | Patch Size | 16 |
| | Hidden Size | 768 |
| | Hidden Layers | 12 |
| | Number Attention Heads | 12 |
| | Intermediate Size | 3072 |
| | Hidden Activation | Gelu |
| | Layer Norm Eps | 1.00E-12 |
| | Initializer Range | 0.02 |
| | Encoder Stride | 16 |
| | Hidden Dropout Prob | 0.0 |
| | Attention Dropout Prob | 0.0 |
| **Sequence Model** | Number Of Blocks | 4 |
| | Intermediate Size | 512 |
| | Hidden Size | 128 |
| | Number Attention Heads | 8 |
| | Number Key Value Heads | 8 |
| | Max Position Embeddings | 200 |
| | Hidden Activation | Silu |
| | Initializer Range | 0.02 |
| | RMS Norm Eps | 1.00E-06 |
| | Attention Dropout | 0.0 |
| **Task Head** | Model | MLP |
| | Number Of Layers | 1 |
| | Hidden Size | 256 |
| | RMS Norm Eps | 1.00E-06 |
| **Fusion Module** | Model | MLP |
| | Number Of Layers | 1 |
| | Hidden Size | 256 |
| | RMS Norm Eps | 1.00E-06 |
| **Projector** | Model | MLP |
| | Number Of Layers | 2 |
| | Hidden Size | 256 |
| | RMS Norm Eps | 1.00E-06 |
| | Hidden Activation | Relu |

For a more comprehensive overview of the trainable parameters in each model component, please refer to Table 1 in our paper.

## Data Augmentations

To improve model robustness and generalization, we applied the following data augmentations:

| Augmentation | Probability | Parameters |
|--------------|-------------|------------|
| CropAndPad | 0.2 | Percent: 0.05 |
| Blur | 0.2 | Limit: 4 |
| Perspective | 0.2 | Scale: [0.05, 0.1] |
| Brightness | 0.2 | Limit: [-0.2, 0.2] |
| Sharpen | 0.2 | Alpha: [0.2, 0.5], Lightness: [0.5, 1.0] |
| Affine | 0.2 | Translate: [-0.1, 0.1] |
| Emboss | 0.2 | Alpha: [0.2, 0.5], Strength: [0.2, 0.7] |
| HueSaturationValue | 0.2 | Hue Shift Limit: 20, Sat Shift Limit: 30, Val Shift Limit: 20 |

Note: Eye gaze coordinates were transformed where applicable to maintain consistency with the augmented frames.

## Hardware Specifications

All models in this research were trained on an NVIDIA RTX 4090 GPU.