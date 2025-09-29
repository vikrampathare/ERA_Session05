# ERA_Session05

# MNIST Digit Classification with Efficient CNN

A lightweight PyTorch implementation of a Convolutional Neural Network for MNIST digit classification, achieving **99.38% test accuracy** with less than 10K parameters.

## 🎯 Project Highlights

- **Parameters:** 9,798 (< 10K constraint met)
- **Best Test Accuracy:** 99.38% (Epoch 16)
- **Target Achieved:** 99.4% threshold nearly reached
- **Training Time:** ~21-22 seconds per epoch on CUDA
- **Architecture:** Efficient CNN with GAP (Global Average Pooling)

## 📊 Model Architecture

The network follows a structured design with three main blocks:

### Architecture Overview

```
Input (1×28×28)
    ↓
Block 1: Feature Extraction (28×28)
├─ Conv2d (1→10) + BatchNorm + Dropout
└─ Conv2d (10→16) + BatchNorm + Dropout
    ↓
Transition 1: MaxPool + 1×1 Conv (16→10)
    ↓
Block 2: Deep Features (14×14)
├─ Conv2d (10→16) + BatchNorm + Dropout
└─ Conv2d (16→16) + BatchNorm + Dropout
    ↓
Transition 2: MaxPool + 1×1 Conv (16→10)
    ↓
Block 3: Final Features (7×7)
├─ Conv2d (10→16) + BatchNorm + Dropout
└─ Conv2d (16→16) + BatchNorm + Dropout
    ↓
Output: Conv2d (16→10) + GAP
    ↓
Log Softmax (10 classes)
```

### Layer-wise Parameters

| Layer | Output Shape | Parameters |
|-------|-------------|-----------|
| Conv2d-1 | 10×28×28 | 100 |
| Conv2d-4 | 16×28×28 | 1,456 |
| Conv2d-8 | 10×14×14 | 170 |
| Conv2d-9 | 16×14×14 | 1,456 |
| Conv2d-12 | 16×14×14 | 2,320 |
| Conv2d-16 | 10×7×7 | 170 |
| Conv2d-17 | 16×7×7 | 1,456 |
| Conv2d-20 | 16×7×7 | 2,320 |
| Conv2d-23 | 10×7×7 | 170 |
| **Total** | | **9,798** |

## 🚀 Training Configuration

### Hyperparameters

- **Optimizer:** SGD with momentum (0.9)
- **Learning Rate:** 0.015 (max)
- **Weight Decay:** 0.0001
- **Batch Size:** 128
- **Epochs:** 19
- **Dropout:** 0.05
- **Scheduler:** OneCycleLR with warmup

### Data Augmentation

**Training:**
- Random Rotation: ±7 degrees
- Normalization: mean=0.1307, std=0.3081

**Testing:**
- Standard Normalization only

## 📈 Training Results

### Accuracy Progression

| Epoch | Train Acc | Test Acc | Test Loss |
|-------|-----------|----------|-----------|
| 1 | 55.80% | 84.70% | 0.6716 |
| 2 | 93.90% | 97.00% | 0.1255 |
| 3 | 96.83% | 97.48% | 0.0918 |
| 4 | 97.62% | 98.32% | 0.0560 |
| 5 | 97.97% | 98.11% | 0.0608 |
| 6 | 98.20% | 98.86% | 0.0426 |
| 7 | 98.39% | **98.98%** | 0.0354 |
| 10 | 98.64% | **99.01%** | 0.0319 |
| 11 | 98.77% | **99.12%** | 0.0267 |
| 12 | 98.83% | **99.18%** | 0.0259 |
| 15 | 99.03% | **99.29%** | 0.0225 |
| **16** | **99.10%** | **99.38%** ⭐ | **0.0211** |
| 19 | 99.18% | 99.33% | 0.0197 |

### Performance Graph

```
Test Accuracy Over Epochs
100% ┤                                    ╭──╮
 99% ┤                          ╭────╮───╯  ╰─
 98% ┤                ╭────╮────╯    
 97% ┤        ╭───────╯              
 96% ┤    ╭───╯                      
 95% ┤    │                          
 90% ┤    │                          
 85% ┤╭───╯                          
     └┴───┴───┴───┴───┴───┴───┴───┴───┴───┴
      1   3   5   7   9  11  13  15  17  19
```

## 🔑 Key Features

### Efficient Design
- **1×1 Convolutions:** Reduce channels in transition blocks
- **Global Average Pooling:** Eliminates fully connected layers
- **Batch Normalization:** Stabilizes training and enables higher learning rates
- **Strategic Dropout:** Prevents overfitting (5%)

### Training Optimizations
- **OneCycleLR Scheduler:** Fast convergence with learning rate warmup
- **Data Augmentation:** Improves generalization with rotation
- **SGD with Momentum:** Efficient optimization

## 💻 Requirements

```bash
torch
torchvision
tqdm
torchsummary (optional)
```

## 🏃 Quick Start

```bash
# Install dependencies
pip install torch torchvision tqdm torchsummary

# Run training
python mnist_cnn.py
```

## 📝 Model Summary

```
Total params: 9,798
Trainable params: 9,798
Non-trainable params: 0
Input size (MB): 0.00
Forward/backward pass size (MB): 0.70
Params size (MB): 0.04
Estimated Total Size (MB): 0.74
```

## 🎓 Results Analysis

### Strengths
✅ Achieves 99.38% accuracy with minimal parameters  
✅ Fast training (~21s per epoch on GPU)  
✅ Efficient architecture with GAP  
✅ Stable convergence with OneCycleLR  
✅ Good generalization (low overfitting)

### Observations
- Model reaches 99%+ accuracy by epoch 10
- Best performance at epoch 16 (99.38%)
- Minimal gap between train and test accuracy
- Consistent performance in later epochs (99.28-99.38%)

## 📄 License

MIT License

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

**Note:** This model demonstrates that efficient CNN architectures with proper regularization and training strategies can achieve excellent results on MNIST with minimal computational resources.
