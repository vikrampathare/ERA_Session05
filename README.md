# ERA_Session05

# MNIST Digit Classification with Efficient CNN

This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
The model is designed with **Batch Normalization, Dropout, and Global Average Pooling (GAP)** to achieve high accuracy with **just ~9.8K trainable parameters**.

---

## 🚀 Features

* Lightweight CNN architecture (~9.8K parameters)
* Data augmentation with random rotations
* Batch Normalization + Dropout for regularization
* Global Average Pooling (GAP) for final classification
* OneCycleLR scheduler for optimized training
* Achieves **>99.4% accuracy** on MNIST test set 🎯

---

## 📂 Project Structure

```
├── model.py        # CNN architecture (Net class)
├── train.py        # Training loop
├── test.py         # Testing function
├── utils.py        # Data loaders & transforms
├── main.py         # Entry point (training + evaluation)
└── README.md       # Documentation
```

---

## 🏗 Model Architecture

| Layer                         | Output Shape | Parameters |
| ----------------------------- | ------------ | ---------- |
| Conv2d (1→10) + BN + Dropout  | 28×28×10     | 120        |
| Conv2d (10→16) + BN + Dropout | 28×28×16     | 1,488      |
| MaxPool2d + 1×1 Conv (16→10)  | 14×14×10     | 170        |
| Conv2d (10→16) + BN + Dropout | 14×14×16     | 1,488      |
| Conv2d (16→16) + BN + Dropout | 14×14×16     | 2,352      |
| MaxPool2d + 1×1 Conv (16→10)  | 7×7×10       | 170        |
| Conv2d (10→16) + BN + Dropout | 7×7×16       | 1,488      |
| Conv2d (16→16) + BN + Dropout | 7×7×16       | 2,352      |
| Conv2d (16→10)                | 7×7×10       | 170        |
| Global Avg Pooling            | 1×1×10       | 0          |
| **Output (Softmax)**          | 10 classes   | -          |

**Total Parameters:** 9,798
**Trainable Parameters:** 9,798

---

## 📊 Training & Results

* Optimizer: **SGD (lr=0.015, momentum=0.9, weight decay=1e-4)**
* Scheduler: **OneCycleLR**
* Epochs: **19 (early stop at 15)**
* Batch Size: **128**

### 📈 Accuracy Progress

| Epoch | Train Accuracy | Test Accuracy |
| ----- | -------------- | ------------- |
| 1     | ~55%           | 86.18%        |
| 2     | ~93%           | 96.11%        |
| 4     | ~97%           | 98.31%        |
| 7     | ~98%           | 98.80%        |
| 10    | ~98.6%         | 99.02%        |
| 13    | ~98.8%         | 99.29%        |
| 15    | ~98.9%         | **99.43% 🎯** |

✅ **Target Accuracy (99.4%) reached at epoch 15!**

---

## 📷 Sample Output (Logs)

```
Epoch 15/19
Epoch=15 Loss=0.0161 Batch_id=468 Accuracy=98.92%
Test set: Average loss: 0.0209, Accuracy: 9943/10000 (99.43%)

Target accuracy of 99.4% reached at epoch 15!
Final best accuracy: 99.43%
Total parameters: 9,798
```

---

## ⚡ How to Run

### 1. Clone Repo

```bash
git clone https://github.com/vikrampathare/ERA_Session05.git
cd mnist-cnn
```

### 2. Install Requirements

```bash
pip install torch torchvision torchsummary tqdm
```

### 3. Train Model

```bash
python main.py
```

---

## 🏆 Final Notes

* Achieves **state-of-the-art performance (99.4%)** on MNIST with a **compact CNN**
* Efficient for deployment on **low-resource devices**
* Can be extended to other image classification datasets

---

✨ **Author:** Vikram Pathare
📌 Built with ❤️ using PyTorch

---



Below is the exact output

Device: cuda
Total parameters: 9,798
Trainable parameters: 9,798
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 28, 28]             100
       BatchNorm2d-2           [-1, 10, 28, 28]              20
           Dropout-3           [-1, 10, 28, 28]               0
            Conv2d-4           [-1, 16, 28, 28]           1,456
       BatchNorm2d-5           [-1, 16, 28, 28]              32
           Dropout-6           [-1, 16, 28, 28]               0
         MaxPool2d-7           [-1, 16, 14, 14]               0
            Conv2d-8           [-1, 10, 14, 14]             170
            Conv2d-9           [-1, 16, 14, 14]           1,456
      BatchNorm2d-10           [-1, 16, 14, 14]              32
          Dropout-11           [-1, 16, 14, 14]               0
           Conv2d-12           [-1, 16, 14, 14]           2,320
      BatchNorm2d-13           [-1, 16, 14, 14]              32
          Dropout-14           [-1, 16, 14, 14]               0
        MaxPool2d-15             [-1, 16, 7, 7]               0
           Conv2d-16             [-1, 10, 7, 7]             170
           Conv2d-17             [-1, 16, 7, 7]           1,456
      BatchNorm2d-18             [-1, 16, 7, 7]              32
          Dropout-19             [-1, 16, 7, 7]               0
           Conv2d-20             [-1, 16, 7, 7]           2,320
      BatchNorm2d-21             [-1, 16, 7, 7]              32
          Dropout-22             [-1, 16, 7, 7]               0
           Conv2d-23             [-1, 10, 7, 7]             170
AdaptiveAvgPool2d-24             [-1, 10, 1, 1]               0
================================================================
Total params: 9,798
Trainable params: 9,798
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.70
Params size (MB): 0.04
Estimated Total Size (MB): 0.74
----------------------------------------------------------------

Epoch 1/19
Epoch=1 Loss=0.6530 Batch_id=468 Accuracy=55.83%: 100%|██████████| 469/469 [00:22<00:00, 20.82it/s]

Test set: Average loss: 0.6015, Accuracy: 8618/10000 (86.18%)

New best accuracy: 86.18%

Epoch 2/19
Epoch=2 Loss=0.1279 Batch_id=468 Accuracy=93.83%: 100%|██████████| 469/469 [00:22<00:00, 20.99it/s]

Test set: Average loss: 0.1470, Accuracy: 9611/10000 (96.11%)

New best accuracy: 96.11%

Epoch 3/19
Epoch=3 Loss=0.0530 Batch_id=468 Accuracy=96.67%: 100%|██████████| 469/469 [00:22<00:00, 20.95it/s]

Test set: Average loss: 0.0920, Accuracy: 9728/10000 (97.28%)

New best accuracy: 97.28%

Epoch 4/19
Epoch=4 Loss=0.0924 Batch_id=468 Accuracy=97.44%: 100%|██████████| 469/469 [00:22<00:00, 21.05it/s]

Test set: Average loss: 0.0582, Accuracy: 9831/10000 (98.31%)

New best accuracy: 98.31%

Epoch 5/19
Epoch=5 Loss=0.0623 Batch_id=468 Accuracy=97.77%: 100%|██████████| 469/469 [00:21<00:00, 21.33it/s]

Test set: Average loss: 0.0641, Accuracy: 9822/10000 (98.22%)


Epoch 6/19
Epoch=6 Loss=0.0519 Batch_id=468 Accuracy=98.12%: 100%|██████████| 469/469 [00:22<00:00, 20.99it/s]

Test set: Average loss: 0.0514, Accuracy: 9846/10000 (98.46%)

New best accuracy: 98.46%

Epoch 7/19
Epoch=7 Loss=0.0482 Batch_id=468 Accuracy=98.27%: 100%|██████████| 469/469 [00:22<00:00, 21.08it/s]

Test set: Average loss: 0.0410, Accuracy: 9880/10000 (98.80%)

New best accuracy: 98.80%

Epoch 8/19
Epoch=8 Loss=0.0162 Batch_id=468 Accuracy=98.46%: 100%|██████████| 469/469 [00:22<00:00, 21.14it/s]

Test set: Average loss: 0.0345, Accuracy: 9893/10000 (98.93%)

New best accuracy: 98.93%

Epoch 9/19
Epoch=9 Loss=0.0820 Batch_id=468 Accuracy=98.56%: 100%|██████████| 469/469 [00:22<00:00, 20.97it/s]

Test set: Average loss: 0.0435, Accuracy: 9868/10000 (98.68%)


Epoch 10/19
Epoch=10 Loss=0.0840 Batch_id=468 Accuracy=98.62%: 100%|██████████| 469/469 [00:21<00:00, 21.46it/s]

Test set: Average loss: 0.0335, Accuracy: 9902/10000 (99.02%)

New best accuracy: 99.02%

Epoch 11/19
Epoch=11 Loss=0.0350 Batch_id=468 Accuracy=98.75%: 100%|██████████| 469/469 [00:22<00:00, 21.29it/s]

Test set: Average loss: 0.0257, Accuracy: 9920/10000 (99.20%)

New best accuracy: 99.20%

Epoch 12/19
Epoch=12 Loss=0.0209 Batch_id=468 Accuracy=98.85%: 100%|██████████| 469/469 [00:21<00:00, 21.74it/s]

Test set: Average loss: 0.0306, Accuracy: 9914/10000 (99.14%)


Epoch 13/19
Epoch=13 Loss=0.0456 Batch_id=468 Accuracy=98.87%: 100%|██████████| 469/469 [00:21<00:00, 21.67it/s]

Test set: Average loss: 0.0231, Accuracy: 9929/10000 (99.29%)

New best accuracy: 99.29%

Epoch 14/19
Epoch=14 Loss=0.0074 Batch_id=468 Accuracy=98.89%: 100%|██████████| 469/469 [00:21<00:00, 21.41it/s]

Test set: Average loss: 0.0247, Accuracy: 9928/10000 (99.28%)


Epoch 15/19
Epoch=15 Loss=0.0161 Batch_id=468 Accuracy=98.92%: 100%|██████████| 469/469 [00:22<00:00, 21.30it/s]

Test set: Average loss: 0.0209, Accuracy: 9943/10000 (99.43%)

New best accuracy: 99.43%
Target accuracy of 99.4% reached at epoch 15!

Final best accuracy: 99.43%
Total parameters: 9,798

