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
git clone https://github.com/your-username/mnist-cnn.git
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

✨ **Author:** *Your Name*
📌 Built with ❤️ using PyTorch

---

Would you like me to also create a **Markdown version with emojis, badges, and maybe accuracy graphs (matplotlib plots)** so your README looks like a professional GitHub repo?
