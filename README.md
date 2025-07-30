# 🧠 CNN from Scratch

This project is a simple **Convolutional Neural Network (CNN)** implemented **entirely from scratch** using only low-level libraries like NumPy, PIL, and SciPy—**no deep learning frameworks** (e.g., TensorFlow or PyTorch) are used. It includes image preprocessing, convolution and pooling operations, ReLU and softmax activations, forward/backward propagation, and a fully connected classifier.

<p align="center">
  <img src="https://github.com/77AXEL/CNN-FS/blob/main/visualized.png" alt="CNN Architecture" width="600"/>
</p>

---

## 🚀 Features

* Manual image preprocessing (RGB separation, resizing, normalization)
* Handcrafted convolution and max-pooling operations
* Fully connected layers (L1, L2, and output)
* Softmax + Cross-Entropy Loss
* Mini-batch gradient descent with backpropagation
* Model saving/loading using `pickle`
* Class prediction on new images

---

## 📁 Project Structure

```bash
.
├── cnn.py           # Main training and prediction script
├── main.py           # You script
├── model.bin         # Saved model (generated after training)
├── data/             # Folder containing class subfolders with images (setup your custom database for more images) 
│   ├── cat/
│   │   ├── cat1.png
│   │   ├── ...
│   └── dog/
│       ├── dog1.png
│       ├── ...
│       ...
```

---

## 🖼 Dataset Structure

Make sure your dataset folder is structured like this:

```
data/
├── class1/
│   ├── image1.png
│   ├── image2.png
├── class2/
│   ├── image1.png
│   ├── image2.png
```

Each subfolder represents a class (e.g., `cat`, `dog`), and contains sample images.

---

## 🧪 How It Works

1. **Image Preprocessing**:

   * Each image is resized to a fixed size and normalized.
   * Filters (e.g., sharpening, edge detection) are applied using 2D convolution.
   * ReLU activation and 2×2 max-pooling reduce spatial dimensions.

2. **Feature Vector**:

   * Flattened pooled feature maps are fed into fully connected layers.

3. **Feedforward + Softmax**:

   * Dense layers compute activations followed by a softmax for classification.

4. **Backpropagation**:

   * Gradients are computed layer-by-layer.
   * Weights and biases are updated using basic gradient descent.

---

## 🛠 Setup

```bash
pip install git+https://github.com/77AXEL/CNN-FS.git
```

---

## ✅ Training

Update and run the training block:

```python
from cnnfs.model import CNN

model = CNN()
model.init(
    image_size=32,
    batch_size=32,
    h1=128,
    h2=64,
    learning_rate=0.01,
    epochs=10,
    dataset_path="data",
    max_image=4000  # per class
)
model.load_dataset()
model.train_model()
model.save_model()
```

---

## 🔍 Predicting New Images

```python
model.load_model("model.bin")
prediction = model.predict("test_images/mycat.png")
print("Predicted class:", prediction)
```

---

## 💡 Example Filters Used

```text
[ [0, -1,  0],   Sharpen
  [-1, 5, -1],
  [0, -1,  0] ]

[ [1,  0, -1],   Edge detection
  [1,  0, -1],
  [1,  0, -1] ]

[[-1, -1, -1],   Laplacian
 [-1,  8, -1],
 [-1, -1, -1] ]
```

---

## 📊 Performance

| Metric   | Value (example)      |
| -------- | -------------------- |
| Accuracy | \~90% (binary class) |
| Epochs   | 10–50                |
| Dataset  | Custom / \~8000 imgs |

* Note that a larger dataset and more training epochs typically lead to higher accuracy.
---

## 🧠 Concepts Demonstrated

* CNNs without frameworks
* Data vectorization
* Forward and backward propagation
* Optimization from scratch
* One-hot encoding for multi-class classification

---

## 📦 Dependencies

* [NumPy](https://numpy.org)
* [Pillow](https://pypi.org/project/pillow/)
* [SciPy](https://scipy.org)

---

## 📜 License

MIT License — feel free to use, modify, and share.

---

## 🤝 Contributing

PRs are welcome! You can help:

* Add evaluation functions
* Improve filter design
* Extend to grayscale or multi-channel separately
* Parallelize dataset loading

---
<img src="https://img.shields.io/badge/Author-A.X.E.L-red?style=flat-square;"></img>
<img src="https://img.shields.io/badge/Open Source-Yes-red?style=flat-square;"></img>
