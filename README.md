<p align="left">
  <img src="https://github.com/77AXEL/CNN-FS/blob/main/logo.png" alt="CNN Architecture" width="100"/>
</p>

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
├── class../
│   ├── ..
..
```

Each subfolder represents a class (e.g., `cat`, `dog`), and contains sample images.
> To help you get started, we’ve included a [starter `data` folder](https://github.com/77AXEL/CNN-FS/tree/main/data) with example class directories.
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
    image_size=64,
    batch_size=32,
    h1=128,
    h2=64,
    learning_rate=0.001,
    epochs=400,
    dataset_path="data", # Your dataset folder path
    max_image=200, # If not specified, the model will load all images for each class
    filters=[
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    ] # If not specified, the model will use its own default filters
)
model.load_dataset() # Processes all images for each class to prepare them for later use in training
model.train_model() # Starts model training based on the classes in your dataset
model.save_model() # Stores the trained model's weights and biases in a model.bin file
```

---

## 🔍 Predicting New Images

```python
model.load_model("model.bin") # Load the trained model
prediction = model.predict("test_images/mycat.png") # Applies the trained model to classify the input image
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
