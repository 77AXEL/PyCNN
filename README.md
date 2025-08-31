# 🧠 PYCNN

This framework project is a simple Convolutional Neural Network (CNN) implemented entirely from scratch using only low-level libraries like NumPy, PIL, and SciPy—no deep learning frameworks (e.g., TensorFlow or PyTorch) are used. It includes image preprocessing, convolution and pooling operations, ReLU and softmax activations, forward/backward propagation, cuda support allowing accelerated training and inference and a fully connected classifier.

<p align="center">
  <img src="https://github.com/77AXEL/PyCNN/blob/main/visualized.png" alt="CNN Architecture" width="600"/>
</p>

## 📦 Releases

| Version | Latest | Stable | Test a trained model |
| ------- | ------ | ------ | -------------------- |
|  [2.0](https://github.com/77AXEL/PyCNN/releases/tag/v2.0)  |   ✅  | ✅ | <a href="https://cnnfsmodel.pythonanywhere.com/pycnn-pretrained-model">Test</a>         |
|  [0.1.2](https://github.com/77AXEL/PyCNN/releases/tag/v0.1.2)  |   ❌  | ✅ |    ❌     |
|  [0.1.1](https://github.com/77AXEL/PyCNN/releases/tag/v0.1.1)  |   ❌  | ✅ |    ❌     |
|  [0.1.0](https://github.com/77AXEL/PyCNN/releases/tag/v0.1.0)  |   ❌  | ✅ |    ❌     |

---

### 🚀 Key Features

* ✅ Fully functional CNN implementation from scratch
* 🧠 Manual convolution, max pooling, and ReLU activations
* 🔁 Forward and backward propagation with mini-batch gradient descent
* 🏷 Multi-class classification via softmax and cross-entropy loss
* 💾 Model save/load using `pickle`
* 🖼 RGB image preprocessing with customizable filters
* 🔍 Predict function to classify new unseen images
* 📊 Real-time training visualization (accuracy & loss per epoch)
* ⚡ **Optional CUDA acceleration** for faster training and inference
* 🆕 **Adam optimizer support** for improved training performance
* 🛠 **Dynamic user-defined layers** for fully customizable architectures
* 🚀 **Performance optimizations** for faster computation and memory efficiency
* 🔄 **Automatic backend conversion** when loading models trained on a different backend

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
> To help you get started, we’ve included a [starter `data` folder](https://github.com/77AXEL/PyCNN/tree/main/data) with example class directories.

---

## 🧪 How It Works

1. **Image Preprocessing**:

   * Each image is resized to a fixed size and normalized.
   * Filters (e.g., sharpening, edge detection) are applied using 2D convolution.
   * ReLU activation and 2×2 max-pooling reduce spatial dimensions.
   * GPU acceleration via **CUDA** allows these operations to run in parallel on the graphics card, significantly speeding up large datasets.

2. **Feature Vector**:

   * Flattened pooled feature maps are fed into fully connected layers.

3. **Feedforward + Softmax**:

   * Dense layers compute activations followed by a softmax for classification.
   * All dense computations can be performed on the GPU for faster matrix multiplications.

4. **Backpropagation**:

   * Gradients are computed layer-by-layer.
   * Weights and biases are updated using the **Adam or SGD optimizer**, which adapts learning rates for each parameter for faster and more stable convergence compared to basic gradient descent.
   * CUDA can also accelerate gradient computations and weight updates.

---

### 🖥️ PyCNN Usage

#### Training model
```python
from pycnn.model import CNN

model = CNN()
model.cuda(True)  # Enable CUDA
model.init(
    image_size=64,
    batch_size=32,
    layers=[256, 128, 64, 32, 16, 8, 4], # Allows you to define any type of dense layer.
    learning_rate=0.0001,
    epochs=1000,
    dataset_path="data",
    max_image=1000, # If unspecified, the framework will use all images from each class.
    filters = [
        [# Custom filter 1],
        [# Custom filter 2],
        [# Custom filter 3],
        [# Custom filter ...],
    ] # If unspecified, the framework will use the default filters
)

model.adam() # If specified, the framework will use the adam optimizer
model.load_dataset()
model.train_model(visualize=True, early_stop=2) 
#visualize: Displays a real-time graph of accuracy and loss per epoch when enabled. Set to False or leave unspecified to disable this feature.
#early_stop: Stops training when overfitting begins and the number of epochs exceeds early_stop. Set to 0 or leave unspecified to disable this feature.
```

#### Saving/Loading model

```python
model.save(path=your_save_path) # if your your_save_path is unspecified the framework will save it in "./model.bin"
model.save(path=your_model_path)
```

#### Prediction

```python
result = model.predict(your_image_path)
print(result)
```
> The model will automatically convert weights, biases, and datasets to the selected backend. Models trained on GPU can still be loaded on CPU and vice versa.

#### Usage exemple

```python
from pycnn.model import CNN
from os import listdir

model = CNN()
model.init(
    image_size=64,
    batch_size=32,
    layers=[256, 128, 64, 32, 16, 8, 4],
    learning_rate=0.0001,
    epochs=1000,
    dataset_path="data",
    max_image=100,
)
model.adam()
model.load_dataset()
model.train_model(early_stop=2)

x = 0
for path in listdir("data/cat"):
    if model.predict(f"data/cat/{path}") == "cat":
        x += 1
    if x == 10:
        break

print(f"cat: {x}/10")

x = 0
for path in listdir("data/dog"):
    if model.predict(f"data/dog/{path}") == "dog":
        x += 1
    if x == 10:
        break

print(f"dog: {x}/10")
```
> Output:
<img src="https://github.com/77AXEL/PyCNN/blob/main/v2.0-output.png">

---

## 📊 Performance

| Metric   | Value (example)      |
| -------- | -------------------- |
| Accuracy | ~90% (binary class) |
| Epochs   |  100–500                |
| Dataset  | ~300 image for each class |

---

### 📌 Installation

```bash
pip install git+https://github.com/77AXEL/PyCNN.git
```

> Optional: Install CuPy for CUDA support:

```bash
pip install cupy-cuda118  # Match your CUDA version
```
* See the <a href="https://nvidia.github.io/cuda-python/latest/">CUDA Documentation</a> for more information on how to set it up

---

### 💬 Feedback & Contributions

We welcome issues, suggestions, and contributions!
Check the [Discussions tab](https://github.com/77AXEL/PyCNN/discussions) or see [CONTRIBUTING.md](https://chatgpt.com/c/CONTRIBUTING.md)

---

### 🛡 Security

Found a security issue? Please report privately to:
📧 [a.x.e.l777444000@gmail.com](mailto:a.x.e.l777444000@gmail.com)

---

### 📜 License

Released under the <a href="https://github.com/77AXEL/PyCNN/blob/b7becb4bef3b0156dad9397b1d95d6465e98fc3c/LICENSE">MIT License</a>

---

. Github page: [https://77axel.github.io/PyCNN](https://77axel.github.io/PyCNN)

<img src="https://img.shields.io/badge/Author-A.X.E.L-red?style=flat-square;">  <img src="https://img.shields.io/badge/Open Source-Yes-red?style=flat-square;">
