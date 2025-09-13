<p align="center">
  <img src="https://github.com/77AXEL/PyCNN/blob/main/logo.png" alt="CNN Architecture"/>
</p>

This is a Convolutional Neural Network (CNN) framework project implemented entirely from scratch using only low-level libraries like NumPy, PIL, SciPy and Cython no deep learning frameworks (e.g., TensorFlow or PyTorch) are used. It can train a CNN model on your local dataset folder or an external Hugging Face dataset, save or load models, support CUDA and the Adam optimizer for better performance, and switch the training backend between CPU and GPU

<p align="center">
  <img src="https://github.com/77AXEL/PyCNN/blob/main/visualized.png" alt="CNN Architecture" width="600"/>
</p>

## 📦 Releases

| Version | Latest | Stable |
| ------- | ------ | ------ |
|  [2.2](https://github.com/77AXEL/PyCNN/releases/tag/v2.2)      |   ✅  | ✅ |
|  [2.0](https://github.com/77AXEL/PyCNN/releases/tag/v2.0)      |   ❌  | ✅ |
|  [0.1.2](https://github.com/77AXEL/PyCNN/releases/tag/v0.1.2)  |   ❌  | ✅ |
|  [0.1.1](https://github.com/77AXEL/PyCNN/releases/tag/v0.1.1)  |   ❌  | ✅ |
|  [0.1.0](https://github.com/77AXEL/PyCNN/releases/tag/v0.1.0)  |   ❌  | ✅ |

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
* 🚀 ** More CPU based performance optimizations** for faster computation and memory efficiency
* 🔄 **Automatic backend conversion** when loading models trained on a different backend
* 🛢️ **Hugging Face** CNN datasets support

---

## 🖼 Dataset Structure (local)

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
from pycnn.pycnn import PyCNN

pycnn= CNN()
pycnn.cuda(True)  # Enable CUDA
pycnn.init(
    image_size=64, # If unspecified the default is 64
    batch_size=32, # If unspecified the default is 32
    layers=[256, 128, 64, 32, 16, 8, 4], # Allows you to define any type of dense layer,  If unspecified the default is [128, 64]
    learning_rate=0.0001, # If unspecified the default is 0.0001
    epochs=1000, # If unspecified the default is 50
    filters = [
        [# Custom filter 1],
        [# Custom filter 2],
        [# Custom filter 3],
        [# Custom filter ...],
    ] # If unspecified, the framework will use the default filters.
)

pycnn.adam() # If specified, the framework will use the adam optimizer.

pycnn.dataset.local(
    path_to_you_dataset_folder, 
    max_image=1000 # If unspecified, the framework will use all images from each class.
) # Use this method if you want to load your local dataset folder.

pycnn.dataset.hf(
    huggingface_dataset_name, 
    max_image=1000, # If unspecified, the framework will use all images from each class.
    cached=True # Using the cached database helps you bypass downloading the dataset each time it is loaded (the default behavior when cached=True).
    split="train" # Specify which split of the dataset to use for training the model (the default is the train split).

) # Use this method if you want to load a HuggingFace dataset folder.

pycnn.train_model(
    visualize=True, # Displays a real-time graph of accuracy and loss per epoch when enabled. Set to False or leave unspecified to disable this feature.
    early_stop=2 # Stops training when overfitting begins and the number of epochs exceeds early_stop. Set to 0 or leave unspecified to disable this feature.
) 
```

#### Saving/Loading model

```python
pycnn.save_model(path=your_save_path) # if your your_save_path is unspecified the framework will save it in "./model.bin" bu default
pycnn.load_model(path=your_model_path)
```

#### Prediction

```python
result = pycnn.predict(your_image_path) # Returns a tuple of (class name, confidense value)
print(result) 
```
> The framework will automatically convert weights, biases, and datasets to the selected backend. Models trained on GPU can still be loaded on CPU and vice versa.

#### Usage exemple

```python
from pycnn.pycnn import PyCNN
from os import listdir

pycnn = PyCNN()
pycnn.cuda(True)

pycnn.init(
    layers=[512, 256],
    epochs=500,
)

pycnn.dataset.hf("cifar10", max_image=500, cached=True)
pycnn.adam()
pycnn.train_model(early_stop=15)
pycnn.save_model("pycnn_cifar10.bin")

testdir = "cifar10_test"
_max = 1000
for classname in listdir(testdir):
  x = 0
  correct = 0
  for filename in listdir(f"{testdir}/{classname}"):
    if x == _max:
      break
    if pycnn.predict(f"{testdir}/{classname}/{filename}")[0] == classname:
      correct += 1
    x += 1
  print(classname, correct)
```
> Output:

> <img src="https://github.com/77AXEL/PyCNN/blob/main/output.png">

* Total prediction accuracy: **48.1%**, which is a **strong result** for a model trained on only **500 images per class**.

> Hardware used while training:
 <img src="https://github.com/77AXEL/PyCNN/blob/main/hardware.png">

---

## 📊 Performance

| Metric   | Value (example)      |
| -------- | -------------------- |
| Accuracy | ~90% |
| Epochs   |  1000                |
| Dataset  | ~500 image for each class |

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
