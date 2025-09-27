import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
from numba import njit
import numpy as np
import pickle
import torch
import os

class PyCNNTorchModel(nn.Module):
    def __init__(self, layers, num_classes, filters, image_size):
        super(PyCNNTorchModel, self).__init__()
        self.filters = torch.tensor(filters, dtype=torch.float32)
        self.image_size = image_size
        
        conv_output_size = image_size - 2
        pool_output_size = conv_output_size // 2
        input_features = pool_output_size * pool_output_size * len(filters)
        
        self.layers_list = nn.ModuleList()
        prev_size = input_features
        
        for layer_size in layers:
            self.layers_list.append(nn.Linear(prev_size, layer_size))
            self.layers_list.append(nn.ReLU())
            prev_size = layer_size
        
        self.output_layer = nn.Linear(prev_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def preprocess_image(self, x):
        """Apply convolution filters and max pooling like PyCNN"""
        batch_size = x.shape[0]
        
        if x.shape[1] == 3:
            r, g, b = x[:, 0], x[:, 1], x[:, 2]
        else:
            r = g = b = x.squeeze(1)
        
        feature_maps = []
        
        for i, filt in enumerate(self.filters):
            filt_tensor = filt.unsqueeze(0).unsqueeze(0)
            
            conv_r = torch.nn.functional.conv2d(r.unsqueeze(1), filt_tensor, padding=0)
            conv_g = torch.nn.functional.conv2d(g.unsqueeze(1), filt_tensor, padding=0)
            conv_b = torch.nn.functional.conv2d(b.unsqueeze(1), filt_tensor, padding=0)
            
            conv = conv_r + conv_g + conv_b
            conv = torch.relu(conv)
            
            pooled = torch.nn.functional.max_pool2d(conv, kernel_size=2, stride=2)
            feature_maps.append(pooled.squeeze(1))
        
        flattened = torch.cat([fm.view(batch_size, -1) for fm in feature_maps], dim=1)
        return flattened
    
    def forward(self, x):
        x = self.preprocess_image(x)
        for layer in self.layers_list:
            x = layer(x)
        
        x = self.output_layer(x)
        return self.softmax(x)

@njit()
def _cpu_softmax_batch(x):
    n_samples, n_classes = x.shape
    out = np.empty_like(x)
    
    for i in range(n_samples):
        row = x[i, :]
        row_max = np.max(row)
        exps = np.exp(row - row_max)
        out[i, :] = exps / np.sum(exps)
    
    return out

@njit
def _cpu_cross_entropy_batch(preds, labels):
    return -(labels * np.log(preds + 1e-9)).sum(axis=1)

class PyCNN:
    def __init__(self):
        self.optimizer = "sgd"
        self.filters = [
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],                                 # Sharpening
            [[1, 0, -1], [1, 0, -1], [1, 0, -1]],                                  # Vertical edges
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],                             # Laplacian
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],                                  # Sobel X
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],                                  # Sobel Y
            [[1, 1, 1], [1, -8, 1], [1, 1, 1]],                                    # High-pass
            [[0.111, 0.111, 0.111], [0.111, 0.111, 0.111], [0.111, 0.111, 0.111]], # Blur
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],                                    # Laplacian (4-neighbors)
            [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],                               # Strong vertical lines
            [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],                               # Strong horizontal lines
            [[2, -1, 0], [-1, 2, -1], [0, -1, 2]],                                 # Diagonal enhancement (↘)
            [[0, -1, 2], [-1, 2, -1], [2, -1, 0]],                                 # Diagonal enhancement (↙)
            [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]],                               # Diagonal edges
            [[1, -2, 1], [-2, 5, -2], [1, -2, 1]],                                 # Strong sharpening
            [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],                                  # Prewitt X (alternative to Sobel)
            [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],                                  # Prewitt Y
            [[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]],                         # Edge + contrast boost
            [[1, 1, 0], [1, -4, 0], [0, 0, 0]],                                    # Corner detection
            [[0, -1, -2], [1, 1, -1], [2, 1, 0]],                                  # Emboss (↙ lighting)
        ]

        
        self.use_cuda = False
        self._setup_backend(False)
        
        self.dataset = self.DatasetLoader(self)

    def _setup_backend(self, use_cuda):
        """Setup the computational backend (CPU or GPU)"""
        self.use_cuda = use_cuda
        
        if self.use_cuda:
            try:
                import cupy as cp
                from cupyx.scipy.signal import convolve2d
                self.cp = cp
                self.zeros = cp.zeros
                self.maximum = cp.maximum
                self.dot = cp.dot
                self.random = cp.random
                self.array = cp.array
                self.exp = cp.exp
                self.log = cp.log
                self.convolve2d = convolve2d
                self.argmax = cp.argmax
                self.sum = cp.sum
                self.max = cp.max
                self.permutation = cp.random.permutation
                self.sqrt = cp.sqrt
                print("CUDA backend initialized successfully")
            except ImportError:
                print("Warning: CuPy not installed. Falling back to CPU.")
                self._setup_cpu_backend()
            except Exception as e:
                print(f"Warning: CUDA not available ({str(e)}). Falling back to CPU.")
                self._setup_cpu_backend()
        else:
            self._setup_cpu_backend()
    
    def _setup_cpu_backend(self):
        """Setup CPU backend"""
        import numpy as np
        from scipy.signal import convolve2d
        from _pycnn.cython.pooling import pooling
        self.cpu_max_pooling = pooling.max_pooling
        from _pycnn.cython.forward_pass import forward_pass
        self.cpu_forward_pass = forward_pass.forward_pass
        from _pycnn.cython.backward_pass import backward_pass
        self.cpu_backward_pass = backward_pass.backward_pass
        from _pycnn.cython.update_parametres import update_parametres
        self.cpu_update_parameters = update_parametres.update_parameters
            
        self.use_cuda = False
        self.zeros = np.zeros
        self.maximum = np.maximum
        self.dot = np.dot
        self.random = np.random
        self.array = np.array
        self.exp = np.exp
        self.log = np.log
        self.convolve2d = convolve2d
        self.argmax = np.argmax
        self.sum = np.sum
        self.max = np.max
        self.permutation = np.random.permutation
        self.sqrt = np.sqrt

    def _to_cpu(self, x):
        """Convert array to CPU (numpy) format"""
        if self.use_cuda and hasattr(x, 'get'):
            return x.get()
        elif hasattr(x, '__array__'):
            import numpy as np
            return np.array(x)
        else:
            return x

    def _to_backend(self, x):
        """Convert array to current backend format"""
        if self.use_cuda:
            return self.array(x)
        else:
            return self.array(x)

    def cuda(self, use_cuda=False):
        """Enable or disable CUDA"""
        if use_cuda != self.use_cuda:
            self._setup_backend(use_cuda)
            
            if hasattr(self, 'weights'):
                for key in self.weights:
                    self.weights[key] = self._to_backend(self._to_cpu(self.weights[key]))
                for key in self.biases:
                    self.biases[key] = self._to_backend(self._to_cpu(self.biases[key]))
                
            if hasattr(self, 'data') and len(self.data) > 0:
                self.data = self._to_backend(self._to_cpu(self.data))
                self.labels = self._to_backend(self._to_cpu(self.labels))

    def adam(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.optimizer = "adam"
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}
        
        if hasattr(self, 'weights') and hasattr(self, 'biases'):
            for key in self.weights:
                self.m[key] = self.zeros(self.weights[key].shape, dtype=float)
                self.v[key] = self.zeros(self.weights[key].shape, dtype=float)
            for key in self.biases:
                self.m[key] = self.zeros(self.biases[key].shape, dtype=float)
                self.v[key] = self.zeros(self.biases[key].shape, dtype=float)
            
        print("Adam optimizer enabled.")

    def init(self, batch_size=32, layers=[128, 64], learning_rate=0.001, epochs=50, filters=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = layers
        self.image_size = None

        if filters is not None:
            self.filters = filters

        self.random.seed(42)
        
        print(f"Network initialized with {len(layers)} hidden layers: {layers}")

    def _softmax_batch(self, x):
        if self.use_cuda:
            x_max = self.max(x, axis=1, keepdims=True) if hasattr(x, 'ndim') and x.ndim > 1 else self.max(x)
            if hasattr(x, 'ndim') and x.ndim > 1:
                x_max = self.max(x, axis=1, keepdims=True)
            else:
                x_max = self.max(x)
            x = x - x_max
            exps = self.exp(x)
            if hasattr(exps, 'ndim') and exps.ndim > 1:
                return exps / self.sum(exps, axis=1, keepdims=True)
            else:
                return exps / self.sum(exps)
        else:
            return _cpu_softmax_batch(x)

    def _cross_entropy_batch(self, preds, labels):
        if self.use_cuda:
            return -(labels * self.log(preds + 1e-9)).sum(axis=1)
        else:
            return _cpu_cross_entropy_batch(preds, labels)
    
    def _max_pooling(self, feature_map):
        if self.use_cuda:
            stride = size = 2
            h, w = feature_map.shape

            fm = self.array(feature_map)

            new_h = h // size
            new_w = w // size
            fm_cropped = fm[:new_h*size, :new_w*size]

            pooled = fm_cropped.reshape(new_h, size, new_w, size).max(axis=(1, 3))
            return pooled
        else:
            return self.cpu_max_pooling(feature_map)

    def _preprocess_image(self, path, image_size, filters):
        if isinstance(path, str):
            img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
        else:
            img = path

        img_arr = self.array(np.asarray(img), dtype=float) / 255.0
        
        r, g, b = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]
        
        feature_maps = []
        for filt in filters:
            filt = self._to_backend(filt)
            conv = self.convolve2d(r, filt, mode='valid') + \
                self.convolve2d(g, filt, mode='valid') + \
                self.convolve2d(b, filt, mode='valid')
            conv = self.maximum(conv, 0)  # ReLU
            pooled = self._max_pooling(conv)
            feature_maps.append(pooled)
        
        return self.array(feature_maps).reshape(-1)

    class DatasetLoader:
        def __init__(self, parent):
            self.parent = parent
        
        def _init_layers(self, classes, input_size):
            self.parent.classes = classes
            self.parent.weights = {}
            self.parent.biases = {}
            
            previous_layer = input_size
            for i, layer_size in enumerate(self.parent.layers):
                weight_key = f"w{i+1}"
                bias_key = f"b{i+1}"
                
                self.parent.weights[weight_key] = self.parent.random.randn(layer_size, previous_layer) * (2 / previous_layer) ** 0.5
                self.parent.biases[bias_key] = self.parent.random.randn(layer_size) * 0.01
                previous_layer = layer_size
            
            self.parent.data = []
            self.parent.labels = []
            
            self.parent.weights["wo"] = self.parent.random.randn(len(classes), previous_layer) * (2 / previous_layer) ** 0.5
            self.parent.biases["bo"] = self.parent.random.randn(len(classes)) * 0.01
            
            if hasattr(self.parent, 'optimizer') and self.parent.optimizer == "adam":
                self.parent.m = {}
                self.parent.v = {}
                for key in self.parent.weights:
                    self.parent.m[key] = self.parent.zeros(self.parent.weights[key].shape, dtype=float)
                    self.parent.v[key] = self.parent.zeros(self.parent.weights[key].shape, dtype=float)
                for key in self.parent.biases:
                    self.parent.m[key] = self.parent.zeros(self.parent.biases[key].shape, dtype=float)
                    self.parent.v[key] = self.parent.zeros(self.parent.biases[key].shape, dtype=float)

        def local(self, path, max_image=None, image_size=64, aug=[]):
            if len(aug) != 0:
                print("Warning: Augmentation enabled")
            self.parent.image_size = image_size
            self.dataset_path = path
            try:
                classes = os.listdir(path)
                if not classes:
                    raise ValueError(f"No class directories found in {path}")
                    
                self._init_layers(classes, ((image_size-2)//2)**2*len(self.parent.filters))
                
                for idx, cls in enumerate(classes):
                    folder = f"{path}/{cls}"
                    if not os.path.isdir(folder):
                        print(f"Warning: {folder} is not a directory, skipping...")
                        continue
                        
                    img_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
                    if not img_files:
                        print(f"Warning: No image files found in {folder}")
                        continue
                        
                    count = 0
                    for img_file in img_files:
                        if max_image is not None and count >= max_image:
                            break
                        img_path = f"{folder}/{img_file}"
                        print(f"\r\033[KLoading local dataset: {cls}/{img_file} ({count+1})", end="", flush=True)
                        try:
                            normal = Image.open(img_path).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
                            images = [
                                normal
                            ]

                            for aug_num in aug:
                                if aug_num == 1:
                                    aug_img = normal.transpose(Image.FLIP_LEFT_RIGHT)
                                elif aug_num == 2:
                                    aug_img = normal.transpose(Image.FLIP_TOP_BOTTOM)
                                elif aug_num == 3:
                                    aug_img = normal.rotate(90, expand=True).resize((image_size, image_size), Image.Resampling.LANCZOS)
                                elif aug_num == 4:
                                    aug_img = normal.rotate(-90, expand=True).resize((image_size, image_size), Image.Resampling.LANCZOS)
                                images.append(aug_img)
                                
                            
                            for image in images:
                                vec = self.parent._preprocess_image(image, image_size, self.parent.filters)
                                self.parent.data.append(vec)
                                one_hot = [0] * len(classes)
                                one_hot[idx] = 1
                                self.parent.labels.append(self.parent.array(one_hot))
                            count += 1
                        except Exception as e:
                            print(f"\nWarning: Failed to process {img_path}: {e}")
                            continue
                
                print(f"\nLoaded {len(self.parent.data)} images from {len(classes)} classes")
                if len(self.parent.data) == 0:
                    raise ValueError("No images were successfully loaded")
                    
            except Exception as e:
                raise RuntimeError(f"Failed to load local dataset from {path}: {e}")

        def hf(self, name, max_image=None, split="train", cached=True, aug=[]):
            if len(aug) != 0:
                print("Warning: Augmentation enabled")
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError("datasets library is required for Hugging Face datasets. Install with: pip install datasets")
                
            try:
                if cached:
                    dataset = load_dataset(name, split=split)
                else:
                    dataset = load_dataset(name, split=split, download_mode="force_redownload")
                    
            except Exception as e:
                raise RuntimeError(f"Failed to load dataset '{name}': {e}")

            # Find label column
            label_column = None
            for col in ["label", "labels", "target", "class"]:
                if col in dataset.column_names:
                    label_column = col
                    break
            
            if label_column is None:
                raise ValueError(f"Dataset '{name}' does not have a recognized label column. Available columns: {dataset.column_names}")
            
            image_column = None
            for col in ["image", "img", "picture", "photo"]:
                if col in dataset.column_names:
                    image_column = col
                    break
                    
            if image_column is None:
                raise ValueError(f"Dataset '{name}' does not have a recognized image column. Available columns: {dataset.column_names}")
                
            if hasattr(dataset.features[label_column], 'names'):
                label_names = dataset.features[label_column].names
            elif hasattr(dataset.features[label_column], '_int2str'):
                label_names = [dataset.features[label_column]._int2str[i] for i in range(len(dataset.features[label_column]._int2str))]
            else:
                unique_labels = set()
                for sample in dataset:
                    unique_labels.add(sample[label_column])
                label_names = sorted(list(unique_labels))
                print(f"Warning: Could not extract class names from dataset features, using: {label_names}")
            
            for idx, sample in enumerate(dataset):
                img = sample[image_column]
                if not isinstance(img, Image.Image):
                    if hasattr(img, 'convert'):
                        img = img
                    else:
                        img = Image.fromarray(np.array(img))
                image_size = img.size[0]
                break

            self._init_layers(label_names, ((image_size-2)//2)**2*len(self.parent.filters))
            self.parent.image_size = image_size
            
            class_counts = {i: 0 for i in range(len(label_names))}
            total_count = 0

            for idx, sample in enumerate(dataset):
                try:
                    img = sample[image_column]
                    label_idx = sample[label_column]
                    
                    if isinstance(label_idx, (list, tuple)):
                        label_idx = label_idx[0]
                    
                    if max_image is not None and class_counts[label_idx] >= max_image:
                        continue
                        
                    label_name = label_names[label_idx] if label_idx < len(label_names) else str(label_idx)
                    
                    print(f"\r\033[KLoading HuggingFace dataset: {name} - {label_name} ({class_counts[label_idx]+1})", end="", flush=True)

                    if not isinstance(img, Image.Image):
                        if hasattr(img, 'convert'):
                            img = img
                        else:
                            img = Image.fromarray(np.array(img))
                    
                    img = img.convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
                    
                    images = [img]
                            
                    for aug_num in aug:
                        if aug_num == 1:
                            aug_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        elif aug_num == 2:
                            aug_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                        elif aug_num == 3:
                            aug_img = img.rotate(90, expand=True).resize((image_size, image_size), Image.Resampling.LANCZOS)
                        elif aug_num == 4:
                            aug_img = img.rotate(-90, expand=True).resize((image_size, image_size), Image.Resampling.LANCZOS)
                        else:
                            continue
                        images.append(aug_img)
                    
                    for image in images:
                        vec = self.parent._preprocess_image(image, image_size, self.parent.filters)
                        self.parent.data.append(vec)
                        one_hot = [0] * len(label_names)
                        one_hot[label_idx] = 1
                        self.parent.labels.append(self.parent.array(one_hot))
                        total_count += 1
                    
                    class_counts[label_idx] += 1
                    
                    if max_image is not None and all(count >= max_image for count in class_counts.values()):
                        print(f"\nAll classes reached maximum of {max_image} images each.")
                        break
                        
                except Exception as e:
                    print(f"\nWarning: Failed to process sample {idx}: {e}")
                    continue
            
            print(f"\nLoaded {total_count} total images:")
            for i, (class_name, count) in enumerate(zip(label_names, class_counts.values())):
                print(f"  {class_name}: {count} images")
                
            if len(self.parent.data) == 0:
                raise ValueError("No images were successfully loaded from the dataset")
        
    def _forward_pass(self, batch_x, activations, z_values):
        if self.use_cuda:
            activations[0] = batch_x
            current_input = batch_x
            
            for i in range(len(self.layers)):
                weight_key = f"w{i+1}"
                bias_key = f"b{i+1}"
                
                z = current_input @ self.weights[weight_key].T
                z += self.biases[bias_key]
                z_values[i] = z
                current_input = self.maximum(z, 0)
                activations[i + 1] = current_input

            z_out = current_input @ self.weights["wo"].T
            z_out += self.biases["bo"]
            z_values[len(self.layers)] = z_out
            output = self._softmax_batch(z_out)
            
            return output
        else:
            return self.cpu_forward_pass(batch_x, activations, z_values, self.weights, self.biases, self.layers)

    def _backward_pass(self, output, batch_y, activations, z_values, gradients_w, gradients_b):
        if self.use_cuda:
            dL_dz_out = output - batch_y
            
            gradients_w["wo"][:] = dL_dz_out.T @ activations[-1]
            gradients_b["bo"][:] = self.sum(dL_dz_out, axis=0)
            
            dL_da = dL_dz_out @ self.weights["wo"]
            
            for i in reversed(range(len(self.layers))):
                weight_key = f"w{i+1}"
                bias_key = f"b{i+1}"
                
                dL_dz = dL_da * (z_values[i] > 0)
                
                gradients_w[weight_key][:] = dL_dz.T @ activations[i]
                gradients_b[bias_key][:] = self.sum(dL_dz, axis=0)
                
                if i > 0:
                    dL_da = dL_dz @ self.weights[weight_key]
        else:
            self.cpu_backward_pass(output, batch_y, activations, z_values,
                       gradients_w,
                       gradients_b,
                       self.weights,
                       self.layers)

    def _update_parameters(self, gradients_w, gradients_b):
        if self.use_cuda:
            if self.optimizer == "sgd":
                for key in gradients_w:
                    self.weights[key] -= self.learning_rate * gradients_w[key]
                for key in gradients_b:
                    self.biases[key] -= self.learning_rate * gradients_b[key]

            elif self.optimizer == "adam":
                self.t += 1
                
                for key in gradients_w:
                    grad = gradients_w[key]
                    self.m[key] *= self.beta1
                    self.m[key] += (1 - self.beta1) * grad
                    
                    self.v[key] *= self.beta2
                    self.v[key] += (1 - self.beta2) * (grad * grad)
                    
                    m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                    
                    update = self.learning_rate * m_hat / (self.sqrt(v_hat) + self.eps)
                    self.weights[key] -= update
                
                for key in gradients_b:
                    grad = gradients_b[key]
                    self.m[key] *= self.beta1
                    self.m[key] += (1 - self.beta1) * grad
                    
                    self.v[key] *= self.beta2
                    self.v[key] += (1 - self.beta2) * (grad * grad)
                    
                    m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                    
                    update = self.learning_rate * m_hat / (self.sqrt(v_hat) + self.eps)
                    self.biases[key] -= update
        else:
            self.cpu_update_parameters(self.weights, self.biases,
                           gradients_w, gradients_b,
                           self.learning_rate,
                           self.optimizer,
                           self.m, self.v,
                           self.beta1, self.beta2,
                           self.eps, self.t)

    def train_model(self, visualize=False, early_stop=0):
        if not hasattr(self, 'data') or len(self.data) == 0:
            raise ValueError("No data loaded. Please load a dataset first using dataset.local() or dataset.hf()")
            
        print()
        self.visualize = visualize
        self.data = self.array(self.data)
        self.labels = self.array(self.labels)
        num_samples = len(self.data)
        best_loss = float('inf')
        stop_counter = 0
        patience = early_stop

        gradients_w = {}
        gradients_b = {}
        for i in range(len(self.layers)):
            weight_key = f"w{i+1}"
            bias_key = f"b{i+1}"
            gradients_w[weight_key] = self.zeros(self.weights[weight_key].shape)
            gradients_b[bias_key] = self.zeros(self.biases[bias_key].shape)
        gradients_w["wo"] = self.zeros(self.weights["wo"].shape)
        gradients_b["bo"] = self.zeros(self.biases["bo"].shape)

        max_batch_size = min(self.batch_size, num_samples)
        activations = [None] * (len(self.layers) + 1)
        z_values = [None] * (len(self.layers) + 1)

        if self.visualize:
            plt.ion()
            fig, ax = plt.subplots()
            fig.patch.set_facecolor("#111")
            ax.set_facecolor("#111")
            loss_line, = ax.plot([], [], label="loss", linewidth=2)
            acc_line, = ax.plot([], [], label="accuracy", linewidth=2)
            ax.set_xlabel("Epoch", color="#f0f0f0")
            ax.set_ylabel("Value", color="#f0f0f0")
            ax.set_xlim(1, max(1, self.epochs))
            ax.set_ylim(0, 1)
            ax.set_title("Training: loss & accuracy per epoch", color="#f0f0f0")
            ax.tick_params(colors="#f0f0f0")
            for spine in ax.spines.values():
                spine.set_color("#f0f0f0")
            ax.grid(True, color="#444", linestyle="--", linewidth=0.5)
            ax.legend(facecolor="#111", edgecolor="#f0f0f0", labelcolor="#f0f0f0")
            epoch_nums = []
            losses = []
            accs = []

        for epoch in range(self.epochs):
            perm = self.permutation(num_samples)
            correct = 0
            total_loss = 0.0
            
            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                batch_idx = perm[start:end]
                batch_x = self.data[batch_idx]
                batch_y = self.labels[batch_idx]
                current_batch_size = end - start

                output = self._forward_pass(batch_x, activations, z_values)

                batch_loss = self._cross_entropy_batch(output, batch_y)
                total_loss += batch_loss.sum()

                pred = self.argmax(output, axis=1)
                true = self.argmax(batch_y, axis=1)
                correct += self.sum(pred == true)

                self._backward_pass(output, batch_y, activations, z_values, gradients_w, gradients_b)

                self._update_parameters(gradients_w, gradients_b)

            epoch_loss = total_loss / max(1, num_samples)
            epoch_acc = correct / max(1, num_samples)

            epoch_loss_cpu = self._to_cpu(epoch_loss)
            epoch_acc_cpu = self._to_cpu(epoch_acc)
            correct_cpu = int(self._to_cpu(correct))

            if early_stop:
                if epoch_loss_cpu < best_loss - 1e-8:
                    best_loss = epoch_loss_cpu
                    stop_counter = 0
                else:
                    stop_counter += 1
                    if stop_counter >= patience:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        break

            print(f"\rTraining model: epoch={epoch + 1}/{self.epochs}  correct={correct_cpu}/{num_samples}  acc={epoch_acc_cpu:.4f}  loss={epoch_loss_cpu:.6f}   ", end="")

            if self.visualize:
                epoch_nums.append(epoch + 1)
                losses.append(float(epoch_loss_cpu))
                accs.append(float(epoch_acc_cpu))

                loss_line.set_data(epoch_nums, losses)
                acc_line.set_data(epoch_nums, accs)

                ax.relim()
                ax.autoscale_view()
                ax.set_xlim(1, max(self.epochs, len(epoch_nums)))
                
                if epoch % max(1, self.epochs // 100) == 0 or epoch == self.epochs - 1:
                    fig.canvas.draw()
                    fig.canvas.flush_events()

            if epoch_loss_cpu <= 1e-6 and epoch_acc_cpu >= 0.999999 and epoch < self.epochs:
                print(f"\nModel reached maximum learning on epoch {epoch}")
                break

        print()
        if self.visualize:
            plt.ioff()
            plt.show()

    def save_model(self, path="model.bin"):
        """Save model with automatic CPU conversion for cross-platform compatibility"""
        if not hasattr(self, 'weights') or not hasattr(self, 'biases'):
            raise ValueError("No trained model to save. Train the model first.")
            
        model_data = {
            "weights": {key: self._to_cpu(val) for key, val in self.weights.items()},
            "biases": {key: self._to_cpu(val) for key, val in self.biases.items()},
            "filters": self._to_cpu(self.filters),
            "image_size": int(self._to_cpu(self.image_size)),
            "classes": list(self.classes),
            "layers": self.layers,
            "trained_with_cuda": self.use_cuda
        }
        
        with open(path, "wb") as model_file:
            pickle.dump(model_data, model_file)

        print(f"Model saved in {path} with {round(os.path.getsize(path) / (1024 * 1024), 2)}MB")
    
    def load_model(self, model_path):
        """Load model with automatic backend conversion"""
        with open(model_path, "rb") as model_file:
            model_data = pickle.load(model_file)
        
        self.weights = {}
        self.biases = {}
        for key, val in model_data["weights"].items():
            self.weights[key] = self._to_backend(val)
        for key, val in model_data["biases"].items():
            self.biases[key] = self._to_backend(val)
            
        self.filters = model_data["filters"]
        self.image_size = model_data["image_size"]
        self.classes = model_data["classes"]
        self.layers = model_data.get("layers", [])
        
        trained_with_cuda = model_data.get("trained_with_cuda", False)
        current_backend = "CUDA" if self.use_cuda else "CPU"
        training_backend = "CUDA" if trained_with_cuda else "CPU"
        
        print(f"Model loaded successfully!")
        print(f"Size: {round(os.path.getsize(model_path) / (1024 * 1024), 2)}MB")
        print(f"Training backend: {training_backend}")
        print(f"Current backend: {current_backend}")
        print(f"Network architecture: {len(self.layers)} hidden layers: {self.layers}")
        if trained_with_cuda != self.use_cuda:
            print("Backend conversion completed automatically.")

    def predict(self, img_path):
        """Make prediction on a single image"""
        if not hasattr(self, 'weights') or not hasattr(self, 'biases'):
            raise ValueError("No trained model available. Train or load a model first.")
            
        test_img = self._preprocess_image(img_path, self.image_size, self.filters)
        
        current_input = test_img
        for i in range(len(self.layers)):
            weight_key = f"w{i+1}"
            bias_key = f"b{i+1}"
            
            z = self.dot(self.weights[weight_key], current_input) + self.biases[bias_key]
            current_input = self.maximum(z, 0)
        
        logits = self.dot(self.weights["wo"], current_input) + self.biases["bo"]
        
        if hasattr(logits, 'ndim') and logits.ndim == 1:
            out = self._softmax_batch(logits.reshape(1, -1))[0]
        else:
            out = self._softmax_batch(logits)
            
        predicted_idx = self.argmax(out)
        predicted_idx = self._to_cpu(predicted_idx)
        
        try:
            predicted_idx = int(predicted_idx.item())
        except (AttributeError, ValueError):
            try:
                predicted_idx = int(predicted_idx[0])
            except (IndexError, TypeError):
                predicted_idx = int(predicted_idx)
        
        predicted_label = self.classes[int(predicted_idx)]
        confidence = float(self._to_cpu(out[predicted_idx]))
        
        return predicted_label, confidence

    def torch(self, save_path="model.pth"):
        """Export trained PyCNN model to PyTorch format"""
        if not hasattr(self, 'weights') or not hasattr(self, 'biases'):
            raise ValueError("No trained model to export. Train the model first.")
        
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for model export. Install with: pip install torch")
        
        pytorch_model = PyCNNTorchModel(
            layers=self.layers,
            num_classes=len(self.classes),
            filters=self.filters,
            image_size=self.image_size
        )
        
        state_dict = {}
        
        for i in range(len(self.layers)):
            weight_key = f"w{i+1}"
            bias_key = f"b{i+1}"
            
            pycnn_weight = self._to_cpu(self.weights[weight_key])
            pycnn_bias = self._to_cpu(self.biases[bias_key])
            
            state_dict[f'layers_list.{i*2}.weight'] = torch.tensor(pycnn_weight, dtype=torch.float32)
            state_dict[f'layers_list.{i*2}.bias'] = torch.tensor(pycnn_bias, dtype=torch.float32)
        
        pycnn_output_weight = self._to_cpu(self.weights["wo"])
        pycnn_output_bias = self._to_cpu(self.biases["bo"])
        
        state_dict['output_layer.weight'] = torch.tensor(pycnn_output_weight, dtype=torch.float32)
        state_dict['output_layer.bias'] = torch.tensor(pycnn_output_bias, dtype=torch.float32)
        
        pytorch_model.load_state_dict(state_dict)
        
        model_data = {
            'model_state_dict': pytorch_model.state_dict(),
            'layers': self.layers,
            'num_classes': len(self.classes),
            'classes': self.classes,
            'filters': self._to_cpu(self.filters),
            'image_size': self.image_size,
            'pycnn_version': 'exported'
        }
        
        torch.save(model_data, save_path)
        
        print(f"Model exported to PyTorch format: {save_path}")
        print(f"Size: {round(os.path.getsize(save_path) / (1024 * 1024), 2)}MB")