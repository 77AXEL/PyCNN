from PIL import Image
from os import listdir
from pickle import load, dump
import matplotlib.pyplot as plt
from pycnn.pooling import max_pooling as cython_max_pooling

class CNN:
    def __init__(self):
        self.optimizer = "sgd"
        self.filters = [
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
            [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        ]
        self.use_cuda = False
        self._setup_backend(False)

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
        
        for key in self.weights:
            self.m[key] = self.zeros(self.weights[key].shape, dtype=float)
            self.v[key] = self.zeros(self.weights[key].shape, dtype=float)
        for key in self.biases:
            self.m[key] = self.zeros(self.biases[key].shape, dtype=float)
            self.v[key] = self.zeros(self.biases[key].shape, dtype=float)
            
        print("Adam optimizer enabled.")

    def init(self, image_size, batch_size, layers, learning_rate, epochs, dataset_path, max_image=None, filters=None):
        self.image_size = image_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_path = dataset_path
        self.max_image = max_image

        if filters != None:
            self.filters = filters

        self.input_size = ((image_size-2)//2)**2*len(self.filters)
        self.classes = listdir(self.dataset_path)
        self.random.seed(42)
        self.layers = layers
        self.weights = {}
        self.biases = {}
        
        previous_layer = self.input_size
        for i, layer_size in enumerate(layers):
            weight_key = f"w{i+1}"
            bias_key = f"b{i+1}"
            
            self.weights[weight_key] = self.random.randn(layer_size, previous_layer) * (2 / previous_layer) ** 0.5
            self.biases[bias_key] = self.random.randn(layer_size) * 0.01
            previous_layer = layer_size

        self.weights["wo"] = self.random.randn(len(self.classes), previous_layer) * (2 / previous_layer) ** 0.5
        self.biases["bo"] = self.random.randn(len(self.classes)) * 0.01
        
        self.data = []
        self.labels = []
        
        print(f"Network initialized with {len(layers)} hidden layers: {layers}")
    
    def _softmax_batch(self, x):
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

    def _cross_entropy_batch(self, preds, labels):
        return -(labels * self.log(preds + 1e-9)).sum(axis=1)
    
    def _max_pooling(self, feature_map):
        if self.use_cuda:
            stride = size = 2
            h, w = feature_map.shape
            pooled = self.zeros(((h - size) // stride + 1, (w - size) // stride + 1), dtype=float)
            for i in range(0, h - size + 1, stride):
                for j in range(0, w - size + 1, stride):
                    pooled[i // stride][j // stride] = self.max(feature_map[i:i+2, j:j+2])
            return pooled
        else:
            return cython_max_pooling(feature_map)

    def _preprocess_image(self, path, image_size, filters):
        img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS).load()
        r = self.zeros((image_size, image_size), dtype=float)
        g = self.zeros((image_size, image_size), dtype=float)
        b = self.zeros((image_size, image_size), dtype=float)
        
        for row in range(image_size):
            for col in range(image_size):
                rr, gg, bb = img[col, row]
                r[row][col] = rr / 255.0
                g[row][col] = gg / 255.0
                b[row][col] = bb / 255.0
        
        feature_maps = []
        for filt in filters:
            filt = self._to_backend(filt)
            conv = self.convolve2d(r, filt, mode='valid') + self.convolve2d(g, filt, mode='valid') + self.convolve2d(b, filt, mode='valid')
            conv = self.maximum(conv, 0)
            pooled = self._max_pooling(conv)
            feature_maps.append(pooled)
        return self.array(feature_maps).reshape(-1)

    def load_dataset(self):
        for idx, cls in enumerate(self.classes):
            folder = f"{self.dataset_path}/{cls}"
            x = 0
            for img_file in listdir(folder):
                x += 1
                if self.max_image is not None and x >= self.max_image:
                    break
                img_path = f"{folder}/{img_file}"
                print(f"\rLoading dataset: {img_path}   ", end="")
                try:
                    vec = self._preprocess_image(img_path, self.image_size, self.filters)
                    self.data.append(vec)
                    one_hot = [0] * len(self.classes)
                    one_hot[idx] = 1
                    self.labels.append(self.array(one_hot))
                except:
                    pass

    def train_model(self, visualize=False, early_stop=0):
        print()
        self.visualize = visualize
        self.data = self.array(self.data)
        self.labels = self.array(self.labels)
        num_samples = len(self.data)
        best_loss = float('inf')
        stop_counter = 0
        patience = early_stop

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

                activations = [batch_x]
                z_values = []
                
                current_input = batch_x
                for i in range(len(self.layers)):
                    weight_key = f"w{i+1}"
                    bias_key = f"b{i+1}"
                    
                    z = current_input @ self.weights[weight_key].T + self.biases[bias_key]
                    z_values.append(z)
                    activation = self.maximum(z, 0)
                    activations.append(activation)
                    current_input = activation

                z_out = current_input @ self.weights["wo"].T + self.biases["bo"]
                z_values.append(z_out)
                out = self._softmax_batch(z_out)

                loss = self._cross_entropy_batch(out, batch_y).sum()
                total_loss += loss

                pred = self.argmax(out, axis=1)
                true = self.argmax(batch_y, axis=1)
                correct += self.sum(pred == true)

                gradients_w = {}
                gradients_b = {}
                
                dL_dz_out = out - batch_y
                gradients_w["wo"] = dL_dz_out.T @ activations[-1]
                gradients_b["bo"] = self.sum(dL_dz_out, axis=0)
                
                dL_da = dL_dz_out @ self.weights["wo"]
                
                for i in reversed(range(len(self.layers))):
                    weight_key = f"w{i+1}"
                    bias_key = f"b{i+1}"
                    
                    dL_dz = dL_da * (z_values[i] > 0)
                    
                    gradients_w[weight_key] = dL_dz.T @ activations[i]
                    gradients_b[bias_key] = self.sum(dL_dz, axis=0)
                    
                    if i > 0:
                        dL_da = dL_dz @ self.weights[weight_key]

                if self.optimizer == "sgd":
                    for key in gradients_w:
                        self.weights[key] -= self.learning_rate * gradients_w[key]
                    for key in gradients_b:
                        self.biases[key] -= self.learning_rate * gradients_b[key]

                elif self.optimizer == "adam":
                    self.t += 1
                    
                    for key, grad in gradients_w.items():
                        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad * grad)
                        
                        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                        v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                        
                        update = self.learning_rate * m_hat / (self.sqrt(v_hat) + self.eps)
                        self.weights[key] -= update
                    
                    for key, grad in gradients_b.items():
                        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad * grad)
                        
                        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                        v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                        
                        update = self.learning_rate * m_hat / (self.sqrt(v_hat) + self.eps)
                        self.biases[key] -= update

            epoch_loss = total_loss / max(1, num_samples)
            epoch_acc = correct / max(1, num_samples)

            epoch_loss = self._to_cpu(epoch_loss)
            epoch_acc = self._to_cpu(epoch_acc)

            if early_stop:
                if epoch_loss < best_loss - 1e-8:
                    best_loss = epoch_loss
                    stop_counter = 0
                else:
                    stop_counter += 1
                    if stop_counter >= patience:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        break

            print(f"\rTraining model: epoch={epoch + 1}/{self.epochs}  correct={int(self._to_cpu(correct))}/{num_samples}  acc={epoch_acc:.4f}  loss={epoch_loss:.6f}   ", end="")

            if self.visualize:
                epoch_nums.append(epoch + 1)
                losses.append(float(epoch_loss))
                accs.append(float(epoch_acc))

                loss_line.set_data(epoch_nums, losses)
                acc_line.set_data(epoch_nums, accs)

                ax.relim()
                ax.autoscale_view()

                ax.set_xlim(1, max(self.epochs, len(epoch_nums)))
                fig.canvas.draw()
                fig.canvas.flush_events()

        print()
        if self.visualize:
            plt.ioff()
            plt.show()

    def save_model(self, path="model.bin"):
        """Save model with automatic CPU conversion for cross-platform compatibility"""
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
            dump(model_data, model_file)

        print(f"Model saved in {path}")
    
    def load_model(self, model_path):
        """Load model with automatic backend conversion"""
        with open(model_path, "rb") as model_file:
            model_data = load(model_file)
        
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
        print(f"Training backend: {training_backend}")
        print(f"Current backend: {current_backend}")
        print(f"Network architecture: {len(self.layers)} hidden layers: {self.layers}")
        if trained_with_cuda != self.use_cuda:
            print("Backend conversion completed automatically.")

    def predict(self, img_path):
        """Make prediction on a single image"""
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
        return predicted_label