from PIL import Image
from os import listdir
from pickle import load, dump
import matplotlib.pyplot as plt

class CNN:
    def __init__(self):
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
            
            if hasattr(self, 'weights_1'):
                self.weights_1 = self._to_backend(self._to_cpu(self.weights_1))
                self.weights_2 = self._to_backend(self._to_cpu(self.weights_2))
                self.weights_out = self._to_backend(self._to_cpu(self.weights_out))
                self.biases_1 = self._to_backend(self._to_cpu(self.biases_1))
                self.biases_2 = self._to_backend(self._to_cpu(self.biases_2))
                self.biases_out = self._to_backend(self._to_cpu(self.biases_out))
                
            if hasattr(self, 'data') and len(self.data) > 0:
                self.data = self._to_backend(self._to_cpu(self.data))
                self.labels = self._to_backend(self._to_cpu(self.labels))

    def init(self, image_size, batch_size, h1, h2, learning_rate, epochs, dataset_path, max_image=None, filters=None):
        self.image_size = image_size
        self.batch_size = batch_size
        self.h1 = h1
        self.h2 = h2
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_path = dataset_path
        self.max_image = max_image

        if filters != None:
            self.filters = filters

        self.input_size = ((image_size-2)//2)**2*len(self.filters)
        self.classes = listdir(self.dataset_path)
        self.random.seed(42)
        
        self.weights_1 = self.random.randn(h1, self.input_size) * (2 / self.input_size) ** 0.5
        self.weights_2 = self.random.randn(h2, h1) * (2 / h1) ** 0.5
        self.weights_out = self.random.randn(len(self.classes), h2) * (2 / h2) ** 0.5
        self.biases_1 = self.random.randn(h1) * 0.01
        self.biases_2 = self.random.randn(h2) * 0.01
        self.biases_out = self.random.randn(len(self.classes)) * 0.01
        self.data = []
        self.labels = []
    
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
        stride = size = 2
        h, w = feature_map.shape
        pooled = self.zeros(((h - size) // stride + 1, (w - size) // stride + 1), dtype=float)
        for i in range(0, h - size + 1, stride):
            for j in range(0, w - size + 1, stride):
                pooled[i // stride][j // stride] = self.max(feature_map[i:i+2, j:j+2])
        return pooled

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
        
    def train_model(self, visualize=False):
        print()
        self.data = self.array(self.data)
        self.labels = self.array(self.labels)
        num_samples = len(self.data)

        if visualize:
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

                z1 = batch_x @ self.weights_1.T + self.biases_1
                l1 = self.maximum(z1, 0)

                z2 = l1 @ self.weights_2.T + self.biases_2
                l2 = self.maximum(z2, 0)

                z3 = l2 @ self.weights_out.T + self.biases_out
                out = self._softmax_batch(z3)

                loss = self._cross_entropy_batch(out, batch_y).sum()
                total_loss += loss

                pred = self.argmax(out, axis=1)
                true = self.argmax(batch_y, axis=1)
                correct += self.sum(pred == true)

                dL_dz3 = out - batch_y

                dL_dw_out = dL_dz3.T @ l2
                dL_db_out = self.sum(dL_dz3, axis=0)

                dL_dl2 = dL_dz3 @ self.weights_out
                dL_dz2 = dL_dl2 * (z2 > 0)

                dL_dw2 = dL_dz2.T @ l1
                dL_db2 = self.sum(dL_dz2, axis=0)

                dL_dl1 = dL_dz2 @ self.weights_2
                dL_dz1 = dL_dl1 * (z1 > 0)

                dL_dw1 = dL_dz1.T @ batch_x
                dL_db1 = self.sum(dL_dz1, axis=0)

                self.weights_out -= self.learning_rate * dL_dw_out
                self.biases_out -= self.learning_rate * dL_db_out

                self.weights_2 -= self.learning_rate * dL_dw2
                self.biases_2 -= self.learning_rate * dL_db2

                self.weights_1 -= self.learning_rate * dL_dw1
                self.biases_1 -= self.learning_rate * dL_db1

            epoch_loss = total_loss / max(1, num_samples)
            epoch_acc = correct / max(1, num_samples)

            epoch_loss = self._to_cpu(epoch_loss)
            epoch_acc = self._to_cpu(epoch_acc)

            print(f"\rTraining model: epoch={epoch + 1}/{self.epochs}  correct={int(self._to_cpu(correct))}/{num_samples}  acc={epoch_acc:.4f}  loss={epoch_loss:.6f}   ", end="")

            if visualize:
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
        if visualize:
            plt.ioff()
            plt.show()

    def save_model(self, path="model.bin"):
        """Save model with automatic CPU conversion for cross-platform compatibility"""
        model_data = {
            "w1": self._to_cpu(self.weights_1),
            "w2": self._to_cpu(self.weights_2),
            "wo": self._to_cpu(self.weights_out),
            "b1": self._to_cpu(self.biases_1),
            "b2": self._to_cpu(self.biases_2),
            "bo": self._to_cpu(self.biases_out),
            "filters": self._to_cpu(self.filters),
            "image_size": int(self._to_cpu(self.image_size)),
            "classes": list(self.classes),
            "trained_with_cuda": self.use_cuda
        }
        
        with open(path, "wb") as model_file:
            dump(model_data, model_file)

        print(f"Model saved in {path}")
    
    def load_model(self, model_path):
        """Load model with automatic backend conversion"""
        with open(model_path, "rb") as model_file:
            model_data = load(model_file)
            
        self.weights_1 = self._to_backend(model_data["w1"])
        self.weights_2 = self._to_backend(model_data["w2"])  
        self.weights_out = self._to_backend(model_data["wo"])
        self.biases_1 = self._to_backend(model_data["b1"])
        self.biases_2 = self._to_backend(model_data["b2"])
        self.biases_out = self._to_backend(model_data["bo"])
        self.filters = model_data["filters"]
        self.image_size = model_data["image_size"]
        self.classes = model_data["classes"]
        
        trained_with_cuda = model_data.get("trained_with_cuda", False)
        current_backend = "CUDA" if self.use_cuda else "CPU"
        training_backend = "CUDA" if trained_with_cuda else "CPU"
        
        print(f"Model loaded successfully!")
        print(f"Training backend: {training_backend}")
        print(f"Current backend: {current_backend}")
        if trained_with_cuda != self.use_cuda:
            print("Backend conversion completed automatically.")

    def predict(self, img_path):
        """Make prediction on a single image"""
        test_img = self._preprocess_image(img_path, self.image_size, self.filters)
        l1 = self.maximum(self.dot(self.weights_1, test_img) + self.biases_1, 0)
        l2 = self.maximum(self.dot(self.weights_2, l1) + self.biases_2, 0)
        
        logits = self.dot(self.weights_out, l2) + self.biases_out
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