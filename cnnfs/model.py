from PIL import Image
from numpy import zeros, maximum, dot, random, array, exp, log
from scipy.signal import convolve2d
from os import listdir
from pickle import load, dump

def softmax_batch(x):
    x = x - x.max(axis=1, keepdims=True)
    exps = exp(x)
    return exps / exps.sum(axis=1, keepdims=True)

def cross_entropy_batch(preds, labels):
    return - (labels * log(preds + 1e-9)).sum(axis=1)

def max_pooling(feature_map):
    stride = size = 2
    h, w = feature_map.shape
    pooled = zeros(((h - size) // stride + 1, (w - size) // stride + 1), dtype=float)
    for i in range(0, h - size + 1, stride):
        for j in range(0, w - size + 1, stride):
            pooled[i // stride][j // stride] = feature_map[i:i+2, j:j+2].max()
    return pooled

def preprocess_image(path, image_size, filters):
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS).load()
    r = zeros((image_size, image_size), dtype=float)
    g = zeros((image_size, image_size), dtype=float)
    b = zeros((image_size, image_size), dtype=float)
    for row in range(image_size):
        for col in range(image_size):
            rr, gg, bb = img[col, row]
            r[row][col] = rr / 255.0
            g[row][col] = gg / 255.0
            b[row][col] = bb / 255.0
    feature_maps = []
    for filt in filters:
        conv = convolve2d(r, filt, mode='valid') + convolve2d(g, filt, mode='valid') + convolve2d(b, filt, mode='valid')
        conv = maximum(conv, 0)
        pooled = max_pooling(conv)
        feature_maps.append(pooled)
    return array(feature_maps).reshape(-1)

class CNN:
    def __init__(self):
        random.seed(42)
        self.filters = [
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
            [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        ]

    def init(self, image_size, batch_size, h1, h2, learning_rate, epochs, dataset_path, max_image=None):
        self.image_size = image_size
        self.batch_size = batch_size
        self.h1 = h1
        self.h2 = h2
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_path = dataset_path
        self.max_image = max_image
        self.input_size = ((image_size-2)//2)**2*3
        self.classes = listdir(self.dataset_path)
        self.weights_1 = random.randn(h1, self.input_size) * (2 / self.input_size) ** 0.5
        self.weights_2 = random.randn(h2, h1) * (2 / h1) ** 0.5
        self.weights_out = random.randn(len(self.classes), h2) * (2 / h2) ** 0.5
        self.biases_1 = random.randn(h1) * 0.01
        self.biases_2 = random.randn(h2) * 0.01
        self.biases_out = random.randn(len(self.classes)) * 0.01
        self.data = []
        self.labels = []
    
    def load_dataset(self):
        for idx, cls in enumerate(self.classes):
            folder = f"data/{cls}"
            x = 0
            for img_file in listdir(folder):
                x += 1
                if x == self.max_image:
                    break
                img_path = f"{folder}/{img_file}"
                print(f"\rLoading dataset: {img_path}   ", end="")
                try:
                    vec = preprocess_image(img_path, self.image_size, self.filters)
                    self.data.append(vec)
                    one_hot = [0] * len(self.classes)
                    one_hot[idx] = 1
                    self.labels.append(array(one_hot))
                except:
                    pass
        
    def train_model(self):
        print()
        self.data = array(self.data)
        self.labels = array(self.labels)
        num_samples = len(self.data)

        for epoch in range(self.epochs):
            perm = random.permutation(num_samples)
            correct = 0
            total_loss = 0
            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                batch_idx = perm[start:end]
                batch_x = self.data[batch_idx]
                batch_y = self.labels[batch_idx]

                z1 = batch_x @ self.weights_1.T + self.biases_1
                l1 = maximum(z1, 0)

                z2 = l1 @ self.weights_2.T + self.biases_2
                l2 = maximum(z2, 0)

                z3 = l2 @ self.weights_out.T + self.biases_out
                out = softmax_batch(z3)

                loss = cross_entropy_batch(out, batch_y).sum()
                total_loss += loss

                pred = out.argmax(axis=1)
                true = batch_y.argmax(axis=1)
                correct += (pred == true).sum()

                dL_dz3 = out - batch_y

                dL_dw_out = dL_dz3.T @ l2
                dL_db_out = dL_dz3.sum(axis=0)

                dL_dl2 = dL_dz3 @ self.weights_out
                dL_dz2 = dL_dl2 * (z2 > 0)

                dL_dw2 = dL_dz2.T @ l1
                dL_db2 = dL_dz2.sum(axis=0)

                dL_dl1 = dL_dz2 @ self.weights_2
                dL_dz1 = dL_dl1 * (z1 > 0)

                dL_dw1 = dL_dz1.T @ batch_x
                dL_db1 = dL_dz1.sum(axis=0)

                self.weights_out -= self.learning_rate * dL_dw_out
                self.biases_out -= self.learning_rate * dL_db_out

                self.weights_2 -= self.learning_rate * dL_dw2
                self.biases_2 -= self.learning_rate * dL_db2

                self.weights_1 -= self.learning_rate * dL_dw1
                self.biases_1 -= self.learning_rate * dL_db1

            print(f"\rTraining model: epoch={epoch + 1}/{self.epochs}  correct={correct} loss={total_loss}   ", end="")

        print()

    def save_model(self):
        with open("model.bin", "wb") as model_file:
            dump({
                "w1": self.weights_1,
                "w2": self.weights_2,
                "wo": self.weights_out,
                "b1": self.biases_1,
                "b2": self.biases_2,
                "bo": self.biases_out
            }, model_file)
        print("Model saved to model.bin")
    
    def load_model(self, model_path):
        with open(model_path, "rb") as model_file:
            model_data = load(model_file)
            self.weights_1 = model_data["w1"]
            self.weights_2 = model_data["w2"]
            self.weights_out = model_data["wo"]
            self.biases_1 = model_data["b1"]
            self.biases_2 = model_data["b2"]
            self.biases_out = model_data["bo"]

    def predict(self, img_path):
        test_img = preprocess_image(img_path, self.image_size, self.filters)
        l1 = maximum(dot(self.weights_1, test_img) + self.biases_1, 0)
        l2 = maximum(dot(self.weights_2, l1) + self.biases_2, 0)
        out = softmax_batch((dot(self.weights_out, l2) + self.biases_out).reshape(1, -1))[0]
        predicted_label = self.classes[out.argmax()]
        return predicted_label