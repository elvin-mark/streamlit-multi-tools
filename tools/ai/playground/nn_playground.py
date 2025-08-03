import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide", page_title="Neural Network Playground")

st.title("🧠 Neural Network Playground")
st.markdown(
    "Train a simple fully connected neural network using only NumPy, with real-time decision boundary and loss curve."
)

# --- Dataset ---
dataset_name = st.sidebar.selectbox(
    "📊 Select Dataset", ["moons", "circles", "blobs", "spirals"]
)
n_samples = st.sidebar.slider("Number of samples", 100, 1000, 300)
noise = st.sidebar.slider("Noise", 0.0, 1.0, 0.2)


def generate_data(name, n_samples, noise):
    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=0)
    elif name == "circles":
        X, y = make_circles(
            n_samples=n_samples, noise=noise, factor=0.5, random_state=0
        )
    elif name == "blobs":
        X, y = make_blobs(
            n_samples=n_samples, centers=2, cluster_std=noise + 0.5, random_state=0
        )
    elif name == "spirals":
        theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi
        r_a = 2 * theta + np.pi
        data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
        data_b = np.array([-np.cos(theta) * r_a, -np.sin(theta) * r_a]).T
        X = np.concatenate([data_a, data_b])
        y = np.array([0] * n_samples + [1] * n_samples)
        return X, y
    return X, y


X, y = generate_data(dataset_name, n_samples, noise)
y = y.reshape(-1, 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# --- Model configuration ---
layers = st.sidebar.slider("Hidden Layers", 1, 5, 2)
neurons_per_layer = st.sidebar.slider("Neurons per Layer", 1, 20, 8)
activation_name = st.sidebar.selectbox("Activation", ["tanh", "relu", "sigmoid"])
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1, step=0.01)
epochs = st.sidebar.slider("Epochs", 10, 1000, 200)
l2_lambda = st.sidebar.slider("L2 Regularization (λ)", 0.0, 1.0, 0.0)


# --- Activations ---
def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2


activation_funcs = {
    "relu": (relu, relu_deriv),
    "sigmoid": (sigmoid, sigmoid_deriv),
    "tanh": (tanh, tanh_deriv),
}

act_fn, act_fn_deriv = activation_funcs[activation_name]


# --- Initialize weights ---
def init_weights(shape):
    return np.random.randn(*shape) * 0.1


# --- MLP ---
class MLP:
    def __init__(self, input_dim, hidden_sizes, output_dim):
        self.layers = []
        self.weights = []
        self.biases = []
        layer_sizes = [input_dim] + hidden_sizes + [output_dim]

        for i in range(len(layer_sizes) - 1):
            self.weights.append(init_weights((layer_sizes[i], layer_sizes[i + 1])))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def forward(self, x):
        self.zs = []
        self.activations = [x]
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.zs.append(z)
            a = act_fn(z)
            self.activations.append(a)

        # Output layer (sigmoid for binary)
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.zs.append(z)
        out = sigmoid(z)
        self.activations.append(out)
        return out

    def backward(self, x, y, lr, l2_lambda):
        m = y.shape[0]
        output = self.activations[-1]
        dz = (output - y) * sigmoid_deriv(self.zs[-1])
        dw = np.dot(self.activations[-2].T, dz) / m + l2_lambda * self.weights[-1]
        db = np.sum(dz, axis=0, keepdims=True) / m
        self.weights[-1] -= lr * dw
        self.biases[-1] -= lr * db

        for i in reversed(range(len(self.weights) - 1)):
            da = np.dot(dz, self.weights[i + 1].T)
            dz = da * act_fn_deriv(self.zs[i])
            dw = np.dot(self.activations[i].T, dz) / m + l2_lambda * self.weights[i]
            db = np.sum(dz, axis=0, keepdims=True) / m
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db

    def compute_loss(self, y_pred, y_true):
        loss = -np.mean(
            y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8)
        )
        l2_penalty = l2_lambda * sum(np.sum(w**2) for w in self.weights)
        return loss + l2_penalty


# --- Train ---
model = MLP(input_dim=2, hidden_sizes=[neurons_per_layer] * layers, output_dim=1)
losses = []

with st.spinner("Training..."):
    for epoch in range(epochs):
        y_pred = model.forward(X_train)
        model.backward(X_train, y_train, learning_rate, l2_lambda)
        loss = model.compute_loss(y_pred, y_train)
        losses.append(loss)

# --- Plotting ---
col1, col2 = st.columns(2)

with col1:
    # Decision boundary
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.forward(grid).reshape(xx.shape)

    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, preds, cmap="RdBu", alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap="RdBu", edgecolors="k")
    plt.title("Decision Boundary")
    st.pyplot(plt.gcf())

with col2:
    # Loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    st.pyplot(plt.gcf())
