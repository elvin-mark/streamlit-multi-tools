import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("🧠 CNN Visualizer for Digits Dataset")

# -----------------------
# Load and prepare data
# -----------------------
digits = load_digits()
X = digits.images
y = digits.target

# Normalize and reshape
X = X / 16.0  # Now in [0, 1]
X = np.expand_dims(X, 1)  # (n, 1, 8, 8)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
)
test_ds = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)
)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)


# -----------------------
# Define CNN model
# -----------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # 8 filters
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # -> [B, 8, 8, 8]
        x = F.max_pool2d(x, 2)  # -> [B, 8, 4, 4]
        x = F.relu(self.conv2(x))  # -> [B, 16, 4, 4]
        x = F.max_pool2d(x, 2)  # -> [B, 16, 2, 2]
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@st.cache_resource
def train_model():
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 10

    history = {"train_loss": [], "train_acc": []}

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            out = model(images)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(images)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

    return model, history


# -----------------------
# Train the model
# -----------------------
with st.spinner("Training CNN model..."):
    model, history = train_model()
st.success("Training complete!")

# -----------------------
# Show training stats
# -----------------------
st.subheader("📈 Training Metrics")

col1, col2 = st.columns(2)
with col1:
    st.line_chart(history["train_loss"], use_container_width=True)
    st.caption("Training Loss")

with col2:
    st.line_chart(history["train_acc"], use_container_width=True)
    st.caption("Training Accuracy")

# -----------------------
# Filter Visualization
# -----------------------
st.subheader("🔍 Convolutional Filters (First Layer)")

filters = model.conv1.weight.data.numpy()
fig, axes = plt.subplots(1, len(filters), figsize=(10, 2))
for i, ax in enumerate(axes):
    ax.imshow(filters[i, 0], cmap="gray")
    ax.axis("off")
    ax.set_title(f"F{i}")
st.pyplot(fig)

# -----------------------
# Feature Map Visualization
# -----------------------
st.subheader("🧪 Feature Maps on Sample Input")

sample_idx = st.slider("Select test image index", 0, len(X_test) - 1, 0)

sample_image = torch.tensor(X_test[sample_idx : sample_idx + 1], dtype=torch.float32)
st.image(
    X_test[sample_idx][0],
    width=150,
    caption=f"Original image (label = {y_test[sample_idx]})",
)


def get_feature_maps(model, x):
    with torch.no_grad():
        fmap1 = F.relu(model.conv1(x))
        fmap2 = F.relu(model.conv2(F.max_pool2d(fmap1, 2)))
    return fmap1, fmap2


fmap1, fmap2 = get_feature_maps(model, sample_image)

st.markdown("**Feature maps after Conv1**")
fig1, ax1 = plt.subplots(1, 8, figsize=(12, 2))
for i in range(8):
    ax1[i].imshow(fmap1[0, i], cmap="gray")
    ax1[i].axis("off")
st.pyplot(fig1)

st.markdown("**Feature maps after Conv2**")
fig2, ax2 = plt.subplots(1, 8, figsize=(12, 2))
for i in range(8):
    ax2[i].imshow(fmap2[0, i], cmap="gray")
    ax2[i].axis("off")
st.pyplot(fig2)
