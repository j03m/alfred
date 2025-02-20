import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# --- Synthetic Dataset ---
def generate_data(num_samples=1000):
    X = torch.randn(num_samples, 10)
    # Create a non-linear relationship for the target
    y = (torch.sin(X[:, 0] * 3) + torch.cos(X[:, 1] * 2) > 0).float().view(-1, 1)
    return X, y

X_train, y_train = generate_data(2000)
X_val, y_val = generate_data(500)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=500)

# --- Model Definitions ---

class CorrectModel(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.bn1 = nn.BatchNorm1d(20)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

class IncorrectModel1(nn.Module):  # Dropout before BN
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(20)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

class IncorrectModel2(nn.Module):  # Dropout before Activation
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.bn1 = nn.BatchNorm1d(20)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

# --- Training Function ---
def train_model(model, train_loader, val_loader, epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
        train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
        val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

# --- Run Experiments ---
dropout_rate = 0.3

correct_model = CorrectModel(dropout_rate)
incorrect_model1 = IncorrectModel1(dropout_rate)
incorrect_model2 = IncorrectModel2(dropout_rate)

print("Training Correct Model...")
correct_train_losses, correct_val_losses = train_model(correct_model, train_loader, val_loader)

print("\nTraining Incorrect Model 1 (Dropout before BN)...")
incorrect1_train_losses, incorrect1_val_losses = train_model(incorrect_model1, train_loader, val_loader)


print("\nTraining Incorrect Model 2 (Dropout before Activation)...")
incorrect2_train_losses, incorrect2_val_losses = train_model(incorrect_model2, train_loader, val_loader)
# --- Plot Results ---
import matplotlib.pyplot as plt

epochs = range(1, len(correct_train_losses) + 1)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.plot(epochs, correct_train_losses, 'b-', label='Train Loss')
plt.plot(epochs, correct_val_losses, 'r-', label='Val Loss')
plt.title('Correct Model (BN-Act-Dropout)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epochs, incorrect1_train_losses, 'b-', label='Train Loss')
plt.plot(epochs, incorrect1_val_losses, 'r-', label='Val Loss')
plt.title('Incorrect Model 1 (Dropout-BN-Act)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.subplot(1, 3, 3)
plt.plot(epochs, incorrect2_train_losses, 'b-', label='Train Loss')
plt.plot(epochs, incorrect2_val_losses, 'r-', label='Val Loss')
plt.title('Incorrect Model 2 (BN-Dropout-Act)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# --- Analyze the Gap ---
print("\n--- Loss Gap (Val - Train) at Last Epoch ---")
print(f"Correct Model:    {correct_val_losses[-1] - correct_train_losses[-1]:.4f}")
print(f"Incorrect Model 1: {incorrect1_val_losses[-1] - incorrect1_train_losses[-1]:.4f}")
print(f"Incorrect Model 2: {incorrect2_val_losses[-1] - incorrect2_train_losses[-1]:.4f}")