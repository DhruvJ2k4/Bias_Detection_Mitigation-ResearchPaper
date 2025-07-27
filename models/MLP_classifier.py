"""
PyTorch MLP Classifier for Loan Approval Prediction
---------------------------------------------------
Modular feedforward neural network (2 hidden layers).
Supports early stopping, dropout, and easy integration
with fairness-aware training (to be added later).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Dict
import numpy as np

class MLPClassifier(nn.Module):
    """A simple 2-hidden-layer MLP for binary classification."""
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, int] = (64, 32),
                 dropout: float = 0.3):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def train_mlp_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    lr: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 50,
    patience: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[MLPClassifier, Dict[str, list]]:
    """
    Train a PyTorch MLP

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training features and labels (binary 0/1).
    X_val, y_val : np.ndarray
        Validation features and labels.
    input_dim : int
        Number of input features.
    lr : float, optional
        Learning rate.
    batch_size : int, optional
        Mini-batch size.
    epochs : int, optional
        Maximum training epochs.
    patience : int, optional
        Early stopping patience (in epochs).
    device : str, optional
        "cuda" or "cpu".

    Returns
    -------
    model : MLPClassifier
        Trained PyTorch model.
    history : Dict[str, list]
        Training and validation losses over epochs.
    """
    # Prepare DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32).view(-1, 1))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model, Loss, Optimizer
    model = MLPClassifier(input_dim=input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop with early stopping
    best_val_loss = np.inf
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "results/best_mlp.pth")  # Save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best weights
    model.load_state_dict(torch.load("results/best_mlp.pth"))
    return model, history

def predict_mlp(model: MLPClassifier, X: np.ndarray, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> np.ndarray:
    """Generate predictions (0/1) for new samples."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = model(X_tensor).cpu().numpy().flatten()
    return (probs >= 0.5).astype(int), probs
