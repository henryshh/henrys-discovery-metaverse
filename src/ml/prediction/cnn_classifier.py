import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Tuple, List

class TighteningCNN(nn.Module):
    def __init__(self, sequence_length: int = 200, num_classes: int = 2):
        super(TighteningCNN, self).__init__()
        # Input shape: (Batch, 1, sequence_length)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate flattened dimension
        self.flattened_dim = 32 * (sequence_length // 4)
        
        self.fc1 = nn.Linear(self.flattened_dim, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x is expected to be shape (Batch, 1, SeqLength)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def preprocess_curves(curves: List[np.ndarray], target_length: int = 200) -> np.ndarray:
    """Interpolate all curves to target_length and row-standardize them."""
    processed = []
    for curve in curves:
        # Interpolate
        x_old = np.linspace(0, 1, len(curve))
        x_new = np.linspace(0, 1, target_length)
        curve_new = np.interp(x_new, x_old, curve)
        
        # Standardize (Z-score normalize per curve)
        std = np.std(curve_new)
        if std > 0:
            curve_new = (curve_new - np.mean(curve_new)) / std
        else:
            curve_new = curve_new - np.mean(curve_new)
            
        processed.append(curve_new)
    return np.array(processed)

def train_cnn_model(
    X_train: np.ndarray, y_train: np.ndarray, 
    X_val: np.ndarray, y_val: np.ndarray, 
    epochs: int = 50, batch_size: int = 16, lr: float = 0.001,
    progress_callback=None
) -> Tuple[nn.Module, List[float], List[float], float]:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reshape for 1D CNN: (Batch, Channels, Length)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    seq_length = X_train.shape[1]
    model = TighteningCNN(sequence_length=seq_length, num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_accuracies = []
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            if len(y_val_tensor) > 0:
                outputs = model(X_val_tensor)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == y_val_tensor).sum().item()
                acc = correct / len(y_val_tensor)
            else:
                acc = 0.0
            val_accuracies.append(acc)
            if acc > best_acc:
                best_acc = acc
        
        if progress_callback:
            progress_callback(epoch + 1, epochs, epoch_loss, acc)
            
    return model, train_losses, val_accuracies, best_acc

class TighteningAutoencoder(nn.Module):
    def __init__(self, sequence_length: int = 200):
        super(TighteningAutoencoder, self).__init__()
        # Encoder (Input: Batch, 1, seq_len)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        original_size = x.size(2)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # Resize to match perfectly in case of length dimension errors
        if decoded.size(2) != original_size:
            if decoded.size(2) > original_size:
                decoded = decoded[:, :, :original_size]
            else:
                padding = original_size - decoded.size(2)
                decoded = nn.functional.pad(decoded, (0, padding))
                
        return decoded

def train_autoencoder_model(
    X_train: np.ndarray, 
    X_val: np.ndarray, 
    epochs: int = 50, batch_size: int = 16, lr: float = 0.001,
    progress_callback=None
) -> Tuple[nn.Module, List[float], List[float], float]:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor) # Autoencoder target is input
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    seq_length = X_train.shape[1]
    model = TighteningAutoencoder(sequence_length=seq_length).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            if len(X_val_tensor) > 0:
                outputs = model(X_val_tensor)
                val_loss = criterion(outputs, X_val_tensor).item()
            else:
                val_loss = 0.0
            val_losses.append(val_loss)
        
        if progress_callback:
            progress_callback(epoch + 1, epochs, epoch_loss, val_loss)
            
    return model, train_losses, val_losses, val_losses[-1] if val_losses else 0.0
