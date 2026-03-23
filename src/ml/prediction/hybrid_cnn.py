import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pywt
from typing import Tuple, List
import sys
from pathlib import Path

# Important: feature_extractor is one level up in src/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from feature_extractor import TighteningFeatureExtractor
except ImportError:
    # Fallback if not found
    pass

class HybridCNNDWT(nn.Module):
    def __init__(self, dwt_seq_length: int, num_expert_features: int, num_classes: int = 2):
        super(HybridCNNDWT, self).__init__()
        
        # Branch 1: DWT CNN (Input: Batch, 2 channels (cA, cD), dwt_seq_length)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Calculate flattened dimension of CNN
        self.cnn_flat_dim = 32 * (dwt_seq_length // 4)
        
        # Branch 2: Expert Features Dense Network
        # The TighteningFeatureExtractor returns around 34 features.
        self.expert_dense = nn.Sequential(
            nn.Linear(num_expert_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Fusion Classifier (CNN + Expert)
        self.fusion = nn.Sequential(
            nn.Linear(self.cnn_flat_dim + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x_dwt, x_expert):
        # DWT CNN branch
        out_cnn = self.cnn(x_dwt)
        out_cnn = out_cnn.view(out_cnn.size(0), -1)
        
        # Expert branch
        out_expert = self.expert_dense(x_expert)
        
        # Fusion
        fused = torch.cat([out_cnn, out_expert], dim=1)
        out = self.fusion(fused)
        return out

def preprocess_for_hybrid(full_curves: List[np.ndarray], target_length: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process raw 4-channel curves into:
    1. DWT coefficients array (cA, cD) of the torque.
    2. Normalized expert features array.
    """
    extractor = TighteningFeatureExtractor()
    
    x_dwt_list = []
    x_expert_list = []
    
    for curve in full_curves:
        # Extract Expert Features
        try:
            feats = extractor.extract_curve_features(curve)
        except Exception:
            # Fallback robust features if extractor fails on a corrupted curve
            feats = [0.0] * 34 
        x_expert_list.append(feats)
        
        # Process Torque for DWT
        torque = curve[:, 0]
        # Interpolate torque to target_length for consistent DWT size
        x_old = np.linspace(0, 1, len(torque))
        x_new = np.linspace(0, 1, target_length)
        torque_interp = np.interp(x_new, x_old, torque)
        
        # Standardize torque
        std = np.std(torque_interp)
        if std > 0:
            torque_interp = (torque_interp - np.mean(torque_interp)) / std
        else:
            torque_interp = torque_interp - np.mean(torque_interp)
            
        # 1-level DWT using Daubechies 4
        cA, cD = pywt.dwt(torque_interp, 'db4')
        
        # Stack as 2 channels: (2, dwt_length)
        dwt_channels = np.vstack([cA, cD])
        x_dwt_list.append(dwt_channels)
        
    X_dwt = np.array(x_dwt_list, dtype=np.float32)
    X_expert = np.array(x_expert_list, dtype=np.float32)
    
    # Handle infinite or NaN expert features
    X_expert = np.nan_to_num(X_expert, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Row-Level or Column-level Normalization for Expert features (Z-score vertically)
    means = np.mean(X_expert, axis=0)
    stds = np.std(X_expert, axis=0)
    stds[stds == 0] = 1.0
    X_expert = (X_expert - means) / stds
    
    return X_dwt, X_expert

def train_hybrid_model(
    X_dwt_train: np.ndarray, X_exp_train: np.ndarray, y_train: np.ndarray, 
    X_dwt_val: np.ndarray, X_exp_val: np.ndarray, y_val: np.ndarray, 
    epochs: int = 50, batch_size: int = 16, lr: float = 0.001,
    progress_callback=None
) -> Tuple[nn.Module, List[float], List[float], float]:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tensors
    td_train_dwt = torch.tensor(X_dwt_train, dtype=torch.float32).to(device)
    td_train_exp = torch.tensor(X_exp_train, dtype=torch.float32).to(device)
    td_train_y = torch.tensor(y_train, dtype=torch.long).to(device)
    
    td_val_dwt = torch.tensor(X_dwt_val, dtype=torch.float32).to(device)
    td_val_exp = torch.tensor(X_exp_val, dtype=torch.float32).to(device)
    td_val_y = torch.tensor(y_val, dtype=torch.long).to(device)
    
    train_dataset = TensorDataset(td_train_dwt, td_train_exp, td_train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    dwt_seq_len = X_dwt_train.shape[2]
    num_expert_feats = X_exp_train.shape[1]
    
    model = HybridCNNDWT(dwt_seq_length=dwt_seq_len, num_expert_features=num_expert_feats, num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_accuracies = []
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for b_dwt, b_exp, b_y in train_loader:
            optimizer.zero_grad()
            outputs = model(b_dwt, b_exp)
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * b_dwt.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            if len(td_val_y) > 0:
                outputs = model(td_val_dwt, td_val_exp)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == td_val_y).sum().item()
                acc = correct / len(td_val_y)
            else:
                acc = 0.0
            val_accuracies.append(acc)
            if acc > best_acc:
                best_acc = acc
        
        if progress_callback:
            progress_callback(epoch + 1, epochs, epoch_loss, acc)
            
    return model, train_losses, val_accuracies, best_acc
