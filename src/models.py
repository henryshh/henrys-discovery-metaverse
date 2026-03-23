"""
拧紧曲线 AI 模型
Deep Learning Models for Tightening Curve Classification
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict
import os


# ==================== PyTorch Dataset ====================

class TighteningCurveDataset(Dataset):
    """拧紧曲线数据集"""
    
    def __init__(self, curves: np.ndarray, labels: np.ndarray):
        self.curves = torch.FloatTensor(curves)  # [N, L, C]
        self.labels = torch.FloatTensor(labels)   # [N]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.curves[idx], self.labels[idx]


# ==================== 1D CNN Model ====================

class CNN1D(nn.Module):
    """
    1D CNN 模型 - 适合序列分类
    
    架构：
    Input -> Conv1D -> BN -> ReLU -> MaxPool -> 
           Conv1D -> BN -> ReLU -> MaxPool -> 
           Conv1D -> BN -> ReLU -> GlobalAvgPool -> 
           FC -> Sigmoid
    """
    
    def __init__(self, input_channels: int = 4, num_classes: int = 1):
        super(CNN1D, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        # 第二层卷积
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        # 第三层卷积
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # 全局平均池化 + 全连接
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [batch, seq_len, channels] -> [batch, channels, seq_len]
        x = x.permute(0, 2, 1)
        
        # Conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        # Conv2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        # Conv3
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        
        # Global pooling + FC
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x.squeeze(-1)


# ==================== LSTM Model ====================

class LSTMClassifier(nn.Module):
    """
    LSTM 分类模型 - 适合时间序列
    
    架构：
    Input -> LSTM -> LSTM -> 
           Attention -> FC -> Sigmoid
    """
    
    def __init__(self, input_size: int = 4, hidden_size: int = 64, 
                 num_layers: int = 2, num_classes: int = 1, dropout: float = 0.3):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention 层
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
        
        # Attention
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # [batch, seq_len, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden*2]
        
        # FC
        context = self.dropout(context)
        out = self.fc(context)
        out = self.sigmoid(out)
        
        return out.squeeze(-1)


# ==================== CNN-LSTM Hybrid Model ====================

class CNNLSTMHybrid(nn.Module):
    """
    CNN-LSTM 混合模型
    
    架构：
    Input -> 1D CNN (特征提取) -> LSTM (时序建模) -> FC -> Sigmoid
    """
    
    def __init__(self, input_channels: int = 4, hidden_size: int = 64, 
                 num_classes: int = 1, dropout: float = 0.3):
        super(CNNLSTMHybrid, self).__init__()
        
        # CNN 特征提取
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM 时序建模
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 全连接
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len, channels]
        x = x.permute(0, 2, 1)  # [batch, channels, seq_len]
        
        # CNN
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        # [batch, channels, seq_len] -> [batch, seq_len, channels]
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        
        # FC
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out.squeeze(-1)


# ==================== Trainer ====================

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model: nn.Module, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for curves, labels in dataloader:
            curves = curves.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(curves)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * curves.size(0)
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for curves, labels in dataloader:
                curves = curves.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(curves)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * curves.size(0)
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 50, save_path: Optional[str] = None) -> Dict:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            save_path: 模型保存路径
        
        Returns:
            训练历史
        """
        best_val_acc = 0
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.evaluate(val_loader)
            
            # 学习率调整
            self.scheduler.step(val_acc)
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            
            # 打印进度
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc and save_path:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path)
                print(f"  → Saved best model (val_acc={val_acc:.4f})")
        
        return self.train_history
    
    def predict(self, curves: np.ndarray) -> np.ndarray:
        """预测"""
        self.model.eval()
        curves = torch.FloatTensor(curves).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(curves)
            predictions = (outputs > 0.5).float().cpu().numpy()
        
        return predictions


# ==================== TCN (Temporal Convolutional Network) ====================

class TemporalBlock(nn.Module):
    """
    TCN 基本块：因果卷积 + 膨胀卷积 + 残差连接
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # 计算因果填充（只填充左侧）
        self.padding = (kernel_size - 1) * dilation
        
        # 第一层卷积
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # 第二层卷积
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=0, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # 1x1 卷积用于残差连接（当输入输出通道不同时）
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: [batch, channels, seq_len]
        residual = x
        
        # 因果填充（左侧填充，右侧不填充）
        x = torch.nn.functional.pad(x, (self.padding, 0))
        
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # 第二层
        x = torch.nn.functional.pad(x, (self.padding, 0))
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        
        # 残差连接
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        # 裁剪残差以匹配输出长度（因果卷积会缩短序列）
        if residual.size(-1) != x.size(-1):
            residual = residual[:, :, -x.size(-1):]
        
        return self.relu(x + residual)


class TCNClassifier(nn.Module):
    """
    TCN 分类模型
    
    特点：
    - 因果卷积：确保预测只依赖历史信息
    - 膨胀卷积：指数级扩大感受野
    - 残差连接：解决梯度消失问题
    """
    def __init__(self, input_channels=4, num_classes=1, 
                 hidden_channels=[32, 64, 128], kernel_size=7, dropout=0.2):
        super(TCNClassifier, self).__init__()
        
        layers = []
        num_levels = len(hidden_channels)
        
        for i in range(num_levels):
            in_channels = input_channels if i == 0 else hidden_channels[i-1]
            out_channels = hidden_channels[i]
            dilation = 2 ** i  # 指数增长的膨胀率
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, dilation, dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels[-1], num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [batch, seq_len, channels] -> [batch, channels, seq_len]
        x = x.permute(0, 2, 1)
        
        # TCN 特征提取
        x = self.network(x)
        
        # 全局池化 + 分类
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x.squeeze(-1)


# ==================== Transformer Model ====================

class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [seq_len, batch, d_model]
        return x + self.pe[:x.size(0), :]


class TransformerClassifier(nn.Module):
    """
    Transformer 分类模型
    
    特点：
    - 多头自注意力：捕获全局依赖关系
    - 位置编码：保留序列顺序信息
    - 适合长序列建模
    """
    def __init__(self, input_size=4, d_model=128, nhead=8, 
                 num_layers=4, dim_feedforward=512, dropout=0.1, num_classes=1):
        super(TransformerClassifier, self).__init__()
        
        self.d_model = d_model
        
        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer 编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # 使用 [seq_len, batch, d_model]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 分类头
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        batch_size, seq_len, _ = x.size()
        
        # 输入投影
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        x = x * np.sqrt(self.d_model)  # 缩放
        
        # 转置为 [seq_len, batch, d_model] 以适应 Transformer
        x = x.transpose(0, 1)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer 编码
        x = self.transformer_encoder(x)  # [seq_len, batch, d_model]
        
        # 转置回 [batch, seq_len, d_model]
        x = x.transpose(0, 1)
        
        # 全局池化 + 分类
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.global_pool(x).squeeze(-1)  # [batch, d_model]
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x.squeeze(-1)


if __name__ == "__main__":
    # 测试模型
    print("Testing models...")
    
    x = torch.randn(2, 500, 4)
    
    # CNN1D
    cnn = CNN1D(input_channels=4)
    out = cnn(x)
    print(f"CNN1D output shape: {out.shape}")
    
    # LSTM
    lstm = LSTMClassifier(input_size=4, hidden_size=64)
    out = lstm(x)
    print(f"LSTM output shape: {out.shape}")
    
    # CNN-LSTM
    hybrid = CNNLSTMHybrid(input_channels=4, hidden_size=64)
    out = hybrid(x)
    print(f"CNN-LSTM output shape: {out.shape}")
    
    # TCN
    tcn = TCNClassifier(input_channels=4)
    out = tcn(x)
    print(f"TCN output shape: {out.shape}")
    
    # Transformer
    transformer = TransformerClassifier(input_size=4)
    out = transformer(x)
    print(f"Transformer output shape: {out.shape}")
    
    print("\nAll models working!")
