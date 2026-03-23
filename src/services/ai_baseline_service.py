import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class ShapeAutoencoder(nn.Module):
    """用于曲线形状特征学习与异常检测的 1D-CNN 自编码器"""
    def __init__(self, input_dim=500, latent_dim=16):
        super(ShapeAutoencoder, self).__init__()
        # Encoder: 1D-CNN 提取局部特征
        self.encoder = nn.Sequential(
            nn.Unflatten(1, (1, input_dim)),
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (input_dim // 4), latent_dim),
            nn.ReLU()
        )
        # Decoder: 重构曲线
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * (input_dim // 4)),
            nn.ReLU(),
            nn.Unflatten(1, (32, input_dim // 4)),
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=2, stride=2),
            nn.Flatten(1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class AIBaselineService:
    """管理每个数据 Silo 的 AI 基准模型与增量训练"""
    def __init__(self, model_dir: str = "models/baselines"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_model_path(self, dataset_id: str) -> Path:
        return self.model_dir / f"ae_{dataset_id}.pt"

    def train_baseline(self, dataset_id: str, curves: List[np.ndarray], epochs=50):
        """训练或微调特定数据集的形状基准"""
        if not curves: return
        
        # 预处理：插值到固定长度 (500)
        from scipy.interpolate import interp1d
        X = []
        for c in curves:
            if len(c) == 500:
                X.append(c)
            else:
                f = interp1d(np.linspace(0, 1, len(c)), c, kind='linear')
                X.append(f(np.linspace(0, 1, 500)))
        
        X = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        model = ShapeAutoencoder(input_dim=500).to(self.device)
        
        # 加载现有权重（如果是增量训练）
        path = self.get_model_path(dataset_id)
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=self.device))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, X)
            loss.backward()
            optimizer.step()
            
        torch.save(model.state_dict(), path)
        return loss.item()

    def calculate_anomaly_score(self, dataset_id: str, curve: np.ndarray) -> float:
        """计算单条曲线的形状异常得分（重构误差）"""
        path = self.get_model_path(dataset_id)
        if not path.exists():
            return 0.0 # 尚无基准模型
            
        model = ShapeAutoencoder(input_dim=500).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        
        # 插值
        from scipy.interpolate import interp1d
        if len(curve) != 500:
            f = interp1d(np.linspace(0, 1, len(curve)), curve, kind='linear')
            x_input = f(np.linspace(0, 1, 500))
        else:
            x_input = curve
            
        x_input = torch.tensor(x_input[np.newaxis, :], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            reconstructed = model(x_input)
            loss = nn.functional.mse_loss(reconstructed, x_input)
            
        return float(loss.cpu().item())

    def calculate_silo_envelope(self, curves: List[np.ndarray], target_len: int = 500) -> Dict[str, np.ndarray]:
        """
        Pillar 3: 计算数据集的统计包络线 (Statistical Tunnel)
        返回包含 mean, std, upper, lower 的字典
        """
        if not curves: return {}
        
        from scipy.interpolate import interp1d
        X = []
        for c in curves:
            if len(c) == target_len:
                X.append(c)
            else:
                f = interp1d(np.linspace(0, 1, len(c)), c, kind='linear', fill_value="extrapolate")
                X.append(f(np.linspace(0, 1, target_len)))
        
        X = np.array(X)
        mean_curve = np.mean(X, axis=0)
        std_curve = np.std(X, axis=0)
        
        return {
            "mean": mean_curve.astype(float),
            "std": std_curve.astype(float),
            "upper": (mean_curve + 3 * std_curve).astype(float),
            "lower": (mean_curve - 3 * std_curve).astype(float)
        }
