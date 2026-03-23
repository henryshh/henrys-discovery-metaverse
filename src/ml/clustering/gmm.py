"""GMM 高斯混合模型聚类"""

import numpy as np
from typing import List

class GMMClustering:
    """高斯混合模型聚类"""
    
    def __init__(self, n_clusters: int = 5, max_iter: int = 100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
    
    def _gaussian_pdf(self, x, mean, cov):
        """高斯概率密度函数"""
        d = len(x)
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)
        diff = x - mean
        return (1.0 / np.sqrt((2 * np.pi) ** d * det)) * np.exp(-0.5 * diff.T @ inv @ diff)
    
    def fit(self, features: np.ndarray):
        """训练 GMM"""
        n_samples, n_features = features.shape
        
        # 初始化参数
        np.random.seed(42)
        self.means_ = features[np.random.choice(n_samples, self.n_clusters, replace=False)]
        self.covariances_ = [np.eye(n_features) for _ in range(self.n_clusters)]
        self.weights_ = np.ones(self.n_clusters) / self.n_clusters
        
        for _ in range(self.max_iter):
            # E-step: 计算责任度
            resp = np.zeros((n_samples, self.n_clusters))
            for i in range(n_samples):
                for k in range(self.n_clusters):
                    resp[i, k] = self.weights_[k] * self._gaussian_pdf(features[i], self.means_[k], self.covariances_[k])
                resp[i] /= resp[i].sum() + 1e-10
            
            # M-step: 更新参数
            Nk = resp.sum(axis=0)
            for k in range(self.n_clusters):
                self.weights_[k] = Nk[k] / n_samples
                self.means_[k] = (resp[:, k:k+1] * features).sum(axis=0) / (Nk[k] + 1e-10)
                diff = features - self.means_[k]
                self.covariances_[k] = (resp[:, k:k+1] * diff).T @ diff / (Nk[k] + 1e-10)
                self.covariances_[k] += np.eye(n_features) * 1e-6  # 正则化
        
        # 预测标签
        labels = np.argmax(resp, axis=1)
        return labels
