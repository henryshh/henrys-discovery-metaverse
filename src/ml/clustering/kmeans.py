"""K-Means特征聚类"""

import numpy as np
from typing import List

class KMeansClustering:
    """基于统计特征的K-Means聚类"""
    
    def __init__(self, n_clusters: int = 5, max_iter: int = 100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.labels_ = None
        self.centroids_ = None
    
    def extract_features(self, curves: List[np.ndarray]) -> np.ndarray:
        """提取统计特征"""
        features = []
        for curve in curves:
            f = [
                np.mean(curve),
                np.std(curve),
                np.max(curve),
                np.min(curve),
                np.percentile(curve, 25),
                np.percentile(curve, 75),
                len(curve)
            ]
            features.append(f)
        return np.array(features)
    
    def fit(self, curves: List[np.ndarray]):
        """训练聚类"""
        # 提取特征
        X = self.extract_features(curves)
        n_samples, n_features = X.shape
        
        # 标准化
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-10
        X_norm = (X - self.mean_) / self.std_
        
        # 随机初始化中心
        np.random.seed(42)
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centroids = X_norm[indices].copy()
        
        for _ in range(self.max_iter):
            # 分配点到最近的中心
            distances = np.linalg.norm(X_norm[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # 更新中心
            new_centroids = np.array([X_norm[labels == k].mean(axis=0) 
                                      if np.sum(labels == k) > 0 
                                      else centroids[k] 
                                      for k in range(self.n_clusters)])
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        self.labels_ = labels
        self.centroids_ = centroids * self.std_ + self.mean_
        return self.labels_
    
    def predict(self, curves: List[np.ndarray]) -> np.ndarray:
        """预测新数据"""
        X = self.extract_features(curves)
        X_norm = (X - self.mean_) / self.std_
        distances = np.linalg.norm(X_norm[:, np.newaxis] - self.centroids_, axis=2)
        return np.argmin(distances, axis=1)
