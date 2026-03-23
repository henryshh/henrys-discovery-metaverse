"""DBSCAN 密度聚类"""

import numpy as np
from typing import List

class DBSCANClustering:
    """DBSCAN 密度聚类"""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
    
    def _region_query(self, X, point_idx):
        """查询邻域"""
        neighbors = []
        for i in range(len(X)):
            if np.linalg.norm(X[point_idx] - X[i]) < self.eps:
                neighbors.append(i)
        return neighbors
    
    def fit(self, features: np.ndarray):
        """训练 DBSCAN"""
        n_samples = len(features)
        labels = np.full(n_samples, -1)  # -1 表示噪声
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0
        
        for i in range(n_samples):
            if visited[i]:
                continue
            
            visited[i] = True
            neighbors = self._region_query(features, i)
            
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # 标记为噪声
            else:
                # 扩展聚类
                labels[i] = cluster_id
                seeds = neighbors.copy()
                j = 0
                while j < len(seeds):
                    if not visited[seeds[j]]:
                        visited[seeds[j]] = True
                        new_neighbors = self._region_query(features, seeds[j])
                        if len(new_neighbors) >= self.min_samples:
                            seeds.extend(new_neighbors)
                    if labels[seeds[j]] == -1:
                        labels[seeds[j]] = cluster_id
                    j += 1
                cluster_id += 1
        
        self.labels_ = labels
        return labels
