"""DTW聚类算法"""

import numpy as np
from typing import List

def dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """计算DTW距离"""
    n, m = len(s1), len(s2)
    dtw = np.inf * np.ones((n+1, m+1))
    dtw[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    return dtw[n, m]

class DTWClustering:
    """DTW + K-Medoids聚类"""
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.labels_ = None
        self.centroids_ = None
    
    def fit(self, curves: List[np.ndarray]):
        """训练聚类"""
        n = len(curves)
        
        # 计算DTW距离矩阵
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = dtw_distance(curves[i], curves[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        # 简单的K-Medoids实现
        # 随机选择中心点
        np.random.seed(42)
        medoids = np.random.choice(n, self.n_clusters, replace=False)
        
        for _ in range(100):  # 最大迭代次数
            # 分配点到最近的中心
            labels = np.argmin(dist_matrix[:, medoids], axis=1)
            
            # 更新中心点
            new_medoids = []
            for k in range(self.n_clusters):
                cluster_points = np.where(labels == k)[0]
                if len(cluster_points) > 0:
                    # 选择使总距离最小的点作为新中心
                    sub_dist = dist_matrix[np.ix_(cluster_points, cluster_points)]
                    total_dist = sub_dist.sum(axis=1)
                    new_medoid = cluster_points[np.argmin(total_dist)]
                    new_medoids.append(new_medoid)
                else:
                    new_medoids.append(medoids[k])
            
            if np.array_equal(medoids, new_medoids):
                break
            medoids = np.array(new_medoids)
        
        self.labels_ = labels
        self.centroids_ = [curves[i] for i in medoids]
        return self.labels_
    
    def predict(self, curves: List[np.ndarray]) -> np.ndarray:
        """预测新数据"""
        labels = []
        for curve in curves:
            dists = [dtw_distance(curve, c) for c in self.centroids_]
            labels.append(np.argmin(dists))
        return np.array(labels)
