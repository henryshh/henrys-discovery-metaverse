"""
DTW 曲线聚类 - 全量数据优化版
由码农实现：并行计算 + 缓存机制
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.interpolate import interp1d
from pathlib import Path
import pickle
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def fast_dtw(s1, s2, max_warp=0.1):
    """优化的 DTW 距离计算"""
    n, m = len(s1), len(s2)
    if n < 2 or m < 2:
        return float('inf')
    
    # 使用 Sakoe-Chiba 带宽限制加速
    window = max(int(max_warp * max(n, m)), abs(n - m))
    
    # 初始化
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)
        
        for j in range(j_start, j_end):
            cost = abs(s1[i-1] - s2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    return dtw[n, m]


from src.data_loader import TighteningDataLoader


def compute_dtw_pair(args):
    """计算一对曲线的 DTW 距离（用于并行）"""
    i, j, curve_i, curve_j = args
    
    # 对齐后的数据已经是单一通道 (Torque aligned by Angle)
    dist = fast_dtw(curve_i, curve_j)
    return i, j, dist


def compute_dtw_matrix_parallel(curves, n_workers=4, cache_file='output/dtw_cache.pkl'):
    """并行计算 DTW 距离矩阵，带缓存"""
    n = len(curves)
    
    # 检查缓存
    if Path(cache_file).exists():
        print(f"Loading DTW matrix from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"Computing DTW matrix for {n} curves (parallel={n_workers})...")
    
    # 准备任务列表
    tasks = []
    for i in range(n):
        for j in range(i + 1, n):
            tasks.append((i, j, curves[i], curves[j]))
    
    # 并行计算
    dist_matrix = np.zeros((n, n))
    completed = 0
    total = len(tasks)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(compute_dtw_pair, task): task for task in tasks}
        
        for future in as_completed(futures):
            i, j, dist = future.result()
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
            completed += 1
            if completed % 100 == 0:
                print(f"  Progress: {completed}/{total} ({completed/total*100:.1f}%)")
    
    # 保存缓存
    Path(cache_file).parent.mkdir(exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(dist_matrix, f)
    print(f"Saved DTW matrix to cache: {cache_file}")
    
    return dist_matrix


def main():
    print("="*60)
    print("DTW 曲线聚类 - 全量数据优化版 (Stage 2)")
    print("执行者: Antigravity 🌌")
    print("="*60)
    
    # 1. 加载数据
    data_path = 'API/demeter_result_2026-03-19-03-31-54.json'
    print(f"\n[1/4] 使用 TighteningDataLoader 加载数据: {data_path}")
    
    loader = TighteningDataLoader(data_path)
    loader.load().extract_curves(target_length=None) 
    
    # ✅ Torque-Angle Coordination: 统一角度坐标系对齐
    curves, angle_grid, metadata = loader.align_curves_by_angle(n_points=200)
    
    print(f"   有效曲线: {len(curves)} 条")
    print(f"   对齐点数: {len(angle_grid)} (Angle-based)")
    
    # 3. DTW 聚类（全量 + 并行 + 缓存）
    print("\n[3/4] DTW 聚类分析（全量数据）...")
    dist_matrix = compute_dtw_matrix_parallel(curves, n_workers=4)
    
    # 层次聚类
    from scipy.cluster.hierarchy import linkage, fcluster
    condensed = squareform(dist_matrix)
    linkage_matrix = linkage(condensed, method='ward')
    
    n_clusters = 5
    labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
    print(f"   聚类数: {n_clusters}")
    
    # 4. 分析结果
    print("\n[4/4] 生成分析报告...")
    print("\n" + "="*60)
    print("聚类结果")
    print("="*60)
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_curves = curves[cluster_indices]
        cluster_metadata = [metadata[i] for i in cluster_indices]
        
        ok_count = sum(1 for m in cluster_metadata if m['report'] == 'OK')
        nok_count = sum(1 for m in cluster_metadata if m['report'] == 'NOK')
        
        print(f"\nCluster {cluster_id}: {len(cluster_curves)} 条")
        print(f"  OK: {ok_count}, NOK: {nok_count}")
        
        # 计算 Medoid (代表曲线)
        # 在该簇内，寻找与其他所有曲线距离之和最小的曲线
        cluster_dist_submatrix = dist_matrix[cluster_indices][:, cluster_indices]
        dist_sums = np.sum(cluster_dist_submatrix, axis=1)
        medoid_idx_in_cluster = np.argmin(dist_sums)
        medoid_global_idx = cluster_indices[medoid_idx_in_cluster]
        medoid_result_number = metadata[medoid_global_idx]['resultNumber']
        
        print(f"  代表曲线 (Medoid): {medoid_result_number}")
        
        # 特征统计 (Channel 0 is torque)
        avg_torque = np.mean(cluster_curves[:, :, 0], axis=0)
        print(f"  平均最大扭矩: {avg_torque.max():.3f}")
        
        # 将 medoid 信息存入一个字典方便后面保存
        if 'cluster_medoids' not in locals():
            cluster_medoids = {}
        cluster_medoids[int(cluster_id)] = medoid_result_number
    
    # 保存结果
    print("\n" + "="*60)
    print("保存结果...")
    # 保存结果 (关联 resultNumber)
    curve_results = []
    for i, m in enumerate(metadata):
        curve_results.append({
            'resultNumber': m['resultNumber'],
            'label': int(labels[i]),
            'report': m['report'],
            'vin': m['vin']
        })
    
    results = {
        'total_curves': len(curves),
        'n_clusters': n_clusters,
        'cluster_medoids': cluster_medoids,
        'curve_results': curve_results,
        'dist_matrix_shape': dist_matrix.shape
    }
    
    with open('output/dtw_full_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("  Saved: output/dtw_full_results.json")
    print("  Saved: output/dtw_cache.pkl (距离矩阵缓存)")
    
    print("\n" + "="*60)
    print("[OK] 码农任务完成！")
    print("="*60)


if __name__ == '__main__':
    main()
