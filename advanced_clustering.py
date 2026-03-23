"""
高级聚类分析方法
包含：层次聚类、K-Shape、DBSCAN、GMM、谱聚类
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_curves(filepath='API/Anord.json'):
    """加载曲线数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    parts = content.split('}{')
    curves = []
    
    for i, part in enumerate(parts):
        if i == 0:
            json_str = part + '}'
        elif i == len(parts) - 1:
            json_str = '{' + part
        else:
            json_str = '{' + part + '}'
        
        try:
            obj = json.loads(json_str)
            if 'model' in obj and 'results' in obj['model']:
                model = obj['model']
                result = model['results'][0] if model['results'] else None
                if result and 'curves' in result:
                    points = result['curves'][0]['data']['points']
                    if len(points) >= 10:
                        curves.append({
                            'id': i,
                            'resultNumber': model.get('resultNumber', i),
                            'report': model.get('report', 'UNKNOWN'),
                            'points': points
                        })
        except:
            pass
    
    return curves


def extract_features(curves):
    """提取特征向量"""
    features = []
    valid_curves = []
    
    for curve in curves:
        points = curve['points']
        if len(points) < 10:
            continue
        
        torque = np.array([p['torque'] for p in points])
        angle = np.array([p['angle'] for p in points])
        current = np.array([p['current'] for p in points])
        speed = np.array([p['speed'] for p in points])
        
        feat = [
            np.max(torque), np.min(torque), np.mean(torque), np.std(torque),
            np.max(angle), np.mean(angle),
            np.max(current), np.mean(current),
            np.max(speed), np.mean(speed),
            torque[-1], angle[-1],
            len(points), np.sum(torque), np.sum(angle)
        ]
        
        features.append(feat)
        valid_curves.append(curve)
    
    return np.array(features), valid_curves


def dtw_distance(s1, s2):
    """计算DTW距离（简化版）"""
    n, m = len(s1), len(s2)
    dtw = np.inf * np.ones((n+1, m+1))
    dtw[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    return dtw[n, m]


def compute_dtw_matrix(curves, max_curves=50):
    """计算DTW距离矩阵（采样，统一角度轴对齐）"""
    print(f"计算DTW距离矩阵（采样{max_curves}条曲线，Angle-aligned）...")
    
    # 采样
    indices = np.random.choice(len(curves), min(max_curves, len(curves)), replace=False)
    sample_curves_data = [curves[i] for i in indices]
    
    # 使用新逻辑进行角度对齐
    # 由于 align_curves_by_angle 是在 TighteningDataLoader 中的，
    # 这里的 curves 是从 load_curves 返回的简单 dict 列表。
    # 我们在这里直接实现对齐逻辑或者重构 load_curves。
    
    torque_series = []
    n_points = 100
    
    # 计算平均最大角度作为参考网格
    max_angles = []
    for c in sample_curves_data:
        points = c['points']
        max_angles.append(points[-1]['angle'])
    common_max_angle = np.mean(max_angles)
    angle_grid = np.linspace(0, common_max_angle, n_points)
    
    from scipy.interpolate import interp1d
    
    for c in sample_curves_data:
        points = c['points']
        torque = np.array([p['torque'] for p in points])
        angle = np.array([p['angle'] for p in points])
        
        f = interp1d(angle, torque, kind='linear', fill_value="extrapolate")
        torque_aligned = f(angle_grid)
        torque_series.append(torque_aligned)
    
    n = len(torque_series)
    dtw_matrix = np.zeros((n, n))
    
    from dtw_cluster_full import fast_dtw
    
    for i in range(n):
        for j in range(i+1, n):
            dist = fast_dtw(torque_series[i], torque_series[j])
            dtw_matrix[i, j] = dist
            dtw_matrix[j, i] = dist
        if i % 10 == 0:
            print(f"  进度: {i}/{n}")
    
    return dtw_matrix, sample_curves_data, indices


def hierarchical_clustering_dtw(curves, output_dir='output'):
    """1. 层次聚类 + DTW"""
    print("\n" + "="*60)
    print("1. 层次聚类 + DTW")
    print("="*60)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # 计算DTW距离矩阵
    dtw_matrix, sample_curves, indices = compute_dtw_matrix(curves, max_curves=50)
    
    # 层次聚类
    Z = linkage(squareform(dtw_matrix), method='ward')
    
    # 绘制树状图
    plt.figure(figsize=(15, 8))
    dendrogram(Z, labels=[f"{sample_curves[i]['resultNumber']}" for i in range(len(sample_curves))])
    plt.title('层次聚类树状图 (DTW距离 + Ward方法)')
    plt.xlabel('曲线 (ResultNumber)')
    plt.ylabel('距离')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hierarchical_dendrogram.png', dpi=150)
    print(f"  保存: {output_dir}/hierarchical_dendrogram.png")
    plt.close()
    
    # 不同层级的聚类
    for n_clusters in [3, 5, 7]:
        labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        print(f"\n  聚类数={n_clusters}:")
        for i in range(1, n_clusters+1):
            count = np.sum(labels == i)
            print(f"    Cluster {i}: {count}条")
        
        # 保存结果
        result = {
            'method': 'Hierarchical',
            'n_clusters': n_clusters,
            'labels': labels.tolist(),
            'resultNumbers': [c['resultNumber'] for c in sample_curves]
        }
        
        with open(f'{output_dir}/hierarchical_result_{n_clusters}.json', 'w') as f:
            json.dump(result, f, indent=2)
    
    return Z


def kshape_clustering(curves, output_dir='output'):
    """2. K-Shape 聚类"""
    print("\n" + "="*60)
    print("2. K-Shape 聚类 (时间序列专用)")
    print("="*60)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        from tslearn.clustering import KShape
        from tslearn.preprocessing import TimeSeriesScalerMeanVariance
        
        # 准备数据（标准化长度）
        max_len = 100
        X = []
        valid_curves = []
        
        for curve in curves[:100]:  # 限制数量
            points = curve['points']
            if len(points) < 10:
                continue
            
            torque = np.array([p['torque'] for p in points])
            # 插值到统一长度
            if len(torque) != max_len:
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, len(torque))
                x_new = np.linspace(0, 1, max_len)
                f = interp1d(x_old, torque, kind='linear')
                torque = f(x_new)
            
            X.append(torque)
            valid_curves.append(curve)
        
        X = np.array(X)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # 标准化
        X = TimeSeriesScalerMeanVariance().fit_transform(X)
        
        # K-Shape聚类
        for n_clusters in [3, 5]:
            print(f"\n  聚类数={n_clusters}:")
            ks = KShape(n_clusters=n_clusters, max_iter=100, random_state=42)
            labels = ks.fit_predict(X)
            
            for i in range(n_clusters):
                count = np.sum(labels == i)
                print(f"    Cluster {i}: {count}条")
            
            # 绘制聚类中心
            plt.figure(figsize=(12, 6))
            for i in range(n_clusters):
                plt.subplot(1, n_clusters, i+1)
                plt.plot(ks.cluster_centers_[i].ravel(), 'r-', linewidth=2)
                plt.title(f'Cluster {i} 中心')
                plt.xlabel('时间')
                plt.ylabel('扭矩')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/kshape_centers_{n_clusters}.png', dpi=150)
            print(f"  保存: {output_dir}/kshape_centers_{n_clusters}.png")
            plt.close()
        
        return True
    except ImportError:
        print("  需要安装 tslearn: pip install tslearn")
        return False


def dbscan_clustering(curves, output_dir='output'):
    """3. DBSCAN 聚类 (自动识别异常)"""
    print("\n" + "="*60)
    print("3. DBSCAN 聚类 (自动识别异常)")
    print("="*60)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # 提取特征
    features, valid_curves = extract_features(curves)
    
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 尝试不同的 eps 值
    eps_values = [0.3, 0.5, 0.8, 1.0, 1.5]
    
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(features_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"\n  eps={eps}:")
        print(f"    发现聚类数: {n_clusters}")
        print(f"    异常点数量: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
        
        for i in range(n_clusters):
            count = np.sum(labels == i)
            print(f"    Cluster {i}: {count}条")
        
        if n_clusters > 0 and n_clusters <= 10:
            # 保存结果
            result = {
                'method': 'DBSCAN',
                'eps': eps,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'labels': labels.tolist(),
                'noise_ratio': n_noise / len(labels)
            }
            
            with open(f'{output_dir}/dbscan_result_eps{eps}.json', 'w') as f:
                json.dump(result, f, indent=2)
            
            # 可视化（PCA降维）
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features_scaled)
            
            plt.figure(figsize=(10, 8))
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters + 1))
            
            for i in range(n_clusters):
                mask = labels == i
                plt.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                           c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=50)
            
            # 异常点
            if n_noise > 0:
                mask_noise = labels == -1
                plt.scatter(features_pca[mask_noise, 0], features_pca[mask_noise, 1], 
                           c='black', marker='x', label='异常点', alpha=0.8, s=100)
            
            plt.title(f'DBSCAN 聚类结果 (eps={eps}, {n_clusters}个聚类, {n_noise}个异常点)')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{output_dir}/dbscan_eps{eps}.png', dpi=150)
            print(f"    保存: {output_dir}/dbscan_eps{eps}.png")
            plt.close()
    
    return True


def gmm_clustering(curves, output_dir='output'):
    """4. GMM 概率聚类"""
    print("\n" + "="*60)
    print("4. GMM 概率聚类")
    print("="*60)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # 提取特征
    features, valid_curves = extract_features(curves)
    
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 尝试不同的聚类数
    for n_components in [3, 5, 7]:
        print(f"\n  聚类数={n_components}:")
        
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        labels = gmm.fit_predict(features_scaled)
        probs = gmm.predict_proba(features_scaled)
        
        # 统计
        for i in range(n_components):
            count = np.sum(labels == i)
            avg_prob = np.mean(probs[labels == i, i]) if count > 0 else 0
            print(f"    Cluster {i}: {count}条, 平均概率={avg_prob:.3f}")
        
        # 找出高置信度和低置信度的样本
        max_probs = np.max(probs, axis=1)
        high_conf = np.sum(max_probs > 0.9)
        low_conf = np.sum(max_probs < 0.5)
        
        print(f"    高置信度(>0.9): {high_conf}条 ({high_conf/len(labels)*100:.1f}%)")
        print(f"    低置信度(<0.5): {low_conf}条 ({low_conf/len(labels)*100:.1f}%)")
        
        # 保存结果
        result = {
            'method': 'GMM',
            'n_components': n_components,
            'labels': labels.tolist(),
            'probabilities': probs.tolist(),
            'aic': gmm.aic(features_scaled),
            'bic': gmm.bic(features_scaled)
        }
        
        with open(f'{output_dir}/gmm_result_{n_components}.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        # 可视化（PCA降维）
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        plt.figure(figsize=(12, 5))
        
        # 左图：聚类结果
        plt.subplot(1, 2, 1)
        colors = plt.cm.tab10(np.linspace(0, 1, n_components))
        for i in range(n_components):
            mask = labels == i
            plt.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                       c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=50)
        plt.title(f'GMM 聚类结果 (n={n_components})')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        
        # 右图：置信度
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], 
                            c=max_probs, cmap='RdYlGn', alpha=0.6, s=50)
        plt.colorbar(scatter, label='最大概率')
        plt.title('聚类置信度 (颜色越深=置信度越高)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/gmm_{n_components}.png', dpi=150)
        print(f"    保存: {output_dir}/gmm_{n_components}.png")
        plt.close()
    
    return True


def spectral_clustering(curves, output_dir='output'):
    """5. 谱聚类"""
    print("\n" + "="*60)
    print("5. 谱聚类")
    print("="*60)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # 提取特征
    features, valid_curves = extract_features(curves)
    
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 尝试不同的聚类数
    for n_clusters in [3, 5, 7]:
        print(f"\n  聚类数={n_clusters}:")
        
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
        labels = spectral.fit_predict(features_scaled)
        
        for i in range(n_clusters):
            count = np.sum(labels == i)
            print(f"    Cluster {i}: {count}条")
        
        # 保存结果
        result = {
            'method': 'Spectral',
            'n_clusters': n_clusters,
            'labels': labels.tolist()
        }
        
        with open(f'{output_dir}/spectral_result_{n_clusters}.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        # 可视化（PCA降维）
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        for i in range(n_clusters):
            mask = labels == i
            plt.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                       c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=50)
        
        plt.title(f'谱聚类结果 (n={n_clusters})')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/spectral_{n_clusters}.png', dpi=150)
        print(f"    保存: {output_dir}/spectral_{n_clusters}.png")
        plt.close()
    
    return True


def main():
    print("="*60)
    print("高级聚类分析")
    print("="*60)
    
    # 加载数据
    print("\n加载数据...")
    curves = load_curves()
    print(f"共 {len(curves)} 条曲线")
    
    output_dir = 'output/advanced_clustering'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. 层次聚类 + DTW
    hierarchical_clustering_dtw(curves, output_dir)
    
    # 2. K-Shape
    kshape_clustering(curves, output_dir)
    
    # 3. DBSCAN
    dbscan_clustering(curves, output_dir)
    
    # 4. GMM
    gmm_clustering(curves, output_dir)
    
    # 5. 谱聚类
    spectral_clustering(curves, output_dir)
    
    print("\n" + "="*60)
    print("分析完成！")
    print(f"结果保存在: {output_dir}/")
    print("="*60)


if __name__ == '__main__':
    main()
