"""
拧紧曲线聚类分析
对 Anord.json 数据集进行聚类分析
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_curves(filepath):
    """解析曲线数据"""
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
                    curves.append({
                        'id': obj.get('id', f'curve_{i}'),
                        'report': model.get('report', 'UNKNOWN'),
                        'result_number': model.get('resultNumber', 0),
                        'points': points
                    })
        except:
            pass
    
    return curves

def extract_features(curves):
    """提取曲线特征"""
    features = []
    valid_curves = []
    
    for curve in curves:
        points = curve['points']
        if len(points) < 10:
            continue
        
        # 提取特征向量
        torque = np.array([p['torque'] for p in points])
        angle = np.array([p['angle'] for p in points])
        current = np.array([p['current'] for p in points])
        speed = np.array([p['speed'] for p in points])
        
        # 基础统计特征
        feat = [
            np.max(torque),                    # 最大扭矩
            np.min(torque),                    # 最小扭矩
            np.mean(torque),                   # 平均扭矩
            np.std(torque),                    # 扭矩标准差
            np.max(angle),                     # 最大角度
            np.mean(angle),                    # 平均角度
            np.max(current),                   # 最大电流
            np.mean(current),                  # 平均电流
            np.max(speed),                     # 最大速度
            np.mean(speed),                    # 平均速度
            torque[-1],                        # 最终扭矩
            angle[-1],                         # 最终角度
            len(points),                       # 曲线长度
            np.sum(torque),                    # 扭矩积分
            np.sum(angle),                     # 角度积分
        ]
        
        features.append(feat)
        valid_curves.append(curve)
    
    return np.array(features), valid_curves

def perform_clustering(features, n_clusters=5):
    """执行聚类分析"""
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-Means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    
    # PCA 降维用于可视化
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    
    return labels, features_pca, kmeans, scaler, pca

def visualize_clusters(features_pca, labels, curves, output_dir='output'):
    """可视化聚类结果"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. 聚类散点图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 按聚类标签着色
    scatter = axes[0].scatter(features_pca[:, 0], features_pca[:, 1], 
                              c=labels, cmap='viridis', alpha=0.6)
    axes[0].set_xlabel('PCA Component 1')
    axes[0].set_ylabel('PCA Component 2')
    axes[0].set_title('Curve Clustering (K-Means)')
    plt.colorbar(scatter, ax=axes[0], label='Cluster')
    
    # 按报告结果着色
    reports = [c['report'] for c in curves]
    colors = ['green' if r == 'OK' else 'red' for r in reports]
    axes[1].scatter(features_pca[:, 0], features_pca[:, 1], 
                    c=colors, alpha=0.6)
    axes[1].set_xlabel('PCA Component 1')
    axes[1].set_ylabel('PCA Component 2')
    axes[1].set_title('Curves by Report (Green=OK, Red=NOK)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_pca.png', dpi=150)
    print(f'Saved: {output_dir}/cluster_pca.png')
    
    # 2. 每个聚类的示例曲线
    n_clusters = len(set(labels))
    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 3*n_clusters))
    if n_clusters == 1:
        axes = [axes]
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        ax = axes[cluster_id]
        
        # 绘制该聚类的前5条曲线
        for idx in cluster_indices[:5]:
            curve = curves[idx]
            points = curve['points']
            torque = [p['torque'] for p in points]
            angle = [p['angle'] for p in points]
            ax.plot(angle, torque, alpha=0.5, label=f"{curve['report']}")
        
        ax.set_xlabel('Angle (°)')
        ax.set_ylabel('Torque (Nm)')
        ax.set_title(f'Cluster {cluster_id} (n={len(cluster_indices)})')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_curves.png', dpi=150)
    print(f'Saved: {output_dir}/cluster_curves.png')
    
    plt.close('all')

def analyze_clusters(features, labels, curves):
    """分析每个聚类的特征"""
    print("\n" + "="*60)
    print("聚类分析报告")
    print("="*60)
    
    n_clusters = len(set(labels))
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_curves = [curves[i] for i in cluster_indices]
        cluster_features = features[cluster_indices]
        
        # 统计报告结果
        ok_count = sum(1 for c in cluster_curves if c['report'] == 'OK')
        nok_count = sum(1 for c in cluster_curves if c['report'] == 'NOK')
        
        print(f"\n📊 Cluster {cluster_id}:")
        print(f"   样本数: {len(cluster_curves)}")
        print(f"   OK: {ok_count}, NOK: {nok_count}")
        print(f"   NOK比例: {nok_count/len(cluster_curves)*100:.1f}%")
        
        # 特征均值
        feature_names = ['MaxTorque', 'MinTorque', 'MeanTorque', 'StdTorque',
                        'MaxAngle', 'MeanAngle', 'MaxCurrent', 'MeanCurrent',
                        'MaxSpeed', 'MeanSpeed', 'FinalTorque', 'FinalAngle',
                        'Length', 'TorqueIntegral', 'AngleIntegral']
        
        mean_features = np.mean(cluster_features, axis=0)
        print(f"   关键特征:")
        print(f"     - 最大扭矩: {mean_features[0]:.2f} Nm")
        print(f"     - 平均扭矩: {mean_features[2]:.2f} Nm")
        print(f"     - 最终扭矩: {mean_features[10]:.2f} Nm")
        print(f"     - 最大角度: {mean_features[4]:.2f} °")
        print(f"     - 曲线长度: {mean_features[12]:.0f} points")

def main():
    print("🔩 拧紧曲线聚类分析")
    print("="*60)
    
    # 1. 加载数据
    print("\n📥 加载数据...")
    curves = parse_curves('API/Anord.json')
    print(f"   共 {len(curves)} 条曲线")
    
    # 2. 提取特征
    print("\n🔍 提取特征...")
    features, valid_curves = extract_features(curves)
    print(f"   有效曲线: {len(valid_curves)}")
    print(f"   特征维度: {features.shape[1]}")
    
    # 3. 聚类分析
    print("\n🤖 执行聚类...")
    n_clusters = 5  # 可以调整
    labels, features_pca, kmeans, scaler, pca = perform_clustering(features, n_clusters)
    print(f"   聚类数: {n_clusters}")
    
    # 4. 可视化
    print("\n📊 生成可视化...")
    visualize_clusters(features_pca, labels, valid_curves)
    
    # 5. 分析报告
    analyze_clusters(features, labels, valid_curves)
    
    # 6. 保存结果
    print("\n💾 保存结果...")
    results = {
        'total_curves': len(valid_curves),
        'n_clusters': n_clusters,
        'cluster_distribution': {i: int(np.sum(labels == i)) for i in range(n_clusters)},
        'cluster_labels': labels.tolist(),
        'curve_ids': [c['id'] for c in valid_curves],
        'curve_reports': [c['report'] for c in valid_curves]
    }
    
    with open('output/cluster_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("   Saved: output/cluster_results.json")
    
    print("\n" + "="*60)
    print("[OK] 分析完成！")
    print("="*60)

if __name__ == '__main__':
    main()
