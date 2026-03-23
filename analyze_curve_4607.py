import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 加载数据
with open('API/Anord.json', 'r', encoding='utf-8') as f:
    content = f.read()

parts = content.split('}{')
curves = []
features = []
result_numbers = []

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
                if len(points) < 10:
                    continue
                
                result_num = model.get('resultNumber', i)
                result_numbers.append(result_num)
                
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
                curves.append({
                    'resultNumber': result_num,
                    'vin': model.get('vin', 'N/A'),
                    'report': model.get('report', 'UNKNOWN'),
                    'points': points
                })
    except:
        pass

# 找到曲线 4607
target_idx = result_numbers.index(4607) if 4607 in result_numbers else None

if target_idx is not None:
    print(f"=== 曲线 4607 分析 ===")
    print(f"索引: {target_idx}")
    print(f"ResultNumber: {curves[target_idx]['resultNumber']}")
    print(f"VIN: {curves[target_idx]['vin']}")
    print(f"Report: {curves[target_idx]['report']}")
    
    # 加载聚类结果
    with open('output/cluster_results.json', 'r') as f:
        cluster_results = json.load(f)
    
    labels = np.array(cluster_results['cluster_labels'])
    assigned_cluster = labels[target_idx]
    
    print(f"\n分配的聚类: Cluster {assigned_cluster}")
    
    # 标准化特征
    features = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # PCA 降维
    pca_2d = PCA(n_components=2)
    features_pca_2d = pca_2d.fit_transform(features_scaled)
    
    pca_3d = PCA(n_components=3)
    features_pca_3d = pca_3d.fit_transform(features_scaled)
    
    print(f"\n=== 15维特征向量 ===")
    feature_names = ['MaxTorque', 'MinTorque', 'MeanTorque', 'StdTorque',
                   'MaxAngle', 'MeanAngle', 'MaxCurrent', 'MeanCurrent',
                   'MaxSpeed', 'MeanSpeed', 'FinalTorque', 'FinalAngle',
                   'Length', 'TorqueIntegral', 'AngleIntegral']
    for name, val in zip(feature_names, features[target_idx]):
        print(f"  {name}: {val:.4f}")
    
    print(f"\n=== 标准化特征 (前5维) ===")
    for name, val in zip(feature_names[:5], features_scaled[target_idx][:5]):
        print(f"  {name}: {val:.4f}")
    
    print(f"\n=== PCA 坐标 ===")
    print(f"  2D PCA: ({features_pca_2d[target_idx][0]:.4f}, {features_pca_2d[target_idx][1]:.4f})")
    print(f"  3D PCA: ({features_pca_3d[target_idx][0]:.4f}, {features_pca_3d[target_idx][1]:.4f}, {features_pca_3d[target_idx][2]:.4f})")
    
    # 计算到各聚类中心的距离
    print(f"\n=== 到各聚类中心的距离 (15维标准化空间) ===")
    for cluster_id in range(5):
        cluster_mask = labels == cluster_id
        cluster_features = features_scaled[cluster_mask]
        cluster_center = np.mean(cluster_features, axis=0)
        distance = np.linalg.norm(features_scaled[target_idx] - cluster_center)
        print(f"  Cluster {cluster_id}: {distance:.4f}")
    
    # 计算到各聚类中心的距离 (2D PCA空间)
    print(f"\n=== 到各聚类中心的距离 (2D PCA空间) ===")
    for cluster_id in range(5):
        cluster_mask = labels == cluster_id
        cluster_pca = features_pca_2d[cluster_mask]
        cluster_center = np.mean(cluster_pca, axis=0)
        distance = np.linalg.norm(features_pca_2d[target_idx] - cluster_center)
        print(f"  Cluster {cluster_id}: {distance:.4f}")
    
    # 计算到各聚类中心的距离 (3D PCA空间)
    print(f"\n=== 到各聚类中心的距离 (3D PCA空间) ===")
    for cluster_id in range(5):
        cluster_mask = labels == cluster_id
        cluster_pca = features_pca_3d[cluster_mask]
        cluster_center = np.mean(cluster_pca, axis=0)
        distance = np.linalg.norm(features_pca_3d[target_idx] - cluster_center)
        print(f"  Cluster {cluster_id}: {distance:.4f}")
    
    # 找出最近的聚类
    distances_15d = []
    distances_2d = []
    distances_3d = []
    for cluster_id in range(5):
        cluster_mask = labels == cluster_id
        cluster_center_15d = np.mean(features_scaled[cluster_mask], axis=0)
        cluster_center_2d = np.mean(features_pca_2d[cluster_mask], axis=0)
        cluster_center_3d = np.mean(features_pca_3d[cluster_mask], axis=0)
        distances_15d.append(np.linalg.norm(features_scaled[target_idx] - cluster_center_15d))
        distances_2d.append(np.linalg.norm(features_pca_2d[target_idx] - cluster_center_2d))
        distances_3d.append(np.linalg.norm(features_pca_3d[target_idx] - cluster_center_3d))
    
    nearest_15d = np.argmin(distances_15d)
    nearest_2d = np.argmin(distances_2d)
    nearest_3d = np.argmin(distances_3d)
    
    print(f"\n=== 最近聚类分析 ===")
    print(f"  在15维空间中最近的聚类: Cluster {nearest_15d} (距离: {distances_15d[nearest_15d]:.4f})")
    print(f"  在2D PCA空间中最近的聚类: Cluster {nearest_2d} (距离: {distances_2d[nearest_2d]:.4f})")
    print(f"  在3D PCA空间中最近的聚类: Cluster {nearest_3d} (距离: {distances_3d[nearest_3d]:.4f})")
    print(f"  实际分配的聚类: Cluster {assigned_cluster}")
    
    if nearest_2d != assigned_cluster or nearest_3d != assigned_cluster:
        print(f"\n⚠️  注意: PCA降维后显示最近的是 Cluster {nearest_2d}(2D) / {nearest_3d}(3D)，但实际分配的是 Cluster {assigned_cluster}")
        print(f"   这是因为 K-Means 在15维空间中聚类，而PCA降维丢失了信息！")
else:
    print("未找到曲线 4607")
