"""
DTW 曲线聚类 + 3D 交互式可视化
基于动态时间规整的拧紧曲线聚类分析
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.interpolate import interp1d
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def dtw_distance(s1, s2, max_warping=0.1):
    """
    计算两条曲线的 DTW 距离
    s1, s2: 1D numpy arrays
    max_warping: 最大扭曲比例
    """
    n, m = len(s1), len(s2)
    
    # 动态规划表
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    # 允许的最大扭曲窗口
    window = max(int(max(max_warping * n, max_warping * m)), abs(n - m))
    
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window + 1)):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j],    # 插入
                                   dtw[i, j - 1],    # 删除
                                   dtw[i - 1, j - 1]) # 匹配
    
    return dtw[n, m]


def normalize_curve(curve, target_length=200):
    """
    归一化曲线：统一长度 + 归一化数值
    """
    # 提取扭矩和角度
    torque = np.array([p['torque'] for p in curve['points']])
    angle = np.array([p['angle'] for p in curve['points']])
    
    # 插值到统一长度
    if len(torque) < 10:
        return None
    
    # 使用角度作为 x 轴，扭矩作为 y 轴
    if len(torque) != target_length:
        x_old = np.linspace(0, 1, len(torque))
        x_new = np.linspace(0, 1, target_length)
        
        # 插值
        f_torque = interp1d(x_old, torque, kind='linear', fill_value='extrapolate')
        f_angle = interp1d(x_old, angle, kind='linear', fill_value='extrapolate')
        
        torque = f_torque(x_new)
        angle = f_angle(x_new)
    
    # 归一化到 [0, 1]
    torque_norm = (torque - torque.min()) / (torque.max() - torque.min() + 1e-8)
    angle_norm = (angle - angle.min()) / (angle.max() - angle.min() + 1e-8)
    
    return {
        'torque': torque_norm,
        'angle': angle_norm,
        'original': curve
    }


def compute_dtw_matrix(curves, sample_size=50):
    """
    计算 DTW 距离矩阵（采样加速）
    """
    n = min(len(curves), sample_size)
    indices = np.random.choice(len(curves), n, replace=False) if len(curves) > sample_size else range(n)
    
    sampled_curves = [curves[i] for i in indices]
    
    print(f"Computing DTW matrix for {n} curves...")
    
    # 计算距离矩阵
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # 使用角度-扭矩组合作为特征
            s1 = np.column_stack([sampled_curves[i]['angle'], sampled_curves[i]['torque']]).flatten()
            s2 = np.column_stack([sampled_curves[j]['angle'], sampled_curves[j]['torque']]).flatten()
            
            dist = dtw_distance(s1, s2)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{n}")
    
    return dist_matrix, sampled_curves, indices


def hierarchical_clustering(dist_matrix, n_clusters=5):
    """
    层次聚类
    """
    # 转换为 condensed distance matrix
    condensed = squareform(dist_matrix)
    
    # 层次聚类
    linkage_matrix = linkage(condensed, method='ward')
    
    # 分配聚类标签
    labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
    
    return labels, linkage_matrix


def create_3d_visualization(curves, labels, output_dir='output'):
    """
    创建 3D 交互式可视化（使用 matplotlib + 保存为 HTML）
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    n_clusters = len(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # 1. 3D 散点图（使用 PCA 降维到 3D）
    # 构建特征矩阵
    features = []
    for c in curves:
        feat = np.concatenate([c['angle'], c['torque']])
        features.append(feat)
    features = np.array(features)
    
    # PCA 降维到 3D（纯 numpy 实现）
    # 标准化
    features_mean = np.mean(features, axis=0)
    features_std = np.std(features, axis=0)
    features_norm = (features - features_mean) / (features_std + 1e-8)
    
    # SVD 计算 PCA
    U, S, Vt = np.linalg.svd(features_norm, full_matrices=False)
    coords_3d = U[:, :3] * S[:3]
    
    # 创建 3D 图
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        ax.scatter(coords_3d[mask, 0], coords_3d[mask, 1], coords_3d[mask, 2],
                   c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                   s=50, alpha=0.6)
    
    # 计算方差解释比例
    total_var = np.sum(S**2)
    explained_var_ratio = (S[:3]**2) / total_var
    
    ax.set_xlabel(f'PC1 ({explained_var_ratio[0]:.1%})')
    ax.set_ylabel(f'PC2 ({explained_var_ratio[1]:.1%})')
    ax.set_zlabel(f'PC3 ({explained_var_ratio[2]:.1%})')
    ax.set_title('DTW Curve Clustering - 3D PCA Visualization')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dtw_cluster_3d.png', dpi=150)
    print(f'Saved: {output_dir}/dtw_cluster_3d.png')
    plt.close()
    
    # 2. 每个聚类的代表性曲线（3D 叠加图）
    fig = plt.figure(figsize=(16, 4 * n_clusters))
    
    for cluster_id in range(n_clusters):
        ax = fig.add_subplot(n_clusters, 1, cluster_id + 1, projection='3d')
        
        cluster_indices = np.where(labels == cluster_id)[0]
        
        # 绘制该聚类的曲线
        for idx in cluster_indices[:10]:  # 最多10条
            c = curves[idx]
            angle = c['angle']
            torque = c['torque']
            
            # 用 z 轴表示曲线索引
            z = np.full_like(angle, idx)
            
            ax.plot(angle, torque, z, alpha=0.5, color=colors[cluster_id])
        
        ax.set_xlabel('Normalized Angle')
        ax.set_ylabel('Normalized Torque')
        ax.set_zlabel('Curve ID')
        ax.set_title(f'Cluster {cluster_id} (n={len(cluster_indices)})')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dtw_cluster_curves_3d.png', dpi=150)
    print(f'Saved: {output_dir}/dtw_cluster_curves_3d.png')
    plt.close()
    
    # 3. 创建交互式 HTML（使用 Plotly）
    create_interactive_html(curves, labels, coords_3d, output_dir)
    
    return coords_3d


def create_interactive_html(curves, labels, coords_3d, output_dir):
    """
    创建交互式 HTML 可视化
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        n_clusters = len(set(labels))
        colors_plotly = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # 1. 3D 散点图
        fig = go.Figure()
        
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            
            fig.add_trace(go.Scatter3d(
                x=coords_3d[mask, 0],
                y=coords_3d[mask, 1],
                z=coords_3d[mask, 2],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(
                    size=6,
                    color=colors_plotly[cluster_id % len(colors_plotly)],
                    opacity=0.7
                ),
                text=[f"ID: {curves[i]['original']['id'][:20]}...<br>"
                      f"Report: {curves[i]['original']['report']}<br>"
                      f"Result: {curves[i]['original'].get('result_number', 'N/A')}"
                      for i in np.where(mask)[0]],
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title='DTW Curve Clustering - Interactive 3D',
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3'
            ),
            width=1000,
            height=700
        )
        
        fig.write_html(f'{output_dir}/dtw_cluster_interactive.html')
        print(f'Saved: {output_dir}/dtw_cluster_interactive.html')
        
        # 2. 曲线对比图（交互式）
        fig2 = make_subplots(
            rows=n_clusters, cols=1,
            subplot_titles=[f'Cluster {i}' for i in range(n_clusters)],
            vertical_spacing=0.1
        )
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            
            for idx in cluster_indices[:5]:  # 每个聚类显示5条
                c = curves[idx]
                fig2.add_trace(
                    go.Scatter(
                        x=c['angle'],
                        y=c['torque'],
                        mode='lines',
                        name=f'Curve {idx}',
                        line=dict(color=colors_plotly[cluster_id % len(colors_plotly)], width=1),
                        opacity=0.6,
                        showlegend=False
                    ),
                    row=cluster_id + 1, col=1
                )
        
        fig2.update_layout(
            title='Curve Clusters Comparison',
            height=300 * n_clusters,
            width=900
        )
        
        fig2.write_html(f'{output_dir}/dtw_curves_comparison.html')
        print(f'Saved: {output_dir}/dtw_curves_comparison.html')
        
    except ImportError:
        print("Plotly not available, skipping interactive HTML generation")


def analyze_clusters(curves, labels):
    """
    分析聚类结果
    """
    print("\n" + "="*60)
    print("DTW 聚类分析报告")
    print("="*60)
    
    n_clusters = len(set(labels))
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_curves = [curves[i] for i in cluster_indices]
        
        # 统计报告结果
        ok_count = sum(1 for c in cluster_curves if c['original']['report'] == 'OK')
        nok_count = sum(1 for c in cluster_curves if c['original']['report'] == 'NOK')
        
        print(f"\n📊 Cluster {cluster_id}:")
        print(f"   样本数: {len(cluster_curves)}")
        print(f"   OK: {ok_count}, NOK: {nok_count}")
        if len(cluster_curves) > 0:
            print(f"   NOK比例: {nok_count/len(cluster_curves)*100:.1f}%")
        
        # 计算平均曲线特征
        avg_torque = np.mean([c['torque'] for c in cluster_curves], axis=0)
        avg_angle = np.mean([c['angle'] for c in cluster_curves], axis=0)
        
        print(f"   平均最大扭矩: {avg_torque.max():.3f} (normalized)")
        print(f"   平均最终角度: {avg_angle[-1]:.3f} (normalized)")


def parse_anord_json(filepath):
    """解析 Anord.json"""
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


def main():
    print("🔩 DTW 曲线聚类 + 3D 交互式可视化")
    print("="*60)
    
    # 1. 加载数据
    print("\n📥 加载数据...")
    raw_curves = parse_anord_json('../API/Anord.json')
    print(f"   共 {len(raw_curves)} 条原始曲线")
    
    # 2. 归一化
    print("\n🔧 归一化曲线...")
    curves = []
    for c in raw_curves:
        norm = normalize_curve(c)
        if norm:
            norm['original'] = c
            curves.append(norm)
    print(f"   有效曲线: {len(curves)}")
    
    # 3. DTW 聚类
    print("\n🤖 DTW 聚类分析...")
    dist_matrix, sampled_curves, indices = compute_dtw_matrix(curves, sample_size=50)
    
    n_clusters = 5
    labels, linkage_matrix = hierarchical_clustering(dist_matrix, n_clusters)
    print(f"   聚类数: {n_clusters}")
    
    # 4. 3D 可视化
    print("\n📊 生成 3D 可视化...")
    coords_3d = create_3d_visualization(sampled_curves, labels, 'output')
    
    # 5. 分析报告
    analyze_clusters(sampled_curves, labels)
    
    # 6. 保存结果
    print("\n💾 保存结果...")
    results = {
        'method': 'DTW + Hierarchical Clustering',
        'total_curves': len(sampled_curves),
        'n_clusters': n_clusters,
        'cluster_distribution': {int(i): int(np.sum(labels == i)) for i in range(n_clusters)},
        'sampled_indices': indices.tolist()
    }
    
    with open('output/dtw_cluster_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("   Saved: output/dtw_cluster_results.json")
    
    print("\n" + "="*60)
    print("[OK] 分析完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  - output/dtw_cluster_3d.png")
    print("  - output/dtw_cluster_curves_3d.png")
    print("  - output/dtw_cluster_interactive.html (交互式)")
    print("  - output/dtw_curves_comparison.html (交互式)")
    print("  - output/dtw_cluster_results.json")


if __name__ == '__main__':
    main()
