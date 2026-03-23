"""3D 可视化"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_3d_scatter(features, labels, title="3D 聚类可视化"):
    """3D 散点图"""
    # 如果特征维度不足3，用PCA扩展到3D
    if features.shape[1] < 3:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        features_3d = pca.fit_transform(features)
    else:
        features_3d = features[:, :3]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=features_3d[:, 0],
        y=features_3d[:, 1],
        z=features_3d[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=labels,
            colorscale='Viridis',
            opacity=0.8
        ),
        text=[f'聚类 {l}' for l in labels],
        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>%{text}'
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=600
    )
    
    return fig

def plot_3d_curves(curves, labels, title="3D 曲线可视化"):
    """3D 曲线图"""
    fig = go.Figure()
    
    unique_labels = sorted(set(labels))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    for idx, label in enumerate(unique_labels):
        cluster_curves = [curves[i] for i in range(len(curves)) if labels[i] == label]
        if not cluster_curves:
            continue
        
        # 只显示前5条曲线避免混乱
        for i, curve in enumerate(cluster_curves[:5]):
            x = np.arange(len(curve))
            y = curve
            z = [label] * len(curve)  # 用标签作为z轴
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                name=f'聚类 {label} - 曲线 {i+1}',
                line=dict(color=colors[idx % len(colors)], width=2),
                opacity=0.7
            ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='采样点',
            yaxis_title='扭矩值',
            zaxis_title='聚类'
        ),
        width=900,
        height=700
    )
    
    return fig

def plot_interactive_heatmap(distance_matrix, title="距离热力图"):
    """交互式热力图"""
    fig = go.Figure(data=go.Heatmap(
        z=distance_matrix,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='样本 %{x} vs 样本 %{y}<br>距离: %{z:.2f}'
    ))
    
    fig.update_layout(
        title=title,
        width=700,
        height=700
    )
    
    return fig
