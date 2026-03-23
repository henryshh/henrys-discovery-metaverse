"""降维可视化"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def plot_tsne(features, labels, title="t-SNE 可视化"):
    """t-SNE 降维可视化"""
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        features_2d = tsne.fit_transform(features)
        
        fig = px.scatter(
            x=features_2d[:, 0], y=features_2d[:, 1],
            color=labels.astype(str),
            title=title,
            labels={'color': '聚类'}
        )
        return fig
    except Exception as e:
        print(f"t-SNE 失败: {e}")
        return None

def plot_pca(features, labels, title="PCA 可视化"):
    """PCA 降维可视化"""
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        fig = px.scatter(
            x=features_2d[:, 0], y=features_2d[:, 1],
            color=labels.astype(str),
            title=title,
            labels={'color': '聚类'}
        )
        return fig
    except Exception as e:
        print(f"PCA 失败: {e}")
        return None
