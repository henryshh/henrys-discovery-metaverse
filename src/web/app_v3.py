"""拧紧曲线聚类分析系统 - Web界面 v3（完整版）"""

import streamlit as st
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="拧紧曲线聚类分析系统", page_icon="🔧", layout="wide")

# 初始化 session state
if 'projects' not in st.session_state:
    st.session_state.projects = []
if 'current_project' not in st.session_state:
    st.session_state.current_project = None
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = {}

# 导入聚类算法
try:
    from ml.clustering import DTWClustering, KMeansClustering
    from ml.clustering.gmm import GMMClustering
    from ml.clustering.dbscan import DBSCANClustering
    CLUSTERING_READY = True
except Exception as e:
    CLUSTERING_READY = False
    st.error(f"聚类算法导入失败: {e}")

st.title("🔧 拧紧曲线聚类分析系统 v3")

# 侧边栏导航
page = st.sidebar.radio("选择功能", [
    "🏠 首页",
    "📁 工程管理",
    "📊 数据集导入",
    "🔍 聚类分析",
    "📈 可视化"
])

# ============ 首页 ============
if page == "🏠 首页":
    st.header("欢迎使用")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("工程数量", len(st.session_state.projects))
    with col2:
        st.metric("聚类算法", "4" if CLUSTERING_READY else "0")
    with col3:
        st.metric("数据集", len(st.session_state.datasets))
    
    st.write("### 功能特性")
    st.write("- 📁 多工程管理")
    st.write("- 📊 数据集导入与预处理")
    st.write("- 🔍 4种聚类算法（DTW、K-Means、GMM、DBSCAN）")
    st.write("- 📈 丰富的可视化")

# ============ 工程管理 ============
elif page == "📁 工程管理":
    st.header("工程管理")
    
    with st.expander("➕ 创建新工程"):
        name = st.text_input("工程名称")
        desc = st.text_area("工程描述")
        if st.button("创建工程", type="primary"):
            if name:
                proj_id = f"proj_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                st.session_state.projects.append({
                    'id': proj_id, 'name': name, 'description': desc,
                    'created_at': datetime.now().isoformat()
                })
                st.success(f"✅ 工程 '{name}' 创建成功！")
    
    st.subheader("工程列表")
    for proj in st.session_state.projects:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{proj['name']}** - {proj.get('description', '无')}")
        with col2:
            if st.button("选择", key=f"sel_{proj['id']}"):
                st.session_state.current_project = proj
                st.success(f"已选择: {proj['name']}")

# ============ 数据集导入 ============
elif page == "📊 数据集导入":
    st.header("数据集导入")
    
    if not st.session_state.current_project:
        st.warning("请先选择一个工程")
    else:
        st.write(f"当前工程: **{st.session_state.current_project['name']}**")
        
        uploaded = st.file_uploader("上传 JSON 文件", type=['json'])
        if uploaded:
            try:
                data = json.loads(uploaded.read().decode('utf-8'))
                total = len(data)
                ok = sum(1 for r in data if r.get('model', {}).get('report') == 'OK')
                nok = total - ok
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("总记录", total)
                col2.metric("OK", ok)
                col3.metric("NOK", nok)
                col4.metric("OK率", f"{ok/total*100:.1f}%" if total else "0%")
                
                if st.button("保存到工程", type="primary"):
                    ds_id = f"ds_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    st.session_state.datasets[ds_id] = {
                        'id': ds_id, 'name': uploaded.name, 'data': data,
                        'stats': {'total': total, 'ok': ok, 'nok': nok}
                    }
                    st.success("✅ 数据集保存成功！")
            except Exception as e:
                st.error(f"数据处理失败: {e}")

# ============ 聚类分析 ============
elif page == "🔍 聚类分析":
    st.header("聚类分析")
    
    if not st.session_state.datasets:
        st.warning("请先导入数据集")
    elif not CLUSTERING_READY:
        st.error("聚类算法未就绪")
    else:
        ds_id = st.selectbox("选择数据集", options=list(st.session_state.datasets.keys()),
                            format_func=lambda x: st.session_state.datasets[x]['name'])
        
        if ds_id:
            data = st.session_state.datasets[ds_id]['data']
            
            # 提取曲线
            curves = []
            for r in data:
                pts = r.get('model', {}).get('results', [{}])[0].get('curves', [{}])[0].get('data', {}).get('points', [])
                if pts:
                    curves.append(np.array([p.get('torque', 0) for p in pts]))
            
            st.write(f"提取到 {len(curves)} 条曲线")
            
            # 选择算法
            algo = st.selectbox("算法", ["DTW + K-Medoids", "K-Means", "GMM", "DBSCAN"])
            n_cls = st.slider("聚类数", 2, 10, 5)
            
            if st.button("运行聚类", type="primary"):
                prog = st.progress(0)
                status = st.empty()
                
                try:
                    status.text("🔄 初始化...")
                    prog.progress(10)
                    
                    if algo == "DTW + K-Medoids":
                        clusterer = DTWClustering(n_clusters=n_cls)
                    elif algo == "K-Means":
                        clusterer = KMeansClustering(n_clusters=n_cls)
                    elif algo == "GMM":
                        # GMM 需要特征
                        from sklearn.decomposition import PCA
                        features = np.array([[np.mean(c), np.std(c), np.max(c), np.min(c), len(c)] for c in curves])
                        clusterer = GMMClustering(n_clusters=n_cls)
                        labels = clusterer.fit(features)
                    else:  # DBSCAN
                        features = np.array([[np.mean(c), np.std(c)] for c in curves])
                        clusterer = DBSCANClustering(eps=0.5, min_samples=5)
                        labels = clusterer.fit(features)
                        n_cls = len(set(labels)) - (1 if -1 in labels else 0)
                    
                    if algo not in ["GMM", "DBSCAN"]:
                        status.text("🔄 执行聚类...")
                        prog.progress(50)
                        labels = clusterer.fit(curves)
                    
                    status.text("🔄 保存结果...")
                    prog.progress(80)
                    
                    rid = f"cluster_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    st.session_state.clustering_results[rid] = {
                        'id': rid, 'dataset_id': ds_id, 'algorithm': algo,
                        'n_clusters': n_cls, 'labels': labels, 'curves': curves
                    }
                    
                    prog.progress(100)
                    status.text("✅ 完成！")
                    st.success(f"✅ 聚类完成！ID: {rid}")
                    
                    # 显示分布
                    unique, counts = np.unique(labels, return_counts=True)
                    st.write("### 聚类分布")
                    for u, c in zip(unique, counts):
                        st.write(f"聚类 {u}: {c} 条")
                
                except Exception as e:
                    st.error(f"聚类失败: {e}")

# ============ 可视化 ============
elif page == "📈 可视化":
    st.header("可视化")
    
    if not st.session_state.clustering_results:
        st.warning("请先运行聚类")
    else:
        rid = st.selectbox("选择结果", options=list(st.session_state.clustering_results.keys()))
        if rid:
            r = st.session_state.clustering_results[rid]
            labels, curves = r['labels'], r['curves']
            
            vt = st.selectbox("类型", ["聚类分布", "中心曲线", "t-SNE降维", "PCA降维"])
            
            if st.button("生成"):
                prog = st.progress(0)
                status = st.empty()
                
                try:
                    if vt == "聚类分布":
                        status.text("🔄 生成分布图...")
                        prog.progress(50)
                        unique, counts = np.unique(labels, return_counts=True)
                        fig = px.bar(x=[f'聚类{i}' for i in unique], y=counts)
                        st.plotly_chart(fig)
                        prog.progress(100)
                        
                    elif vt == "中心曲线":
                        for i in range(r['n_clusters']):
                            prog.progress(int((i+1)/r['n_clusters']*100))
                            status.text(f"🔄 生成聚类{i}...")
                            cc = [curves[j] for j in range(len(curves)) if labels[j] == i]
                            if cc:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(y=np.mean(cc, axis=0), mode='lines'))
                                st.plotly_chart(fig)
                        
                    elif vt in ["t-SNE降维", "PCA降维"]:
                        status.text("🔄 降维计算中...")
                        prog.progress(30)
                        
                        # 提取特征
                        features = np.array([[np.mean(c), np.std(c), np.max(c), np.min(c)] for c in curves])
                        
                        if vt == "t-SNE降维":
                            from sklearn.manifold import TSNE
                            reducer = TSNE(n_components=2, random_state=42)
                        else:
                            from sklearn.decomposition import PCA
                            reducer = PCA(n_components=2)
                        
                        features_2d = reducer.fit_transform(features)
                        prog.progress(80)
                        
                        fig = px.scatter(x=features_2d[:,0], y=features_2d[:,1], color=labels.astype(str))
                        st.plotly_chart(fig)
                        prog.progress(100)
                    
                    status.text("✅ 完成！")
                
                except Exception as e:
                    st.error(f"可视化失败: {e}")

st.caption("拧紧曲线聚类分析系统 v3.0")
