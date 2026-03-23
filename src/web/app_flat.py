"""拧紧曲线聚类分析系统 - 扁平化UI"""

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

# 初始化
if 'projects' not in st.session_state:
    st.session_state.projects = []
if 'current_project' not in st.session_state:
    st.session_state.current_project = None
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = {}

try:
    from ml.clustering import DTWClustering, KMeansClustering
    from ml.clustering.gmm import GMMClustering
    from ml.clustering.dbscan import DBSCANClustering
    CLUSTERING_READY = True
except:
    CLUSTERING_READY = False

# ============ 顶部导航栏 ============
st.markdown("""
<style>
.main-header {
    font-size: 2rem;
    font-weight: bold;
    color: #1f77b4;
}
.nav-button {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 10px 20px;
    margin: 5px;
}
.metric-card {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 20px;
    border-left: 5px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
with col1:
    st.markdown('<p class="main-header">🔧 拧紧曲线聚类分析</p>', unsafe_allow_html=True)
with col2:
    if st.button("🏠 首页", use_container_width=True):
        st.session_state.page = "home"
with col3:
    if st.button("📁 工程", use_container_width=True):
        st.session_state.page = "project"
with col4:
    if st.button("📊 数据", use_container_width=True):
        st.session_state.page = "data"
with col5:
    if st.button("🔍 分析", use_container_width=True):
        st.session_state.page = "analysis"

page = st.session_state.get('page', 'home')

# ============ 首页 ============
if page == "home":
    st.markdown("---")
    
    # 统计卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📁 工程</h3>
            <h2>{len(st.session_state.projects)}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 数据集</h3>
            <h2>{len(st.session_state.datasets)}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔍 聚类结果</h3>
            <h2>{len(st.session_state.clustering_results)}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚡ 算法</h3>
            <h2>{"4" if CLUSTERING_READY else "0"}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 快速开始
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚀 快速开始")
        st.write("1. 创建工程")
        st.write("2. 导入数据集")
        st.write("3. 运行聚类分析")
        st.write("4. 查看可视化结果")
        
        if st.button("➕ 创建新工程", type="primary"):
            st.session_state.page = "project"
            st.rerun()
    
    with col2:
        st.subheader("📈 最近结果")
        if st.session_state.clustering_results:
            for rid in list(st.session_state.clustering_results.keys())[-3:]:
                r = st.session_state.clustering_results[rid]
                st.write(f"• {r['algorithm']} - {r['n_clusters']}个聚类")
        else:
            st.info("暂无聚类结果")

# ============ 工程管理 ============
elif page == "project":
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("➕ 创建工程")
        name = st.text_input("工程名称", key="proj_name")
        desc = st.text_area("描述", key="proj_desc")
        if st.button("创建", type="primary"):
            if name:
                proj_id = f"proj_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                st.session_state.projects.append({
                    'id': proj_id, 'name': name, 'description': desc,
                    'created_at': datetime.now().isoformat()
                })
                st.success(f"✅ 工程 '{name}' 创建成功！")
    
    with col2:
        st.subheader("📋 工程列表")
        for proj in st.session_state.projects:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"**{proj['name']}**")
                st.caption(proj.get('description', '无描述'))
            with col_b:
                if st.button("选择", key=f"sel_{proj['id']}"):
                    st.session_state.current_project = proj
                    st.success(f"已选择: {proj['name']}")
            st.divider()

# ============ 数据导入 ============
elif page == "data":
    st.markdown("---")
    
    if not st.session_state.current_project:
        st.warning("⚠️ 请先选择一个工程")
    else:
        st.write(f"当前工程: **{st.session_state.current_project['name']}**")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📤 导入数据")
            uploaded = st.file_uploader("选择 JSON 文件", type=['json'])
            
            if uploaded:
                try:
                    data = json.loads(uploaded.read().decode('utf-8'))
                    total = len(data)
                    ok = sum(1 for r in data if r.get('model', {}).get('report') == 'OK')
                    
                    st.metric("总记录", total)
                    st.metric("OK", ok)
                    st.metric("NOK", total - ok)
                    
                    if st.button("保存到工程", type="primary"):
                        ds_id = f"ds_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        st.session_state.datasets[ds_id] = {
                            'id': ds_id, 'name': uploaded.name, 'data': data,
                            'stats': {'total': total, 'ok': ok, 'nok': total - ok}
                        }
                        st.success("✅ 保存成功！")
                except Exception as e:
                    st.error(f"处理失败: {e}")
        
        with col2:
            st.subheader("📊 数据集列表")
            for ds_id, ds in st.session_state.datasets.items():
                st.write(f"**{ds['name']}**")
                st.caption(f"记录: {ds['stats']['total']} | OK: {ds['stats']['ok']}")
                st.divider()

# ============ 聚类分析 ============
elif page == "analysis":
    st.markdown("---")
    
    if not st.session_state.datasets:
        st.warning("⚠️ 请先导入数据集")
    elif not CLUSTERING_READY:
        st.error("❌ 聚类算法未就绪")
    else:
        # 选择数据集
        ds_id = st.selectbox("选择数据集", options=list(st.session_state.datasets.keys()),
                            format_func=lambda x: st.session_state.datasets[x]['name'])
        
        if ds_id:
            data = st.session_state.datasets[ds_id]['data']
            
            # 提取曲线（过滤NOK）
            filter_nok = st.checkbox("只使用 OK 曲线", value=True)
            curves = []
            curve_info = []
            for r in data:
                model = r.get('model', {})
                if filter_nok and model.get('report') != 'OK':
                    continue
                pts = model.get('results', [{}])[0].get('curves', [{}])[0].get('data', {}).get('points', [])
                if pts:
                    curves.append(np.array([p.get('torque', 0) for p in pts]))
                    curve_info.append({
                        'resultNumber': model.get('resultNumber',