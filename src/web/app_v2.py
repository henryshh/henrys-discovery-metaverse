"""拧紧曲线聚类分析系统 - Web界面 v2（功能完整版）"""

import streamlit as st
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入核心模块
try:
    from core.database import DatabaseManager
    from services.project_service import ProjectService
    from services.dataset_service import DatasetService
    from ml.clustering import DTWClustering, KMeansClustering
    db = DatabaseManager()
    project_service = ProjectService(db)
    dataset_service = DatasetService(db)
    st.session_state['db_ready'] = True
except Exception as e:
    st.session_state['db_ready'] = False
    st.session_state['db_error'] = str(e)

st.set_page_config(
    page_title="拧紧曲线聚类分析系统",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state
if 'projects' not in st.session_state:
    st.session_state.projects = []
if 'current_project' not in st.session_state:
    st.session_state.current_project = None
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = {}

def load_projects():
    """加载工程列表"""
    try:
        if st.session_state.get('db_ready'):
            projects = project_service.list_projects()
            st.session_state.projects = projects
            return projects
    except Exception as e:
        st.error(f"加载工程失败: {e}")
    return []

def analyze_dataset(file_content):
    """分析数据集"""
    try:
        data = json.loads(file_content)
        
        # 统计信息
        total_records = len(data)
        ok_count = sum(1 for r in data if r.get('model', {}).get('report') == 'OK')
        nok_count = total_records - ok_count
        
        # 曲线统计
        curves_with_data = sum(1 for r in data if r.get('model', {}).get('results', [{}])[0].get('curves'))
        
        # 时间范围
        timestamps = [r.get('model', {}).get('controllerDateTime', '') for r in data]
        timestamps = [t for t in timestamps if t]
        
        return {
            'total_records': total_records,
            'ok_count': ok_count,
            'nok_count': nok_count,
            'ok_rate': ok_count / total_records * 100 if total_records > 0 else 0,
            'curves_count': curves_with_data,
            'time_range': f"{min(timestamps)} ~ {max(timestamps)}" if timestamps else "N/A",
            'sample': data[0] if data else None
        }
    except Exception as e:
        return {'error': str(e)}

def preprocess_curves(data):
    """预处理曲线数据"""
    processed = []
    for record in data:
        model = record.get('model', {})
        results = model.get('results', [{}])[0]
        curves = results.get('curves', [])
        
        if curves and curves[0].get('data', {}).get('points'):
            points = curves[0]['data']['points']
            torque = [p.get('torque', 0) for p in points]
            angle = [p.get('angle', 0) for p in points]
            
            # 数据清洗：移除异常值
            torque = np.array(torque)
            angle = np.array(angle)
            
            # 插值统一长度
            target_len = 100
            if len(torque) > 0:
                x_old = np.linspace(0, 1, len(torque))
                x_new = np.linspace(0, 1, target_len)
                torque_interp = np.interp(x_new, x_old, torque)
                
                processed.append({
                    'id': model.get('resultNumber', 'unknown'),
                    'torque': torque_interp,
                    'angle': angle,
                    'report': model.get('report', 'N/A'),
                    'vin': model.get('vin', 'N/A')
                })
    
    return processed

st.title("🔧 拧紧曲线聚类分析系统 v2")
st.markdown("---")

# 侧边栏导航
st.sidebar.title("导航")
page = st.sidebar.radio("选择功能", [
    "🏠 首页",
    "📁 工程管理",
    "📊 数据集导入",
    "🔍 聚类分析",
    "📈 可视化",
    "🔧 诊断预测"
])

if page == "🏠 首页":
    st.header("欢迎使用")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        projects = load_projects()
        st.metric("工程数量", len(projects))
    with col2:
        st.metric("聚类算法", "8+")
    with col3:
        st.metric("可视化", "10+")
    
    st.write("---")
    st.write("### 功能特性")
    st.write("- 📁 多工程管理")
    st.write("- 📊 数据集导入与预处理")
    st.write("- 🔍 多种聚类算法（DTW、K-Means等）")
    st.write("- 📈 丰富的可视化")
    st.write("- 🔧 异常检测与预测")

elif page == "📁 工程管理":
    st.header("工程管理")
    
    # 创建工程
    with st.expander("➕ 创建新工程", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            project_name = st.text_input("工程名称", key="new_project_name")
        with col2:
            project_desc = st.text_area("工程描述", key="new_project_desc")
        
        if st.button("创建工程", type="primary"):
            if project_name:
                try:
                    if st.session_state.get('db_ready'):
                        project_id = project_service.create_project(project_name, project_desc)
                        st.success(f"✅ 工程 '{project_name}' 创建成功！ID: {project_id}")
                        load_projects()
                    else:
                        # 模拟创建
                        project_id = f"proj_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        st.session_state.projects.append({
                            'id': project_id,
                            'name': project_name,
                            'description': project_desc,
                            'created_at': datetime.now().isoformat()
                        })
                        st.success(f"✅ 工程 '{project_name}' 创建成功！")
                except Exception as e:
                    st.error(f"创建失败: {e}")
            else:
                st.warning("请输入工程名称")
    
    # 工程列表
    st.write("---")
    st.subheader("📋 工程列表")
    
    projects = load_projects()
    if projects:
        for proj in projects:
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"**{proj.get('name', '未命名')}**")
                    st.caption(f"ID: {proj.get('id', 'N/A')}")
                with col2:
                    st.caption(f"创建时间: {proj.get('created_at', 'N/A')[:10]}")
                with col3:
                    if st.button("选择", key=f"select_{proj.get('id')}"):
                        st.session_state.current_project = proj
                        st.success(f"已选择工程: {proj.get('name')}")
                st.write(f"描述: {proj.get('description', '无')}")
                st.markdown("---")
    else:
        st.info("暂无工程，请创建新工程")

elif page == "📊 数据集导入":
    st.header("数据集导入与预处理")
    
    if not st.session_state.current_project:
        st.warning("⚠️ 请先选择一个工程")
    else:
        st.write(f"当前工程: **{st.session_state.current_project.get('name')}**")
        
        # 上传数据
        uploaded_file = st.file_uploader("上传 JSON 文件", type=['json'], key="dataset_upload")
        
        if uploaded_file:
            file_content = uploaded_file.read().decode('utf-8')
            
            with st.spinner("正在分析数据..."):
                analysis = analyze_dataset(file_content)
            
            if 'error' in analysis:
                st.error(f"分析失败: {analysis['error']}")
            else:
                st.success("✅ 数据分析完成！")
                
                # 显示统计信息
                st.write("---")
                st.subheader("📊 数据统计")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总记录数", analysis['total_records'])