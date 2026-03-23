import streamlit as st
import pandas as pd
from src.web.web_utils import get_services, get_current_project

def render_dashboard_tab():
    services = get_services()
    project_service = services["project"]
    dataset_service = services["dataset"]
    
    st.markdown("#### 📊 System Overview")
    
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Projects", len(project_service.list_projects()))
        with col2:
            current_proj = get_current_project()
            ds_count = len(dataset_service.list_datasets(current_proj.id)) if current_proj else 0
            st.metric("Total Datasets", ds_count)
        with col3:
            st.metric("Cluster Models", len(st.session_state.clustering_results))
        with col4:
            st.metric("Algorithms Ready", 4 if st.session_state.get('CLUSTERING_READY', False) else 0)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        with st.container(border=True):
            st.markdown("#### 🚀 Quick Actions")
            if st.button("➕ Create Project", type="primary", use_container_width=True):
                st.session_state.active_tab = "project"
                st.rerun()
            if st.button("📤 Import Data", use_container_width=True):
                st.session_state.active_tab = "analysis"
                st.rerun()
    
    with col2:
        with st.container(border=True):
            st.markdown("#### 📈 Recent Activity")
            if st.session_state.clustering_results:
                for rid in list(st.session_state.clustering_results.keys())[-3:]:
                    r = st.session_state.clustering_results[rid]
                    st.write(f"• **{r['algorithm']}** - {r['n_clusters']} clusters generated for {r.get('dataset_id', 'unknown')}")
            else:
                st.info("No recent clustering activity. Import data and start analysis.")
