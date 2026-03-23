import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.web.web_utils import get_services, get_current_project
from src.ml.clustering import DTWClustering, KMeansClustering, GMMClustering, DBSCANClustering
from datetime import datetime

def render_ml_lab_tab():
    services = get_services()
    dataset_service = services["dataset"]
    
    st.markdown("#### 🧠 Machine Learning & AI Lab")
    
    current_proj = get_current_project()
    if not current_proj:
        st.warning("⚠️ Please select a project.")
        return

    ml_mode = st.tabs(["🧩 Clustering Analysis", "🚂 CNN Training", "🧠 AI Shape Diagnostic"])
    
    with ml_mode[0]:
        render_clustering_section(current_proj, services)
    with ml_mode[1]:
        st.write("CNN Training Module (Draft)")
    with ml_mode[2]:
        st.write("AI Shape Diagnostic Module (Draft)")

def render_clustering_section(current_proj, services):
    dataset_service = services["dataset"]
    datasets = dataset_service.list_datasets(current_proj.id)
    if not datasets:
        st.info("No datasets available for clustering.")
        return
        
    ds_id = st.selectbox("Select Target Dataset for Clustering", [d['id'] for d in datasets], key="ml_cluster_ds",
                        format_func=lambda x: next(d['name'] for d in datasets if d['id'] == x))
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        algo = st.selectbox("Algorithm Strategy", ["DTW + K-Medoids", "K-Means", "GMM", "DBSCAN"], key="ml_algo")
    with col2:
        n_clusters = st.slider("Clusters (k)", 2, 10, 5, key="ml_k")
    with col3:
        st.write("")
        st.write("")
        if st.button("▶️ Execute", type="primary"):
            st.info(f"Executing {algo} with {n_clusters} clusters...")
            # (Clustering execution logic from app.py:900-928)
