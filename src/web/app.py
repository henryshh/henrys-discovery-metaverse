"""Tightening AI Curve Clustering System - English UI"""

import streamlit as st
import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Path setup to include project root
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.web.web_utils import get_services, init_app_state, get_current_project
from src.feature_extractor import TighteningFeatureExtractor
from src.services.ai_baseline_service import AIBaselineService
from src.services.diagnosis_service import DiagnosisService

# Stage 4 Modular Tabs
from src.web.tabs.dashboard import render_dashboard_tab
from src.web.tabs.project import render_project_tab
from src.web.tabs.analysis import render_analysis_tab
from src.web.tabs.workstation import render_workstation_tab
from src.web.tabs.ml_lab import render_ml_lab_tab
from src.web.tabs.research import render_research_tab

# Set page config
st.set_page_config(page_title="Henry's Discovery Metaverse", page_icon="⚛️", layout="wide")

# ============ Premium Industrial CSS ============
st.markdown("""
<style>
    /* Global Typography */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Metric Styling - High Contrast & Clean */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #4A90E2 !important;
        margin-bottom: -5px !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.0rem !important;
        color: #E2E8F0 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
    }
    
    /* Solid Card Containers */
    div.st-emotion-cache-1r6slb0, div.st-emotion-cache-6q9sum {
        background-color: #1A1C24 !important;
        border: 1px solid #3F444E !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4) !important;
    }

    /* Main Content Area Optimization */
    [data-testid="stAppViewContainer"] {
        background-color: #F1F5F9 !important;
    }
    .block-container {
        background-color: #F1F5F9 !important;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Header Slogan Restoration */
    header[data-testid="stHeader"]::before {
        content: '“Assembly Metaverse: Converging Reality & Intelligence for Global Precision”';
        color: #8A99AC !important;
        font-size: 0.85rem !important;
        font-style: italic !important;
        position: absolute;
        left: 20px;
        top: 25px;
        letter-spacing: 0.01rem !important;
        white-space: nowrap !important;
        pointer-events: none !important;
    }

    /* Content Headings */
    .block-container h4 {
        color: #4A90E2 !important;
        font-weight: 700 !important;
        border-left: 5px solid #4A90E2;
        padding-left: 12px;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }

    /* Sidebar Refinement - MAXIMUM VISIBILITY */
    [data-testid="stSidebar"] {
        background-color: #090B10 !important;
        border-right: 1px solid #2D3748 !important;
    }

    /* Selection Header (选择功能域) */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-size: 1.1rem !important;
        color: #4A90E2 !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 0.2rem;
        margin-bottom: 20px !important;
        margin-top: 20px !important;
    }

    /* Radio Pillar Options (📊 Management, etc.) */
    [data-testid="stSidebar"] [data-baseweb="radio"] label div {
        color: #FFFFFF !important;  /* Pure White */
        font-size: 1.3rem !important;  /* Large size */
        font-weight: 600 !important;
        line-height: 1.6 !important;
        opacity: 1 !important;
    }

    /* Selected Pillar Highlighting */
    [data-testid="stSidebar"] [data-baseweb="radio"] div[aria-checked="true"] + div {
        color: #60A5FA !important;
        font-weight: 900 !important;
        text-shadow: 0 0 15px rgba(96, 165, 250, 0.5);
    }

    /* Sidebar Sub-tab Buttons */
    [data-testid="stSidebar"] button {
        border-radius: 10px !important;
        background-color: #1A202C !important;
        border: 1px solid #4A5568 !important;
        color: #F7FAFC !important;
        text-align: left !important;
        padding: 0.8rem 1.4rem !important;
        font-size: 1.1rem !important;
        width: 100% !important;
        margin-top: 8px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
    }

    [data-testid="stSidebar"] button:hover {
        border-color: #60A5FA !important;
        color: #60A5FA !important;
        background-color: rgba(96, 165, 250, 0.1) !important;
        transform: translateY(-1px);
    }

    [data-testid="stSidebar"] button[kind="primary"] {
        background-color: #4A90E2 !important;
        color: white !important;
        font-weight: 800 !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.4) !important;
    }

    /* Global Sidebar Font Fix */
    [data-testid="stSidebar"] * {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }
</style>
""", unsafe_allow_html=True)

# Initialization
services = get_services()
init_app_state()
project_service = services["project"]
dataset_service = services["dataset"]

if 'clustering_results' not in st.session_state: st.session_state.clustering_results = {}
if 'active_tab' not in st.session_state: st.session_state.active_tab = "dashboard"
if 'active_pillar' not in st.session_state: st.session_state.active_pillar = "control"

# Check backends (Lazy check)
@st.cache_resource
def check_backends():
    try:
        from src.ml.clustering import DTWClustering
        ready_clustering = True
    except:
        ready_clustering = False
    
    try:
        from src.ml.prediction.cnn_classifier import preprocess_curves
        ready_cnn = True
    except:
        ready_cnn = False
    return ready_clustering, ready_cnn

CLUSTERING_READY, CNN_READY = check_backends()

# ============ Navigation Management ============
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 0 0 10px 0; margin-top: -20px;'>
            <div style='color: #4A90E2; font-size: 2.2rem; font-weight: 800; line-height: 1.1; margin-bottom: 5px;'>Henry's</div>
            <div style='color: #D1D5DB; font-size: 0.95rem; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase;'>⚛️ DISCOVERY METAVERSE</div>
        </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    pillar_options = {
        "control": "📊 Management",
        "vision": "👁️ Machine Vision",
        "ai": "🧠 Machine Learning",
        "nlp": "💬 NLP Intelligence",
        "metaverse": "🌐 Assembly Metaverse"
    }
    
    selected_pillar_key = st.radio(
        "选择功能域 (Pillar)",
        options=list(pillar_options.keys()),
        format_func=lambda x: pillar_options[x],
        key="active_pillar_radio",
        index=list(pillar_options.keys()).index(st.session_state.active_pillar) if st.session_state.active_pillar in pillar_options else 0
    )
    
    if selected_pillar_key != st.session_state.active_pillar:
        st.session_state.active_pillar = selected_pillar_key
        defaults = {"control":"dashboard", "vision":"cv", "ai":"analysis", "nlp":"chat", "metaverse":"twin"}
        st.session_state.active_tab = defaults.get(selected_pillar_key, "dashboard")
        st.rerun()

    st.divider()
    
    # Context-sensitive Sub-tabs
    if st.session_state.active_pillar == "control":
        if st.button("📊 Dashboard", use_container_width=True, type="primary" if st.session_state.active_tab == "dashboard" else "secondary"):
            st.session_state.active_tab = "dashboard"; st.rerun()
        if st.button("📁 Project Config", use_container_width=True, type="primary" if st.session_state.active_tab == "project" else "secondary"):
            st.session_state.active_tab = "project"; st.rerun()
    elif st.session_state.active_pillar == "ai":
        if st.button("📈 Data & Sync", use_container_width=True, type="primary" if st.session_state.active_tab == "analysis" else "secondary"):
            st.session_state.active_tab = "analysis"; st.rerun()
        if st.button("🔬 DDS Viz", use_container_width=True, type="primary" if st.session_state.active_tab == "viz" else "secondary"):
            st.session_state.active_tab = "viz"; st.rerun()
        if st.button("🧠 AI Model Lab", use_container_width=True, type="primary" if st.session_state.active_tab == "classification" else "secondary"):
            st.session_state.active_tab = "classification"; st.rerun()

    # Infrastructure status card
    st.markdown("<div style='margin-top: 80px;'></div>", unsafe_allow_html=True)
    st.divider()
    with st.container():
        curr_proj = get_current_project()
        st.markdown(f"""
            <div style='background-color: #1A1C24; border: 1px solid #4A90E2; border-radius: 8px; padding: 12px;'>
                <div style='color: #4A90E2; font-size: 0.7rem; font-weight: 800;'>STATION LOADED</div>
                <div style='color: #E2E8F0; font-size: 0.9rem; font-weight: 600;'>{curr_proj.id if curr_proj else 'N/A'}</div>
                <div style='margin-top: 10px; display: flex; justify-content: space-between;'>
                    <span style='color: #10B981; font-size: 0.7rem;'>● ONLINE</span>
                    <span style='color: #64748B; font-size: 0.7rem;'>v2.1.4</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ============ UI Rendering ============
st.session_state.CLUSTERING_READY = CLUSTERING_READY
st.session_state.CNN_READY = CNN_READY

if st.session_state.active_tab == "dashboard":
    render_dashboard_tab()
elif st.session_state.active_tab == "project":
    render_project_tab()
elif st.session_state.active_tab == "analysis":
    render_analysis_tab()
elif st.session_state.active_tab in ["viz", "cv", "twin"]:
    render_workstation_tab()
elif st.session_state.active_tab == "classification":
    render_ml_lab_tab()
elif st.session_state.active_tab == "research":
    render_research_tab()
elif st.session_state.active_tab == "chat":
    st.markdown("#### 💬 NLP Intelligence Assistant")
    st.info("Chat module integration pending.")

# Footer
st.markdown("---")
st.caption("这个人间还是有像你一样的纯净的心灵与你作伴 | Henryshh@139.com")
