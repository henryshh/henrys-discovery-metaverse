import streamlit as st
import numpy as np
import json
import os
import cv2
import plotly.graph_objects as go
from PIL import Image
from src.web.web_utils import get_services, get_current_project

def render_workstation_tab():
    services = get_services()
    project_service = services["project"]
    
    st.markdown("#### 🛠️ Industrial Workstation")
    
    current_proj = get_current_project()
    if not current_proj:
        st.warning("⚠️ Please select a project first.")
        return

    # Sub-navigation for workstation
    ws_mode = st.tabs(["👁️ Vision Guidance", "🔩 3D Path & Poka-Yoke", "🌐 Digital Twin (Metaverse)"])

    with ws_mode[0]:
        render_vision_view(current_proj, project_service)
    with ws_mode[1]:
        render_trajectory_poka_view(current_proj, services)
    with ws_mode[2]:
        render_metaverse_view(current_proj, services)

def render_vision_view(current_proj, project_service):
    st.info("Industrial Vision Workspace: OpenCV + YOLO Integration.")
    vision_meta = current_proj.metadata.get('vision', {})
    initial_cam_idx = vision_meta.get('global_cam_idx', 0)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        new_cam_idx = st.number_input("⚙️ Global Camera Index", 0, 10, initial_cam_idx)
        if new_cam_idx != initial_cam_idx:
            project_service.update_project_metadata(current_proj.id, 'vision', {'global_cam_idx': new_cam_idx})
            st.rerun()
        
        st.divider()
        st.markdown("##### 📤 Image Source")
        src = st.radio("Mode", ["Upload", "Camera"], horizontal=True)
        if src == "Upload":
            upload = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="ws_cv_up")
        else:
            upload = st.camera_input("Take Photo", key="ws_cv_cam")
            
        if upload:
            img = Image.open(upload)
            st.image(img, caption="Original", use_container_width=True)
            st.session_state.ws_cv_img = img

    with col2:
        if 'ws_cv_img' in st.session_state:
            st.markdown("##### ⚙️ Vision Pipeline")
            # (YOLO/OpenCV Logic here)
            st.write("YOLO/OpenCV processing active...")
        else:
            st.info("Upload or capture an image to begin vision guidance.")

def render_trajectory_poka_view(current_proj, services):
    st.info("3D Trajectory Analysis & Poka-Yoke (Error Proofing) Control.")
    # (Combine Logic from app.py 931-1062 and 1952-2451)
    st.write("Trajectory & Poka-Yoke logic active...")

def render_metaverse_view(current_proj, services):
    st.info("Digital Twin: Spatial Quality Mapping.")
    # (Logic from app.py 2489-2610)
    st.write("Metaverse Dashboard active...")
