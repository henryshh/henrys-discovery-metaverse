import streamlit as st
import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.web.web_utils import get_services, get_current_project
from src.feature_extractor import TighteningFeatureExtractor
from src.services.ai_baseline_service import AIBaselineService
from src.services.diagnosis_service import DiagnosisService

def render_analysis_tab():
    services = get_services()
    dataset_service = services["dataset"]
    db_service = services["db"]
    
    st.markdown("#### 📊 Data Import & Processing")
    
    current_proj = get_current_project()
    if not current_proj:
        st.warning("⚠️ Please select a project first from the Projects tab.")
        return

    st.info(f"Current Project Context: **{current_proj.name}**")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        with st.container(border=True):
            st.markdown("##### 📤 Import JSON Data")
            uploaded = st.file_uploader("Upload exported curves (.json)", type=['json'])
            
            if uploaded:
                try:
                    data = json.loads(uploaded.read().decode('utf-8'))
                    total = len(data)
                    ok = sum(1 for r in data if r.get('model', {}).get('report') == 'OK')
                    
                    cols = st.columns(3)
                    cols[0].metric("Total", total)
                    cols[1].metric("OK", ok)
                    cols[2].metric("NOK", total - ok)
                    
                    st.info("All curves are saved. Point consistency is validated during import.")
                    
                    if st.button("💾 Save Dataset", type="primary", use_container_width=True):
                        temp_path = f"tmp_{uploaded.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded.getbuffer())
                        
                        try:
                            sync_report = dataset_service.import_from_json(current_proj.id, temp_path, name_prefix=uploaded.name)
                            st.session_state.last_sync_report = sync_report
                            
                            st.markdown("##### 📊 Sync Status Report")
                            for ds_id, rep in sync_report.items():
                                with st.expander(f"📦 Silo: {rep['name']}", expanded=True):
                                    st.write(f"Added Curves: **{rep['added']}**")
                                    if rep['anomalies'] > 0:
                                        st.warning(f"⚠️ Detected **{rep['anomalies']}** inconsistencies (Length or Shape).")
                                    else:
                                        st.success("✅ Consistency verified.")
                            
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Import failed: {e}")
                except Exception as e:
                    st.error(f"Failed to process file: {e}")
    
    with col2:
        with st.container(border=True):
            st.markdown("##### 📋 Datasets")
            datasets = dataset_service.list_datasets(current_proj.id)
            if not datasets:
                st.info("No datasets available.")
            for ds in datasets:
                with st.container(border=True):
                    st.markdown(f"**{ds['name']}**")
                    st.caption(f"Created: {str(ds['created_at'])[:10]}")
    
    # --- Curve Analytics Explorer ---
    all_datasets = dataset_service.list_datasets(current_proj.id)
    if st.session_state.get('last_sync_report') or all_datasets:
        st.divider()
        st.markdown("#### 🔍 Curve Analytics Explorer (Torque-Angle Coordination)")
        
        sync_report = st.session_state.get('last_sync_report')
        options = [d['id'] for d in all_datasets]
        format_func = lambda x: next(d['name'] for d in all_datasets if d['id'] == x)
        default_sel = options[:1] if options else []
        
        if options:
            selected_ids = st.multiselect("Select Silos to Overlay", options=options, default=default_sel, format_func=format_func)
            
            if selected_ids:
                viz_col1, viz_col2 = st.columns([1, 4])
                with viz_col1:
                    view_mode = st.radio("View Dimension", ["2D (Torque-Angle)", "3D (Torque-Angle-Time)"])
                    # ✅ Industry Standard alignment is now a first-class citizen
                    align_strategy = st.selectbox(
                        "Alignment Strategy",
                        ["Torque-Angle (Industry Standard)", "Align at Torque Start (>0.5Nm)", "Original (No Alignment)"]
                    )
                    show_ok = st.checkbox("Show OK", value=True)
                    show_nok = st.checkbox("Show NOK", value=True)
                    sota_mode = st.toggle("Enable Expert Dynamics (dT/dα)", value=True)
                    show_tunnel = st.toggle("Show Statistical Tunnel (±3σ Envelope)", value=False)
                    max_curves = st.slider("Max Curves", 1, 500, 50)
                
                with viz_col2:
                    curves_to_plot = []
                    for ds_id in selected_ids:
                        if align_strategy == "Torque-Angle (Industry Standard)":
                            aligned_data = dataset_service.get_aligned_curves(current_proj.id, ds_id, n_points=200, limit=max_curves)
                            angle_grid = aligned_data["angle_grid"]
                            for i, c_torque in enumerate(aligned_data["curves"]):
                                meta = aligned_data["metadata"][i]
                                report = str(meta.get('report', '')).upper()
                                is_ok = ('OK' in report and 'NOK' not in report)
                                if (is_ok and show_ok) or (not is_ok and show_nok):
                                    # Create a temporary curve structure for plotting
                                    curves_to_plot.append({
                                        'torque': c_torque.tolist(),
                                        'angle': angle_grid.tolist(),
                                        'report': report,
                                        'resultNumber': meta.get('resultNumber'),
                                        'vin': meta.get('vin'),
                                        'is_pre_aligned': True
                                    })
                                if len(curves_to_plot) >= max_curves: break
                        else:
                            with db_service.get_project_db(current_proj.id) as p_conn:
                                p_cursor = p_conn.cursor()
                                p_cursor.execute("SELECT data FROM curves WHERE dataset_id = ?", (ds_id,))
                                for row in p_cursor.fetchall():
                                    c_data = json.loads(row[0])
                                    report = str(c_data.get('report', '')).upper()
                                    is_ok = ('OK' in report and 'NOK' not in report)
                                    if (is_ok and show_ok) or (not is_ok and show_nok):
                                        curves_to_plot.append(c_data)
                                    if len(curves_to_plot) >= max_curves: break
                        if len(curves_to_plot) >= max_curves: break
                    
                    if not curves_to_plot:
                        st.warning("No curves match selected filters.")
                    else:
                        render_plots(curves_to_plot, view_mode, align_strategy, sota_mode, show_tunnel)

def render_plots(curves_to_plot, view_mode, align_strategy, sota_mode, show_tunnel):
    extractor = TighteningFeatureExtractor()
    ai_service = AIBaselineService()
    diag_service = DiagnosisService()
    
    fig = go.Figure()
    fig_grad = go.Figure() if sota_mode else None
    expert_stats = []
    curve_data_list = []

    for c in curves_to_plot:
        y = np.array(c.get('torque', []))
        x = np.array(c.get('angle', []))
        if len(y) == 0: continue
        
        # Alignment
        x_aligned = x
        if not c.get('is_pre_aligned'):
            x_offset = 0
            if align_strategy == "Align at Torque Start (>0.5Nm)":
                start_idx = np.where(y > 0.5)[0]
                if len(start_idx) > 0: x_offset = x[start_idx[0]]
            x_aligned = x - x_offset
        
        report = str(c.get('report', '')).upper()
        color = '#2ecc71' if ('OK' in report and 'NOK' not in report) else '#e74c3c'
        res_num = c.get('resultNumber', 'N/A')
        
        if view_mode == "2D (Torque-Angle)":
            curve_data_list.append(y)
            fig.add_trace(go.Scatter(
                x=x_aligned, y=y, mode='lines',
                line=dict(width=1.5, color=color),
                name=f"RES {res_num}"
            ))
            
            if sota_mode:
                # Basic gradient analysis
                da = np.diff(x)
                dt = np.diff(y)
                grad = np.zeros_like(da)
                valid = da > 0.001
                grad[valid] = dt[valid] / da[valid]
                smooth_grad = np.convolve(grad, np.ones(5)/5, mode='same')
                
                fig_grad.add_trace(go.Scatter(
                    x=x_aligned[:-1], y=smooth_grad, mode='lines',
                    line=dict(width=1, color=color, dash='dot'),
                    name=f"Grad {res_num}"
                ))
    
    fig.update_layout(template="plotly_white", height=600, xaxis_title="Angle (°)", yaxis_title="Torque (Nm)")
    st.plotly_chart(fig, use_container_width=True)
    if sota_mode:
        fig_grad.update_layout(template="plotly_white", height=400, xaxis_title="Angle (°)", yaxis_title="Gradient (Nm/°)")
        st.plotly_chart(fig_grad, use_container_width=True)
