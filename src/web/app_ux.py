                            st.session_state.current_project = proj
                            st.success(f"已选择: {proj['name']}")
                st.divider()
        else:
            st.info("暂无工程，请创建新工程")

# ============ 数据分析页（扁平化） ============
elif st.session_state.active_tab == "analysis":
    st.markdown("#### 📊 数据分析")
    
    if not st.session_state.current_project:
        st.warning("⚠️ 请先选择一个工程")
    else:
        st.write(f"当前工程: **{st.session_state.current_project['name']}**")
        
        # 两列布局：导入 + 列表
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("##### 📤 导入数据")
            
            uploaded = st.file_uploader("选择 JSON 文件", type=['json'])
            
            if uploaded:
                try:
                    data = json.loads(uploaded.read().decode('utf-8'))
                    total = len(data)
                    ok = sum(1 for r in data if r.get('model', {}).get('report') == 'OK')
                    nok = total - ok
                    
                    # 统计卡片
                    st.markdown(f"""
                    <div style="background: #f0f2f6; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>总记录</span><strong>{total}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; color: green;">
                            <span>OK</span><strong>{ok}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; color: red;">
                            <span>NOK</span><strong>{nok}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    filter_nok = st.checkbox("只使用 OK 曲线", value=True)
                    
                    if st.button("💾 保存到工程", type="primary", use_container_width=True):
                        ds_id = f"ds_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        
                        # 提取曲线和信息
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
                                    'resultNumber': model.get('resultNumber', 'N/A'),
                                    'vin': model.get('vin', 'N/A'),
                                    'report': model.get('report', 'N/A')
                                })
                        
                        st.session_state.datasets[ds_id] = {
                            'id': ds_id, 'name': uploaded.name, 'data': data,
                            'curves': curves, 'curve_info': curve_info,
                            'stats': {'total': total, 'ok': ok, 'nok': nok, 'filtered': len(curves)}
                        }
                        st.success(f"✅ 已保存 {len(curves)} 条曲线！")
                        
                except Exception as e:
                    st.error(f"处理失败: {e}")
        
        with col2:
            st.markdown("##### 📋 数据集列表")
            
            if st.session_state.datasets:
                for ds_id, ds in list(st.session_state.datasets.items())[-5:]:
                    with st.container():
                        cols = st.columns([3, 2, 1])
                        with cols[0]:
                            st.write(f"**{ds['name']}**")
                            st.caption(f"记录: {ds['stats']['filtered']} 条")
                        with cols[1]:
                            st.progress(ds['stats']['ok'] / ds['stats']['total'] if ds['stats']['total'] > 0 else 0)
                            st.caption(f"OK率: {ds['stats']['ok']/ds['stats']['total']*100:.1f}%")
                        with cols[2]:
                            if st.button("分析", key=f"analyze_{ds_id}"):
                                st.session_state.active_tab = "viz"
                                st.session_state.selected_dataset = ds_id
                                st.rerun()
                        st.divider()
            else:
                st.info("暂无数据集，请导入数据")

# ============ 聚类可视化页（2D/3D并排） ============
elif st.session_state.active_tab == "viz":
    st.markdown("#### 🔍 聚类分析与可视化")
    
    if not st.session_state.datasets:
        st.warning("⚠️ 请先导入数据")
    elif not CLUSTERING_READY:
        st.error("❌ 聚类算法未就绪")
    else:
        # 选择数据集
        ds_options = list(st.session_state.datasets.keys())
        selected_ds = st.session_state.get('selected_dataset', ds_options[0] if ds_options else None)
        
        ds_id = st.selectbox("选择数据集", options=ds_options,
                            format_func=lambda x: st.session_state.datasets[x]['name'],
                            index=ds_options.index(selected_ds) if selected_ds in ds_options else 0)
        
        if ds_id:
            ds = st.session_state.datasets[ds_id]
            curves = ds['curves']
            curve_info = ds.get('curve_info', [{}] * len(curves))
            
            st.write(f"数据集: **{ds['name']}** | 曲线数: **{len(curves)}**")
            
            # 聚类参数行
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                algo = st.selectbox("算法", ["DTW + K-Medoids", "K-Means", "GMM", "DBSCAN"])
            with col2:
                n_clusters = st.slider("聚类数", 2, 10, 5)
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                run_cluster = st.button("▶️ 运行聚类", type="primary", use_container_width=True)
            with col4:
                st.markdown("<br>", unsafe_allow_html=True)
                clear_results = st.button("🗑️ 清除结果", use_container_width=True)
            
            if clear_results:
                st.session_state.clustering_results = {}
                st.rerun()
            
            # 运行聚类
            if run_cluster:
                with st.spinner("正在聚类..."):
                    try:
                        if algo == "DTW + K-Medoids":
                            clusterer = DTWClustering(n_clusters=n_clusters)
                            labels = clusterer.fit(curves)
                        elif algo == "K-Means":
                            clusterer = KMeansClustering(n_clusters=n_clusters)
                            labels = clusterer.fit(curves)
                        elif algo == "GMM":
                            features = np.array([[np.mean(c), np.std(c), np.max(c), np.min(c), len(c)] for c in curves])
                            clusterer = GMMClustering(n_clusters=n_clusters)
                            labels = clusterer.fit(features)
                        else:  # DBSCAN
                            features = np.array([[np.mean(c), np.std(c)] for c in curves])
                            clusterer = DBSCANClustering(eps=0.5, min_samples=5)
                            labels = clusterer.fit(features)
                            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        
                        rid = f"cluster_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        st.session_state.clustering_results[rid] = {
                            'id': rid, 'dataset_id': ds_id, 'algorithm': algo,
                            'n_clusters': n_clusters, 'labels': labels, 'curves': curves,
                            'curve_info': curve_info, 'created_at': datetime.now().isoformat()
                        }
                        st.success(f"✅ 聚类完成！生成 {n_clusters} 个聚类")
                        
                    except Exception as e:
                        st.error(f"聚类失败: {e}")
            
            # 显示结果
            if st.session_state.clustering_results:
                st.divider()
                st.markdown("#### 📈 可视化结果")
                
                # 选择结果
                result_options = list(st.session_state.clustering_results.keys())
                selected_result = st.selectbox("选择结果", options=result_options,
                                              format_func=lambda x: f"{st.session_state.clustering_results[x]['algorithm']} ({st.session_state.clustering_results[x]['created_at'][:16]})")
                
                if selected_result:
                    r = st.session_state.clustering_results[selected_result]
                    labels, curves = r['labels'], r['curves']
                    curve_info = r.get('curve_info', [{}] * len(curves))
                    
                    # 统计行
                    unique, counts = np.unique(labels, return_counts=True)
                    cols = st.columns(len(unique))
                    for col, (u, c) in zip(cols, zip(unique, counts)):
                        with col:
                            st.metric(f"聚类 {u}", f"{c} 条")
                    
                    st.divider()
                    
                    # 2D/3D并排展示
                    st.markdown("##### 🔄 2D/3D 并排视图")
                    
                    col_2d, col_3d = st.columns(2)
                    
                    with col_2d:
                        st.markdown("**2D t-SNE 降维视图**")
                        try:
                            from sklearn.manifold import TSNE
                            features = np.array([[np.mean(c), np.std(c), np.max(c), np.min(c)] for c in curves])
                            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
                            features_2d = tsne.fit_transform(features)
                            
                            fig_2d = px.scatter(
                                x=features_2d[:, 0], y=features_2d[:, 1],
                                color=labels.astype(str),
                                title="t-SNE 2D 视图",
                                labels={'color': '聚类'},
                                height=400
                            )
                            fig_2d.update_layout(showlegend=True)
                            st.plotly_chart(fig_2d, use_container_width=True, key="tsne_2d")
                        except Exception as e:
                            st.error(f"2D视图失败: {e}")
                    
                    with col_3d:
                        st.markdown("**3D 交互式视图**")
                        try:
                            from visualization.plot_3d import plot_3d_scatter
                            features = np.array([[np.mean(c), np.std(c), np.max(c), np.min(c)] for c in curves])
                            fig_3d = plot_3d_scatter(features, labels, "3D 聚类视图")
                            fig_3d.update_layout(height=400)
                            st.plotly_chart(fig_3d, use_container_width=True, key="3d_view")
                        except Exception as e:
                            st.error(f"3D视图失败: {e}")
                    
                    st.divider()
                    
                    # 曲线对比区
                    st.markdown("##### 📊 聚类中心曲线对比")
                    
                    # 选择要对比的聚类
                    selected_clusters = st.multiselect(
                        "选择要对比的聚类",
                        options=list(range(r['n_clusters'])),
                        default=list(range(min(3, r['n_clusters'])))
                    )
                    
                    if selected_clusters:
                        fig_compare = go.Figure()
                        colors = px.colors.qualitative.Set1
                        
                        for idx, cluster_id in enumerate(selected_clusters):
                            cluster_indices = [j for j in range(len(curves)) if labels[j] == cluster_id]
                            if cluster_indices:
                                cc = [curves[j] for j in cluster_indices]
                                avg_curve = np.mean(cc, axis=0)
                                
                                fig_compare.add_trace(go.Scatter(
                                    y=avg_curve,
                                    mode='lines',
                                    name=f'聚类 {cluster_id} (n={len(cc)})',
                                    line=dict(color=colors[idx % len(colors)], width=3)
                                ))
                        
                        fig_compare.update_layout(
                            title="聚类中心曲线对比",
                            xaxis_title="采样点",
                            yaxis_title="扭矩值",
                            height=450,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_compare, use_container_width=True)
                        
                        # 统计对比表
                        st.markdown("##### 📋 统计对比")
                        stats_data = []
                        for cluster_id in selected_clusters:
                            cluster_indices = [j for j in range(len(curves)) if labels[j] == cluster_id]
                            if cluster_indices:
                                cc = [curves[j] for j in cluster_indices]
                                all_vals = np.concatenate(cc)
                                info_list = [curve_info[j] for j in cluster_indices]
                                sample_rn = info_list[0].get('resultNumber', 'N/A') if info_list else 'N/A'
                                
                                stats_data.append({
                                    '聚类': f'聚类 {cluster_id}',
                                    '曲线数': len(cc),
                                    '均值': f'{np.mean(all_vals):.2f}',
                                    '标准差': f'{np.std(all_vals):.2f}',
                                    '最大值': f'{np.max(all_vals):.2f}',
                                    '最小值': f'{np.min(all_vals):.2f}',
                                    '示例ResultNumber': sample_rn
                                })
                        
                        if stats_data:
                            import pandas as pd
                            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

st.markdown("---")
st.caption("拧紧曲线聚类分析系统 v2.0 | UX优化版")
