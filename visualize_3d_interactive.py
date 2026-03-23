"""
拧紧曲线 3D 交互式可视化脚本
Tightening Curve 3D Interactive Visualization (Plotly)

使用方法:
    python visualize_3d_interactive.py

功能:
    - 生成可旋转、缩放的 3D 交互式 HTML 图
    - 支持鼠标拖拽旋转、滚轮缩放
    - 悬停显示数据点详情
"""

import json
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_data(data_path: str, target_length: int = 500):
    """加载拧紧曲线数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    curves = []
    labels = []
    metadata = []
    
    for record in data:
        model = record.get('model', {})
        results = model.get('results', [])
        report = model.get('report', 'OK')
        label = 1 if report == 'OK' else 0
        
        for result in results:
            curves_data = result.get('curves', [])
            for curve in curves_data:
                points = curve.get('data', {}).get('points', [])
                if len(points) == 0:
                    continue
                
                # 提取序列
                torque_seq = [p.get('torque', 0) for p in points]
                angle_seq = [p.get('angle', 0) for p in points]
                current_seq = [p.get('current', 0) for p in points]
                
                # 标准化长度
                if len(torque_seq) < target_length:
                    pad = target_length - len(torque_seq)
                    torque_seq = torque_seq + [0] * pad
                    angle_seq = angle_seq + [0] * pad
                    current_seq = current_seq + [0] * pad
                else:
                    torque_seq = torque_seq[:target_length]
                    angle_seq = angle_seq[:target_length]
                    current_seq = current_seq[:target_length]
                
                curves.append({
                    'torque': torque_seq,
                    'angle': angle_seq,
                    'current': current_seq
                })
                labels.append(label)
                
                # 获取 step 数据
                steps = result.get('steps', [{}])
                step_data = steps[0].get('data', {}) if steps else {}
                
                metadata.append({
                    'vin': model.get('vin'),
                    'report': report,
                    'finalTorque': step_data.get('finalTorque'),
                    'finalAngle': step_data.get('finalAngle'),
                })
    
    return curves, labels, metadata


def create_3d_torque_angle_html(curves, labels, metadata, output_path: str, sample_count: int = 20):
    """创建 Torque-Angle 3D 交互式图"""
    
    ok_indices = [i for i, l in enumerate(labels) if l == 1]
    nok_indices = [i for i, l in enumerate(labels) if l == 0]
    
    np.random.seed(42)
    
    fig = go.Figure()
    
    # OK 曲线
    sample_ok = np.random.choice(ok_indices, min(sample_count, len(ok_indices)), replace=False)
    for idx in sample_ok:
        curve = curves[idx]
        meta = metadata[idx]
        points = np.arange(len(curve['torque']))
        
        fig.add_trace(go.Scatter3d(
            x=points,
            y=curve['torque'],
            z=curve['angle'],
            mode='lines',
            line=dict(width=3, color='green'),
            opacity=0.4,
            name=f'OK #{idx}',
            hovertemplate=f'<b>OK Curve</b><br>VIN: {meta.get("vin", "N/A")}<br>Points: %{{x}}<br>Torque: %{{y:.2f}}<br>Angle: %{{z:.2f}}<extra></extra>'
        ))
    
    # NOK 曲线（如果有）
    if nok_indices:
        sample_nok = np.random.choice(nok_indices, min(sample_count // 3, len(nok_indices)), replace=False)
        for idx in sample_nok:
            curve = curves[idx]
            meta = metadata[idx]
            points = np.arange(len(curve['torque']))
            
            fig.add_trace(go.Scatter3d(
                x=points,
                y=curve['torque'],
                z=curve['angle'],
                mode='lines',
                line=dict(width=4, color='red'),
                opacity=0.6,
                name=f'NOK #{idx}',
                hovertemplate=f'<b>NOK Curve</b><br>VIN: {meta.get("vin", "N/A")}<br>Points: %{{x}}<br>Torque: %{{y:.2f}}<br>Angle: %{{z:.2f}}<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(
            text='<b>拧紧曲线 3D 可视化 (Torque-Angle 空间)</b><br>🖱️ 拖拽旋转 | 滚轮缩放 | 悬停查看详情',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(title=dict(text='采样点 (Sample Point)', font=dict(size=14))),
            yaxis=dict(title=dict(text='扭矩 (Torque)', font=dict(size=14))),
            zaxis=dict(title=dict(text='角度 (Angle)', font=dict(size=14))),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        height=800,
        showlegend=False,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    print(f"[OK] Interactive 3D saved: {output_path}")
    return output_path


def create_3d_torque_current_html(curves, labels, metadata, output_path: str, sample_count: int = 20):
    """创建 Torque-Current 3D 交互式图"""
    
    ok_indices = [i for i, l in enumerate(labels) if l == 1]
    
    np.random.seed(42)
    
    fig = go.Figure()
    
    sample_ok = np.random.choice(ok_indices, min(sample_count, len(ok_indices)), replace=False)
    for idx in sample_ok:
        curve = curves[idx]
        meta = metadata[idx]
        points = np.arange(len(curve['torque']))
        
        fig.add_trace(go.Scatter3d(
            x=points,
            y=curve['torque'],
            z=curve['current'],
            mode='lines',
            line=dict(width=3, color='blue'),
            opacity=0.4,
            name=f'Curve #{idx}',
            hovertemplate=f'<b>Curve #{idx}</b><br>VIN: {meta.get("vin", "N/A")}<br>Points: %{{x}}<br>Torque: %{{y:.2f}}<br>Current: %{{z:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='<b>拧紧曲线 3D 可视化 (Torque-Current 空间)</b><br>🖱️ 拖拽旋转 | 滚轮缩放 | 悬停查看详情',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(title=dict(text='采样点 (Sample Point)', font=dict(size=14))),
            yaxis=dict(title=dict(text='扭矩 (Torque)', font=dict(size=14))),
            zaxis=dict(title=dict(text='电流 (Current)', font=dict(size=14))),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        height=800,
        showlegend=False,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    print(f"[OK] Interactive 3D saved: {output_path}")
    return output_path


def create_3d_distribution_html(curves, labels, metadata, output_path: str):
    """创建最终结果 3D 分布交互式图"""
    
    fig = go.Figure()
    
    final_torque = []
    final_angle = []
    final_current = []
    hover_text = []
    
    for i, curve in enumerate(curves):
        if labels[i] == 1:  # OK
            final_torque.append(curve['torque'][-1])
            final_angle.append(curve['angle'][-1])
            final_current.append(curve['current'][-1])
            meta = metadata[i]
            hover_text.append(f'<b>OK #{i}</b><br>VIN: {meta.get("vin", "N/A")}<br>Torque: {curve["torque"][-1]:.2f}<br>Angle: {curve["angle"][-1]:.2f}<br>Current: {curve["current"][-1]:.2f}')
    
    fig.add_trace(go.Scatter3d(
        x=final_torque,
        y=final_angle,
        z=final_current,
        mode='markers',
        marker=dict(
            size=8,
            color='green',
            opacity=0.7,
            line=dict(color='darkgreen', width=2)
        ),
        name='OK',
        hovertext=hover_text,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>拧紧结果 3D 分布图</b><br>🖱️ 拖拽旋转 | 滚轮缩放 | 悬停查看详情',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(title=dict(text='最终扭矩 (Final Torque)', font=dict(size=14))),
            yaxis=dict(title=dict(text='最终角度 (Final Angle)', font=dict(size=14))),
            zaxis=dict(title=dict(text='最终电流 (Final Current)', font=dict(size=14))),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        height=800,
        showlegend=True,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    print(f"[OK] Interactive 3D saved: {output_path}")
    return output_path


def create_combined_html(curves, labels, metadata, output_path: str):
    """创建包含所有 3D 视图的单个 HTML 文件（标签页切换）"""
    
    ok_indices = [i for i, l in enumerate(labels) if l == 1]
    np.random.seed(42)
    
    # ========== 图 1: Torque-Angle ==========
    fig1 = go.Figure()
    sample_ok = np.random.choice(ok_indices, min(20, len(ok_indices)), replace=False)
    for idx in sample_ok:
        curve = curves[idx]
        points = np.arange(len(curve['torque']))
        fig1.add_trace(go.Scatter3d(
            x=points, y=curve['torque'], z=curve['angle'],
            mode='lines', line=dict(width=3, color='green'), opacity=0.4, name='OK'
        ))
    
    fig1.update_layout(
        title='<b>📐 Torque-Angle 空间</b>',
        scene=dict(
            xaxis=dict(title='Sample Point'),
            yaxis=dict(title='Torque'),
            zaxis=dict(title='Angle')
        ),
        height=600, showlegend=False
    )
    
    # ========== 图 2: Torque-Current ==========
    fig2 = go.Figure()
    sample_ok2 = np.random.choice(ok_indices, min(20, len(ok_indices)), replace=False)
    for idx in sample_ok2:
        curve = curves[idx]
        points = np.arange(len(curve['torque']))
        fig2.add_trace(go.Scatter3d(
            x=points, y=curve['torque'], z=curve['current'],
            mode='lines', line=dict(width=3, color='blue'), opacity=0.4, name='OK'
        ))
    
    fig2.update_layout(
        title='<b>⚡ Torque-Current 空间</b>',
        scene=dict(
            xaxis=dict(title='Sample Point'),
            yaxis=dict(title='Torque'),
            zaxis=dict(title='Current')
        ),
        height=600, showlegend=False
    )
    
    # ========== 图 3: 分布图 ==========
    fig3 = go.Figure()
    final_torque = [curves[i]['torque'][-1] for i in ok_indices]
    final_angle = [curves[i]['angle'][-1] for i in ok_indices]
    final_current = [curves[i]['current'][-1] for i in ok_indices]
    
    fig3.add_trace(go.Scatter3d(
        x=final_torque, y=final_angle, z=final_current,
        mode='markers', marker=dict(size=8, color='green', opacity=0.7),
        name='OK'
    ))
    
    fig3.update_layout(
        title='<b>📍 最终结果分布</b>',
        scene=dict(
            xaxis=dict(title='Final Torque'),
            yaxis=dict(title='Final Angle'),
            zaxis=dict(title='Final Current')
        ),
        height=600, showlegend=False
    )
    
    # ========== 组合 HTML ==========
    from plotly.subplots import make_subplots
    
    fig_combined = make_subplots(
        rows=3, cols=1,
        specs=[[{'type': 'scatter3d'}], [{'type': 'scatter3d'}], [{'type': 'scatter3d'}]],
        subplot_titles=('Torque-Angle 空间', 'Torque-Current 空间', '最终结果分布')
    )
    
    # 添加图 1
    sample_ok1 = np.random.choice(ok_indices, min(15, len(ok_indices)), replace=False)
    for idx in sample_ok1:
        curve = curves[idx]
        points = np.arange(len(curve['torque']))
        fig_combined.add_trace(go.Scatter3d(
            x=points, y=curve['torque'], z=curve['angle'],
            mode='lines', line=dict(width=3, color='green'), opacity=0.5,
            showlegend=False
        ), row=1, col=1)
    
    # 添加图 2
    sample_ok2 = np.random.choice(ok_indices, min(15, len(ok_indices)), replace=False)
    for idx in sample_ok2:
        curve = curves[idx]
        points = np.arange(len(curve['torque']))
        fig_combined.add_trace(go.Scatter3d(
            x=points, y=curve['torque'], z=curve['current'],
            mode='lines', line=dict(width=3, color='blue'), opacity=0.5,
            showlegend=False
        ), row=2, col=1)
    
    # 添加图 3
    fig_combined.add_trace(go.Scatter3d(
        x=final_torque, y=final_angle, z=final_current,
        mode='markers', marker=dict(size=6, color='red', opacity=0.8),
        showlegend=False
    ), row=3, col=1)
    
    fig_combined.update_layout(
        title=dict(
            text='<b>🔩 拧紧曲线 3D 交互式可视化</b><br>🖱️ 拖拽旋转 | 滚轮缩放 | 悬停查看数据点',
            font=dict(size=18)
        ),
        height=1800,
        margin=dict(l=0, r=0, t=100, b=0)
    )
    
    fig_combined.write_html(output_path, include_plotlyjs=True, full_html=True)
    print(f"[OK] Combined interactive 3D saved: {output_path}")
    return output_path


if __name__ == "__main__":
    # 数据路径
    data_file = Path(__file__).parent / '..' / 'API' / 'example data.json'
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    if not data_file.exists():
        print(f"[ERROR] Data file not found: {data_file}")
    else:
        print("[INFO] Loading data...")
        curves, labels, metadata = load_data(str(data_file))
        print(f"[OK] Loaded {len(curves)} curves")
        print(f"[OK] OK: {sum(labels)}, NOK: {len(labels) - sum(labels)}")
        
        print("\n[INFO] Generating interactive 3D HTML visualizations...")
        
        # 生成独立的 3D 图
        create_3d_torque_angle_html(
            curves, labels, metadata, 
            str(output_dir / 'curves_3d_torque_angle_interactive.html')
        )
        
        create_3d_torque_current_html(
            curves, labels, metadata,
            str(output_dir / 'curves_3d_torque_current_interactive.html')
        )
        
        create_3d_distribution_html(
            curves, labels, metadata,
            str(output_dir / 'curves_3d_distribution_interactive.html')
        )
        
        # 生成组合图
        create_combined_html(
            curves, labels, metadata,
            str(output_dir / 'curves_3d_combined_interactive.html')
        )
        
        print("\n[DONE] All interactive 3D visualizations generated!")
        print("\n[INFO] Open the HTML files in your browser to interact with the 3D plots!")
