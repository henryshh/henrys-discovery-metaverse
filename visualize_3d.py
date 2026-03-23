"""
拧紧曲线 3D 可视化脚本
Tightening Curve 3D Visualization

使用方法:
    python visualize_3d.py

功能:
    - 3D 空间展示拧紧曲线 (torque-angle-current)
    - 可旋转查看不同角度
    - 区分 OK/NOK 曲线（如果有）
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


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
                metadata.append({
                    'vin': model.get('vin'),
                    'report': report
                })
    
    return curves, labels, metadata


def plot_3d_curves(curves, labels, output_path: str = None, sample_count: int = 10):
    """
    3D 空间绘制拧紧曲线
    
    Args:
        curves: 曲线数据列表
        labels: 标签列表 (1=OK, 0=NOK)
        output_path: 输出图片路径
        sample_count: 采样显示的曲线数量
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 采样显示
    ok_indices = [i for i, l in enumerate(labels) if l == 1]
    nok_indices = [i for i, l in enumerate(labels) if l == 0]
    
    # 随机采样
    np.random.seed(42)
    sample_ok = np.random.choice(ok_indices, min(sample_count, len(ok_indices)), replace=False)
    sample_nok = np.random.choice(nok_indices, min(sample_count // 3, len(nok_indices)), replace=False) if nok_indices else []
    
    # 绘制 OK 曲线（绿色）
    for idx in sample_ok:
        curve = curves[idx]
        points = np.arange(len(curve['torque']))
        ax.plot(points, curve['torque'], curve['angle'], 
                alpha=0.3, linewidth=1, color='green', label='OK' if idx == sample_ok[0] else '')
    
    # 绘制 NOK 曲线（红色）
    for idx in sample_nok:
        curve = curves[idx]
        points = np.arange(len(curve['torque']))
        ax.plot(points, curve['torque'], curve['angle'], 
                alpha=0.5, linewidth=2, color='red', label='NOK' if idx == sample_nok[0] else '')
    
    ax.set_xlabel('采样点 (Sample Point)', fontsize=12)
    ax.set_ylabel('扭矩 (Torque)', fontsize=12)
    ax.set_zlabel('角度 (Angle)', fontsize=12)
    ax.set_title('拧紧曲线 3D 可视化 (Torque-Angle 空间)', fontsize=14, fontweight='bold')
    
    # 图例
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(loc='upper left')
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = Path(__file__).parent / 'output' / 'curves_3d_torque_angle.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 3D curve saved: {output_path}")
    
    return output_path


def plot_3d_current(curves, labels, output_path: str = None, sample_count: int = 10):
    """
    3D 空间绘制拧紧曲线 - 含电流维度
    
    X 轴：采样点
    Y 轴：扭矩
    Z 轴：电流
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ok_indices = [i for i, l in enumerate(labels) if l == 1]
    
    np.random.seed(42)
    sample_ok = np.random.choice(ok_indices, min(sample_count, len(ok_indices)), replace=False)
    
    # 绘制 OK 曲线
    for idx in sample_ok:
        curve = curves[idx]
        points = np.arange(len(curve['torque']))
        ax.plot(points, curve['torque'], curve['current'], 
                alpha=0.3, linewidth=1, color='blue')
    
    ax.set_xlabel('采样点 (Sample Point)', fontsize=12)
    ax.set_ylabel('扭矩 (Torque)', fontsize=12)
    ax.set_zlabel('电流 (Current)', fontsize=12)
    ax.set_title('拧紧曲线 3D 可视化 (Torque-Current 空间)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = Path(__file__).parent / 'output' / 'curves_3d_torque_current.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 3D curve saved: {output_path}")
    
    return output_path


def plot_curve_distribution(curves, labels, output_path: str = None):
    """
    绘制曲线最终值的 3D 散点分布
    
    X 轴：最终扭矩
    Y 轴：最终角度
    Z 轴：最终电流
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ok_final_torque = []
    ok_final_angle = []
    ok_final_current = []
    
    for i, curve in enumerate(curves):
        if labels[i] == 1:  # OK
            ok_final_torque.append(curve['torque'][-1])
            ok_final_angle.append(curve['angle'][-1])
            ok_final_current.append(curve['current'][-1])
    
    # 散点图
    scatter = ax.scatter(ok_final_torque, ok_final_angle, ok_final_current, 
                         c='green', alpha=0.6, s=50, label='OK', depthshade=True)
    
    ax.set_xlabel('最终扭矩 (Final Torque)', fontsize=12)
    ax.set_ylabel('最终角度 (Final Angle)', fontsize=12)
    ax.set_zlabel('最终电流 (Final Current)', fontsize=12)
    ax.set_title('拧紧结果 3D 分布图', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = Path(__file__).parent / 'output' / 'curves_3d_distribution.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 3D distribution saved: {output_path}")
    
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
        
        print("\n[INFO] Generating 3D visualizations...")
        
        # 生成 3 种 3D 视图
        plot_3d_curves(curves, labels, str(output_dir / 'curves_3d_torque_angle.png'))
        plot_3d_current(curves, labels, str(output_dir / 'curves_3d_torque_current.png'))
        plot_curve_distribution(curves, labels, str(output_dir / 'curves_3d_distribution.png'))
        
        print("\n[DONE] All 3D visualizations generated!")
