"""
训练历史可视化脚本
Training History Visualization

使用方法:
    python visualize_history.py
"""

import json
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 修复 Windows 中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_training_history(history_path: str, output_path: str = None):
    """绘制训练历史曲线"""
    
    # 加载训练历史
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ========== Loss 曲线 ==========
    ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失 (Train Loss)', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失 (Val Loss)', linewidth=2)
    ax1.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax1.set_ylabel('损失 (Loss)', fontsize=12)
    ax1.set_title('训练损失曲线', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=1)
    
    # ========== Accuracy 曲线 ==========
    ax2.plot(epochs, history['train_acc'], 'g-', label='训练准确率 (Train Acc)', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'm-', label='验证准确率 (Val Acc)', linewidth=2)
    ax2.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax2.set_ylabel('准确率 (Accuracy)', fontsize=12)
    ax2.set_title('训练准确率曲线', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=1)
    ax2.set_ylim(bottom=0, top=1.05)
    
    # 添加最佳 epoch 标注
    best_epoch = history['val_acc'].index(max(history['val_acc'])) + 1
    best_val_acc = max(history['val_acc'])
    ax2.axvline(x=best_epoch, color='orange', linestyle='--', alpha=0.7, 
                label=f'最佳 Epoch {best_epoch} (Acc={best_val_acc:.2%})')
    ax2.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图片
    if output_path is None:
        output_path = Path(history_path).parent / 'training_history.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Training curve saved: {output_path}")
    
    # 显示图片（可选）
    # plt.show()
    
    return output_path


if __name__ == "__main__":
    # 默认路径
    history_file = Path(__file__).parent / 'output' / 'cnn_history.json'
    
    if not history_file.exists():
        print(f"[ERROR] History file not found: {history_file}")
        print("Please run training script first")
    else:
        plot_training_history(str(history_file))
        print("\n[DONE]")
