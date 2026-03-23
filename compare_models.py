"""
模型对比实验
对比 CNN、TCN、Transformer 在拧紧曲线分类任务上的性能
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import numpy as np
from pathlib import Path
import json
import time

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import TighteningDataLoader
from models import CNN1D, TCNClassifier, TransformerClassifier, TighteningCurveDataset, ModelTrainer
from torch.utils.data import DataLoader, random_split


def train_and_evaluate(model, model_name, train_loader, val_loader, epochs=30):
    """训练并评估模型"""
    print(f"\n{'='*60}")
    print(f"训练模型: {model_name}")
    print('='*60)
    
    trainer = ModelTrainer(model)
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    history = trainer.train(train_loader, val_loader, epochs=epochs, save_path=None)
    train_time = time.time() - start_time
    
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    
    results = {
        'model': model_name,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'train_time': train_time,
        'total_params': total_params,
        'history': history
    }
    
    print(f"\n📊 {model_name} 结果:")
    print(f"  最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"  训练时间: {train_time:.1f}s")
    print(f"  参数量: {total_params:,}")
    
    return results


def main():
    print("="*60)
    print("拧紧曲线分类 - 模型对比实验")
    print("="*60)
    
    # ========== 1. 加载数据 ==========
    print("\n[1/4] 加载数据...")
    data_path = Path(__file__).parent / 'API' / 'Anord.json'
    
    loader = TighteningDataLoader(str(data_path))
    loader.load().extract_curves(target_length=500)
    curves, labels, metadata = loader.get_data()
    
    print(f"数据形状: {curves.shape}")
    print(f"样本数: {len(curves)}")
    print(f"标签分布: OK={sum(labels)}, NOK={len(labels)-sum(labels)}")
    
    # ========== 2. 划分数据集 ==========
    print("\n[2/4] 划分数据集...")
    dataset = TighteningCurveDataset(curves, labels)
    
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # ========== 3. 定义模型 ==========
    print("\n[3/4] 定义模型...")
    
    models = {
        'CNN': CNN1D(input_channels=4, num_classes=1),
        'TCN': TCNClassifier(input_channels=4, num_classes=1, 
                           hidden_channels=[32, 64, 128], kernel_size=7),
        'Transformer': TransformerClassifier(input_size=4, num_classes=1,
                                           d_model=128, nhead=8, num_layers=4)
    }
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {params:,} 参数")
    
    # ========== 4. 训练和对比 ==========
    print("\n[4/4] 开始训练...")
    
    results = []
    epochs = 30
    
    for name, model in models.items():
        result = train_and_evaluate(model, name, train_loader, val_loader, epochs)
        results.append(result)
    
    # ========== 5. 输出对比结果 ==========
    print("\n" + "="*60)
    print("模型对比总结")
    print("="*60)
    
    print(f"\n{'模型':<15} {'最佳准确率':<12} {'训练时间(s)':<12} {'参数量':<15}")
    print("-"*60)
    for r in results:
        print(f"{r['model']:<15} {r['best_val_acc']:<12.4f} {r['train_time']:<12.1f} {r['total_params']:<15,}")
    
    # 保存结果
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'model_comparison.json', 'w') as f:
        # 移除 history 后再保存
        results_save = [{k: v for k, v in r.items() if k != 'history'} for r in results]
        json.dump(results_save, f, indent=2)
    
    print(f"\n结果已保存到: {output_dir / 'model_comparison.json'}")
    
    # 找出最佳模型
    best_model = max(results, key=lambda x: x['best_val_acc'])
    print(f"\n🏆 最佳模型: {best_model['model']} (准确率: {best_model['best_val_acc']:.4f})")
    
    print("\n" + "="*60)
    print("对比实验完成!")
    print("="*60)


if __name__ == '__main__':
    main()
