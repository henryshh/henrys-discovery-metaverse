"""
拧紧曲线 AI 分类模型 - 训练脚本
Tightening Curve AI Classification - Training Script

使用方法:
    python train.py --model cnn --epochs 50 --batch_size 16
"""

# 修复 Windows 控制台中文乱码
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import numpy as np
import os
import torch
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import TighteningDataLoader
from feature_extractor import TighteningFeatureExtractor
from models import CNN1D, LSTMClassifier, CNNLSTMHybrid, TCNClassifier, TransformerClassifier, ModelTrainer, TighteningCurveDataset
from torch.utils.data import DataLoader, random_split


def main():
    parser = argparse.ArgumentParser(description='拧紧曲线 AI 分类模型训练')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='../API/example data.json',
                        help='数据文件路径')
    parser.add_argument('--curve_length', type=int, default=500,
                        help='曲线标准化长度')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'lstm', 'hybrid', 'tcn', 'transformer'],
                        help='模型类型：cnn, lstm, hybrid, tcn, transformer')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='LSTM 隐藏层大小')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='验证集比例')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录')
    parser.add_argument('--save_model', action='store_true',
                        help='是否保存模型')
    
    args = parser.parse_args()
    
    # ========== 1. 加载数据 ==========
    print("=" * 60)
    print("STEP 1: 加载数据")
    print("=" * 60)
    
    data_path = Path(__file__).parent / args.data_path
    if not data_path.exists():
        print(f"❌ 数据文件不存在：{data_path}")
        return
    
    loader = TighteningDataLoader(str(data_path))
    loader.load().extract_curves(target_length=args.curve_length)
    
    curves, labels, metadata = loader.get_data()
    print(f"数据形状：{curves.shape}")
    print(f"通道：[torque, angle, current, marker]")
    print(f"标签分布：OK={sum(labels)}, NOK={len(labels)-sum(labels)}")
    
    # ========== 2. 划分数据集 ==========
    print("\n" + "=" * 60)
    print("STEP 2: 划分训练集/验证集")
    print("=" * 60)
    
    dataset = TighteningCurveDataset(curves, labels)
    
    # 按比例划分（保持类别平衡）
    n_train = int(len(dataset) * (1 - args.val_split))
    n_val = len(dataset) - n_train
    
    # Simple random split (no fixed seed to avoid torch issue)
    train_size = int(len(dataset) * (1 - args.val_split))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"训练集：{len(train_dataset)} 样本")
    print(f"验证集：{len(val_dataset)} 样本")
    
    # ========== 3. 创建模型 ==========
    print("\n" + "=" * 60)
    print(f"STEP 3: 创建模型 ({args.model.upper()})")
    print("=" * 60)
    
    if args.model == 'cnn':
        model = CNN1D(input_channels=4, num_classes=1)
    elif args.model == 'lstm':
        model = LSTMClassifier(input_size=4, hidden_size=args.hidden_size, num_classes=1)
    elif args.model == 'hybrid':
        model = CNNLSTMHybrid(input_channels=4, hidden_size=args.hidden_size, num_classes=1)
    elif args.model == 'tcn':
        model = TCNClassifier(input_channels=4, num_classes=1)
    elif args.model == 'transformer':
        model = TransformerClassifier(input_size=4, num_classes=1)
    else:
        raise ValueError(f"未知模型类型：{args.model}")
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量：{total_params:,}")
    print(f"可训练参数：{trainable_params:,}")
    
    # ========== 4. 训练模型 ==========
    print("\n" + "=" * 60)
    print("STEP 4: 训练模型")
    print("=" * 60)
    
    trainer = ModelTrainer(model)
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 创建输出目录
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 模型保存路径
    save_path = None
    if args.save_model:
        save_path = output_dir / f'{args.model}_model.pth'
    
    # 开始训练
    history = trainer.train(
        train_loader, 
        val_loader, 
        epochs=args.epochs,
        save_path=str(save_path) if save_path else None
    )
    
    # ========== 5. 输出结果 ==========
    print("\n" + "=" * 60)
    print("STEP 5: 训练完成")
    print("=" * 60)
    
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    
    print(f"\n📊 最佳验证准确率：{best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"📈 最终训练准确率：{history['train_acc'][-1]:.4f}")
    print(f"📉 最终验证准确率：{history['val_acc'][-1]:.4f}")
    
    if args.save_model:
        print(f"\n💾 模型已保存：{save_path}")
    
    # 保存训练历史
    import json
    history_path = output_dir / f'{args.model}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"📝 训练历史已保存：{history_path}")
    
    # ========== 6. 预测测试 ==========
    print("\n" + "=" * 60)
    print("STEP 6: 预测测试")
    print("=" * 60)
    
    # 用验证集测试
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for curves_batch, labels_batch in val_loader:
            curves_batch = curves_batch.to(trainer.device)
            labels_batch = labels_batch.to(trainer.device)
            
            outputs = model(curves_batch)
            predictions = (outputs > 0.5).float()
            
            correct += (predictions == labels_batch).sum().item()
            total += labels_batch.size(0)
    
    test_acc = correct / total
    print(f"验证集测试准确率：{test_acc:.4f} ({correct}/{total})")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
