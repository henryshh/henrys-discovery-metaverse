# 拧紧曲线分类 - 模型对比报告

## 模型概述

实现了三种深度学习模型用于拧紧曲线质量分类：

| 模型 | 核心特点 | 参数量 | 适用场景 |
|------|---------|--------|---------|
| **CNN** | 局部特征提取 + 池化 | ~50K | 短序列，局部模式 |
| **TCN** | 因果卷积 + 残差连接 + 指数膨胀 | ~200K | 长序列，时序依赖 |
| **Transformer** | 自注意力 + 位置编码 | ~500K | 全局依赖，长距离关系 |

---

## 模型架构详解

### 1. CNN (卷积神经网络)

```
Input [500, 4]
  → Conv1D(4→32, k=7) + BN + ReLU + MaxPool
  → Conv1D(32→64, k=5) + BN + ReLU + MaxPool
  → Conv1D(64→128, k=3) + BN + ReLU
  → GlobalAvgPool
  → FC(128→1) + Sigmoid
```

**优点**：
- 计算效率高
- 平移不变性
- 适合局部特征提取

**缺点**：
- 感受野有限
- 难以捕获长距离依赖

---

### 2. TCN (时序卷积网络)

```
Input [500, 4]
  → TemporalBlock(dilation=1)
  → TemporalBlock(dilation=2)
  → TemporalBlock(dilation=4)
  → GlobalAvgPool
  → FC(128→1) + Sigmoid
```

**TemporalBlock**:
- 因果卷积（只依赖历史）
- 膨胀卷积（指数扩大感受野）
- 残差连接（解决梯度消失）

**优点**：
- 感受野随层数指数增长
- 并行计算（比RNN快）
- 稳定的梯度传播

**缺点**：
- 需要更多内存
- 对超参数敏感

---

### 3. Transformer

```
Input [500, 4]
  → Linear Projection(4→128)
  → Positional Encoding
  → TransformerEncoder × 4
      - MultiHeadAttention(8 heads)
      - FeedForward(128→512→128)
  → GlobalAvgPool
  → FC(128→1) + Sigmoid
```

**优点**：
- 全局注意力机制
- 捕获长距离依赖
- 可解释性强（注意力权重）

**缺点**：
- 计算复杂度高 O(n²)
- 需要更多数据
- 位置信息需额外编码

---

## 技术对比

| 特性 | CNN | TCN | Transformer |
|------|-----|-----|-------------|
| **感受野** | 局部 | 指数增长 | 全局 |
| **并行性** | [OK] | [OK] | [OK] |
| **长距离依赖** | ❌ | [OK] | [OK] |
| **计算复杂度** | O(n) | O(n) | O(n²) |
| **训练稳定性** | [OK] | [OK] | ⚠️ |
| **小数据集** | [OK] | [OK] | ❌ |

---

## 训练与评估

### 运行对比实验

```bash
cd tightening-ai
python compare_models.py
```

### 单独训练特定模型

```bash
# CNN
python train.py --model cnn --epochs 50

# TCN
python train.py --model tcn --epochs 50

# Transformer
python train.py --model transformer --epochs 50
```

---

## 预期结果

基于理论分析：

| 指标 | CNN | TCN | Transformer |
|------|-----|-----|-------------|
| 准确率 | ~95% | ~98% | ~97% |
| 训练速度 | 快 | 中等 | 慢 |
| 推理速度 | 快 | 快 | 中等 |
| 模型大小 | 小 | 中等 | 大 |

**推荐**：
- **生产环境**: CNN（速度快，模型小）
- **高精度需求**: TCN（平衡性能和效率）
- **研究分析**: Transformer（可解释性强）

---

## 未来改进方向

1. **混合架构**: CNN + Transformer（局部+全局）
2. **注意力可视化**: 分析模型关注哪些时间点
3. **对比学习**: 自监督预训练
4. **知识蒸馏**: 大模型 → 小模型

---

*创建时间: 2026-03-12*
*作者: 小刀 🔪*
