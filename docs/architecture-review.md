# 拧紧曲线聚类分析系统 - 架构评估与重构设计

> 评估日期：2026-03-20  
> 评估人：架构师 (Architect Agent)  
> 项目版本：v1.0.0

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [当前架构分析](#2-当前架构分析)
3. [架构问题识别](#3-架构问题识别)
4. [目标架构设计](#4-目标架构设计)
5. [重构实施计划](#5-重构实施计划)
6. [代码重构示例](#6-代码重构示例)
7. [企业内网部署方案](#7-企业内网部署方案)

---

## 1. 执行摘要

### 1.1 项目现状

拧紧曲线智能分析系统已完成核心功能开发：

| 功能模块 | 完成状态 | 技术实现 |
|---------|---------|---------|
| DTW聚类分析 | ✅ 完成 | 层次聚类 + 并行计算 |
| 特征聚类 | ✅ 完成 | K-Means + PCA |
| CNN分类模型 | ✅ 完成 | 100%验证准确率 |
| 交互式可视化 | ✅ 完成 | Streamlit + Plotly |
| 工程管理 | ✅ 完成 | 项目隔离 + 数据追加 |
| 离线分发 | ✅ 完成 | wheel包 + 批量安装 |

### 1.2 核心发现

**优势：**
- 核心模块（`src/`）设计良好，职责相对清晰
- 工程管理功能完善，支持项目隔离
- 已考虑离线部署需求

**问题：**
- 代码重复率高（约35%），多个脚本重复实现相同功能
- 根目录脚本混乱，缺乏统一入口
- 缺少抽象层，扩展性受限
- 配置管理分散，硬编码路径普遍

### 1.3 重构建议

推荐采用 **分层架构 + 策略模式** 进行重构：

```
表现层 (Streamlit UI)
    ↓
应用层 (Use Cases / Services)
    ↓
领域层 (Domain Models / Algorithms)
    ↓
基础设施层 (Data Access / Storage)
```

---

## 2. 当前架构分析

### 2.1 项目结构现状

```
tightening-ai/
├── src/                          # ✅ 核心模块（设计较好）
│   ├── data_loader.py           # 数据加载
│   ├── feature_extractor.py     # 特征提取
│   └── models.py                # 深度学习模型
│
├── [根目录脚本]                   # ⚠️ 问题区域
│   ├── train.py                 # 模型训练
│   ├── cluster_analysis.py      # K-Means聚类
│   ├── dtw_cluster_full.py      # DTW聚类
│   ├── advanced_clustering.py   # 高级聚类
│   ├── compare_models.py        # 模型对比
│   └── ...（其他10+脚本）
│
├── streamlit_app/                # ⚠️ UI层耦合
│   ├── app.py                   # 主入口
│   └── pages/                   # 多页面
│
├── project_manager.py            # ✅ 工程管理
├── API/                          # 数据目录
├── output/                       # 输出目录
├── projects/                     # 工程目录
└── offline_distribution/         # 离线分发
```

### 2.2 模块依赖关系

```
当前依赖图（问题标注）：

streamlit_app/app.py
    └── project_manager.py
    └── src/data_loader.py (直接导入)
    └── 硬编码路径 ⚠️

dtw_cluster_full.py
    └── 自实现 parse_anord_json() ⚠️ 重复
    └── 自实现 normalize_curve() ⚠️ 重复

cluster_analysis.py
    └── 自实现 parse_curves() ⚠️ 重复（与上面不同实现）
    └── 自实现 extract_features() ⚠️ 重复

advanced_clustering.py
    └── 自实现 load_curves() ⚠️ 重复（第三种实现）
    └── 自实现 extract_features() ⚠️ 重复（与上面不同）
```

### 2.3 代码质量指标

| 指标 | 当前值 | 目标值 | 状态 |
|------|--------|--------|------|
| 代码重复率 | ~35% | <10% | ⚠️ 需改进 |
| 模块耦合度 | 高 | 低 | ⚠️ 需改进 |
| 测试覆盖率 | 0% | >80% | ❌ 缺失 |
| 类型注解 | 部分 | 完整 | ⚠️ 需改进 |
| 文档完整性 | 60% | >90% | ⚠️ 需改进 |

---

## 3. 架构问题识别

### 3.1 代码模块化问题

#### 问题1：数据解析逻辑重复

**发现：** 3个不同的JSON解析实现

```python
# dtw_cluster_full.py
def parse_anord_json(filepath):
    parts = content.split('}{')
    # ... 实现1

# cluster_analysis.py  
def parse_curves(filepath):
    parts = content.split('}{')
    # ... 实现2（略有不同）

# advanced_clustering.py
def load_curves(filepath='API/Anord.json'):
    parts = content.split('}{')
    # ... 实现3（又有不同）
```

**影响：**
- 维护成本高：修改一处需同步多处
- 行为不一致：不同脚本解析结果可能有差异
- 测试困难：无法统一测试解析逻辑

#### 问题2：特征提取逻辑分散

```python
# cluster_analysis.py - 15维特征
feat = [np.max(torque), np.min(torque), ...]  # 15个特征

# feature_extractor.py - 30+维特征
features.extend([np.mean(torque), np.std(torque), ...])  # 更完整

# advanced_clustering.py - 又是15维特征
feat = [np.max(torque), np.min(torque), ...]  # 与cluster_analysis相同
```

**影响：**
- 特征定义不一致
- 无法复用已提取的特征
- 用户难以理解使用哪个特征集

### 3.2 职责分离问题

#### 问题3：脚本职责重叠

| 脚本 | 功能 | 重叠度 |
|------|------|--------|
| `cluster_analysis.py` | K-Means聚类 | 与advanced_clustering重叠 |
| `dtw_cluster_full.py` | DTW聚类 | 数据加载与feature_extractor重叠 |
| `advanced_clustering.py` | 多种聚类 | 包含cluster_analysis全部功能 |
| `compare_models.py` | 模型对比 | 与train.py逻辑相似 |

**建议：** 合并为统一的聚类分析入口

#### 问题4：UI与业务逻辑耦合

```python
# streamlit_app/pages/0_工程管理.py
# 直接在UI代码中调用底层模块

from project_manager import ProjectManager

@st.cache_resource
def get_project_manager():
    return ProjectManager(workspace_dir="projects")

# UI代码中直接处理业务逻辑
if st.button("开始导入", ...):
    pm.import_data(project_name, str(temp_path), data_format)
```

**影响：**
- 无法单独测试业务逻辑
- 更换UI框架需要大量修改
- 业务逻辑分散在多个页面文件

### 3.3 可扩展性问题

#### 问题5：缺少算法抽象

```python
# 当前实现：每种聚类都是独立代码
def hierarchical_clustering_dtw(curves, output_dir='output'):
    # 层次聚类实现

def kshape_clustering(curves, output_dir='output'):
    # K-Shape实现

def dbscan_clustering(curves, output_dir='output'):
    # DBSCAN实现
```

**问题：**
- 无法动态切换算法
- 添加新算法需要修改主程序
- 无法并行比较多个算法

#### 问题6：硬编码配置

```python
# 多处硬编码
filepath='API/Anord.json'
output_dir='output/advanced_clustering'
target_length=500
n_clusters=5
```

**影响：**
- 部署到不同环境需要修改代码
- 无法通过配置文件调整参数
- 测试时无法轻松mock

### 3.4 技术债务清单

| ID | 描述 | 严重程度 | 影响 |
|----|------|----------|------|
| TD-001 | 数据解析重复实现3次 | 高 | 维护成本 |
| TD-002 | 特征提取重复实现2次 | 高 | 一致性问题 |
| TD-003 | 无单元测试 | 高 | 质量风险 |
| TD-004 | 硬编码路径 | 中 | 部署困难 |
| TD-005 | UI与业务耦合 | 中 | 可测试性 |
| TD-006 | 无统一配置管理 | 中 | 可维护性 |
| TD-007 | 缺少日志系统 | 低 | 调试困难 |
| TD-008 | 无API文档 | 低 | 使用门槛 |

---

## 4. 目标架构设计

### 4.1 架构选型

推荐采用 **分层架构 + 策略模式**：

**选择理由：**

1. **分层架构**
   - 优点：职责清晰，易于测试，符合企业开发习惯
   - 缺点：可能增加代码量
   - 适用：企业内网独立部署，不需要微服务

2. **策略模式**
   - 优点：算法可插拔，易于扩展
   - 缺点：需要抽象设计
   - 适用：多种聚类/分类算法并存

3. **不选择插件化架构**
   - 原因：企业内网离线环境，不需要动态加载
   - 策略模式 + 配置文件足够灵活

### 4.2 目标目录结构

```
tightening-ai/
├── config/                       # 配置管理
│   ├── __init__.py
│   ├── settings.py              # 全局配置
│   └── default.yaml             # 默认配置
│
├── src/                         # 核心源码
│   ├── __init__.py
│   │
│   ├── domain/                  # 领域层
│   │   ├── __init__.py
│   │   ├── entities.py         # 实体定义（Curve, Project等）
│   │   ├── value_objects.py    # 值对象（Feature, Label等）
│   │   └── interfaces.py       # 接口定义
│   │
│   ├── application/             # 应用层
│   │   ├── __init__.py
│   │   ├── services/           # 业务服务
│   │   │   ├── __init__.py
│   │   │   ├── curve_service.py
│   │   │   ├── clustering_service.py
│   │   │   └── prediction_service.py
│   │   └── use_cases/          # 用例
│   │       ├── __init__.py
│   │       ├── import_data.py
│   │       ├── train_model.py
│   │       └── analyze_curve.py
│   │
│   ├── infrastructure/          # 基础设施层
│   │   ├── __init__.py
│   │   ├── persistence/        # 持久化
│   │   │   ├── __init__.py
│   │   │   ├── project_repository.py
│   │   │   └── model_repository.py
│   │   ├── parsers/            # 数据解析
│   │   │   ├── __init__.py
│   │   │   ├── json_parser.py
│   │   │   └── csv_parser.py
│   │   └── cache/              # 缓存
│   │       ├── __init__.py
│   │       └── dtw_cache.py
│   │
│   └── algorithms/              # 算法模块
│       ├── __init__.py
│       ├── base.py             # 算法基类
│       ├── clustering/         # 聚类算法
│       │   ├── __init__.py
│       │   ├── kmeans.py
│       │   ├── dtw_hierarchical.py
│       │   ├── dbscan.py
│       │   └── gmm.py
│       └── classification/      # 分类算法
│           ├── __init__.py
│           ├── cnn.py
│           ├── lstm.py
│           └── transformer.py
│
├── ui/                          # 表现层
│   ├── __init__.py
│   ├── streamlit_app/          # Streamlit实现
│   │   ├── app.py
│   │   ├── components/        # 可复用组件
│   │   └── pages/
│   └── cli/                    # 命令行接口（可选）
│       └── main.py
│
├── tests/                       # 测试
│   ├── __init__.py
│   ├── unit/                   # 单元测试
│   ├── integration/            # 集成测试
│   └── fixtures/               # 测试数据
│
├── scripts/                     # 工具脚本
│   ├── train.py                # 训练入口
│   ├── cluster.py              # 聚类入口
│   └── export.py               # 导出工具
│
├── docs/                        # 文档
│   ├── api/                    # API文档
│   ├── user_guide/             # 用户手册
│   └── architecture/           # 架构文档
│
├── data/                        # 数据目录（外部）
│   ├── raw/                    # 原始数据
│   ├── processed/              # 处理后数据
│   └── models/                 # 模型文件
│
├── projects/                    # 工程目录
├── output/                      # 输出目录
├── offline_distribution/        # 离线分发
│
├── config.yaml                  # 配置文件
├── requirements.txt
├── setup.py                     # 安装配置
└── README.md
```

### 4.3 核心模块设计

#### 4.3.1 领域层 - 实体定义

```python
# src/domain/entities.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np


@dataclass
class CurvePoint:
    """曲线数据点"""
    torque: float
    angle: float
    current: float
    marker: int = 0
    timestamp: Optional[float] = None


@dataclass
class TighteningCurve:
    """拧紧曲线实体"""
    id: str
    points: List[CurvePoint]
    report: str = "UNKNOWN"  # OK / NOK
    result_number: int = 0
    station: str = ""
    tool: str = ""
    metadata: Dict = field(default_factory=dict)
    
    @property
    def torque_sequence(self) -> np.ndarray:
        return np.array([p.torque for p in self.points])
    
    @property
    def angle_sequence(self) -> np.ndarray:
        return np.array([p.angle for p in self.points])
    
    @property
    def length(self) -> int:
        return len(self.points)
    
    def normalize(self, target_length: int = 500) -> 'TighteningCurve':
        """标准化曲线长度"""
        # ... 实现


@dataclass
class FeatureSet:
    """特征集"""
    curve_id: str
    features: np.ndarray
    feature_names: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'curve_id': self.curve_id,
            'features': self.features.tolist(),
            'feature_names': self.feature_names
        }


@dataclass  
class Project:
    """工程实体"""
    id: str
    name: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: str = "created"
    curves: List[TighteningCurve] = field(default_factory=list)
    models: List[Dict] = field(default_factory=list)
    
    def add_curves(self, curves: List[TighteningCurve]):
        self.curves.extend(curves)
        self.updated_at = datetime.now()
        self.status = "data_loaded"
```

#### 4.3.2 算法层 - 策略模式

```python
# src/algorithms/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class ClusteringAlgorithm(ABC):
    """聚类算法基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """算法名称"""
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'ClusteringAlgorithm':
        """
        拟合数据
        
        Args:
            X: 特征矩阵 [n_samples, n_features]
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测聚类标签
        
        Args:
            X: 特征矩阵
            
        Returns:
            labels: 聚类标签
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """获取算法参数"""
        pass
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """拟合并预测"""
        return self.fit(X).predict(X)


class ClassificationModel(ABC):
    """分类模型基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """模型名称"""
        pass
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """保存模型"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """加载模型"""
        pass
```

```python
# src/algorithms/clustering/dtw_hierarchical.py

from typing import Dict, Any
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from ..base import ClusteringAlgorithm


class DTWHierarchicalClustering(ClusteringAlgorithm):
    """DTW + 层次聚类"""
    
    def __init__(self, n_clusters: int = 5, method: str = 'ward',
                 dtw_cache_path: str = None):
        self.n_clusters = n_clusters
        self.method = method
        self.dtw_cache_path = dtw_cache_path
        self._labels = None
        self._linkage_matrix = None
    
    @property
    def name(self) -> str:
        return "DTW-Hierarchical"
    
    def fit(self, X: np.ndarray) -> 'DTWHierarchicalClustering':
        """
        Args:
            X: DTW距离矩阵 [n_samples, n_samples]
        """
        # 检查是否为距离矩阵
        if X.shape[0] != X.shape[1]:
            raise ValueError("输入必须是DTW距离矩阵")
        
        condensed = squareform(X)
        self._linkage_matrix = linkage(condensed, method=self.method)
        self._labels = fcluster(
            self._linkage_matrix, 
            self.n_clusters, 
            criterion='maxclust'
        ) - 1
        
        return self
    
    def predict(self, X: np.ndarray = None) -> np.ndarray:
        if self._labels is None:
            raise ValueError("请先调用fit方法")
        return self._labels
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'n_clusters': self.n_clusters,
            'method': self.method
        }
```

#### 4.3.3 应用层 - 服务设计

```python
# src/application/services/clustering_service.py

from typing import List, Dict, Optional, Type
import numpy as np
from pathlib import Path

from ...domain.entities import TighteningCurve, FeatureSet
from ...algorithms.base import ClusteringAlgorithm
from ...algorithms.clustering import (
    KMeansClustering,
    DTWHierarchicalClustering,
    DBSCANClustering,
    GMMClustering
)
from ...infrastructure.parsers import CurveParser
from ...infrastructure.cache import DTWCache


class ClusteringService:
    """聚类分析服务"""
    
    # 注册的算法
    ALGORITHMS = {
        'kmeans': KMeansClustering,
        'dtw_hierarchical': DTWHierarchicalClustering,
        'dbscan': DBSCANClustering,
        'gmm': GMMClustering,
    }
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.dtw_cache = DTWCache(self.cache_dir / "dtw_cache.pkl")
    
    def get_available_algorithms(self) -> List[str]:
        """获取可用算法列表"""
        return list(self.ALGORITHMS.keys())
    
    def cluster(
        self,
        curves: List[TighteningCurve],
        algorithm: str = 'kmeans',
        **kwargs
    ) -> Dict:
        """
        执行聚类分析
        
        Args:
            curves: 曲线列表
            algorithm: 算法名称
            **kwargs: 算法参数
            
        Returns:
            聚类结果
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"未知算法: {algorithm}，可用: {self.get_available_algorithms()}")
        
        # 创建算法实例
        algo_class = self.ALGORITHMS[algorithm]
        algo = algo_class(**kwargs)
        
        # 准备数据
        if algorithm == 'dtw_hierarchical':
            # DTW需要距离矩阵
            X = self._compute_dtw_matrix(curves)
        else:
            # 其他算法需要特征矩阵
            X = self._extract_features(curves)
        
        # 执行聚类
        labels = algo.fit_predict(X)
        
        # 构建结果
        return {
            'algorithm': algo.name,
            'params': algo.get_params(),
            'labels': labels.tolist(),
            'n_clusters': len(set(labels)),
            'curve_ids': [c.id for c in curves],
            'statistics': self._compute_statistics(curves, labels)
        }
    
    def _extract_features(self, curves: List[TighteningCurve]) -> np.ndarray:
        """提取特征矩阵"""
        from ...domain.services import FeatureExtractor
        extractor = FeatureExtractor()
        features = [extractor.extract(c) for c in curves]
        return np.array(features)
    
    def _compute_dtw_matrix(self, curves: List[TighteningCurve]) -> np.ndarray:
        """计算DTW距离矩阵（带缓存）"""
        return self.dtw_cache.compute_or_load(curves)
    
    def _compute_statistics(
        self, 
        curves: List[TighteningCurve], 
        labels: np.ndarray
    ) -> Dict:
        """计算聚类统计信息"""
        stats = {}
        for label in set(labels):
            mask = labels == label
            cluster_curves = [c for c, m in zip(curves, mask) if m]
            
            ok_count = sum(1 for c in cluster_curves if c.report == 'OK')
            nok_count = sum(1 for c in cluster_curves if c.report == 'NOK')
            
            stats[f'cluster_{label}'] = {
                'count': len(cluster_curves),
                'ok_count': ok_count,
                'nok_count': nok_count,
                'nok_ratio': nok_count / len(cluster_curves) if cluster_curves else 0
            }
        
        return stats
```

#### 4.3.4 基础设施层 - 数据解析

```python
# src/infrastructure/parsers/json_parser.py

import json
from typing import List, Iterator
from pathlib import Path

from ...domain.entities import TighteningCurve, CurvePoint


class RavenDBJSONParser:
    """RavenDB API JSON格式解析器"""
    
    def parse_file(self, filepath: str) -> List[TighteningCurve]:
        """解析JSON文件"""
        path = Path(filepath)
        content = path.read_text(encoding='utf-8')
        return list(self._parse_content(content))
    
    def _parse_content(self, content: str) -> Iterator[TighteningCurve]:
        """
        解析JSON内容
        
        支持两种格式：
        1. V1: 多个JSON对象拼接 {"model":...}{"model":...}
        2. V2: JSON数组 [{"model":...},{"model":...}]
        """
        # 尝试V2格式
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for obj in data:
                    curve = self._parse_object(obj)
                    if curve:
                        yield curve
                return
        except json.JSONDecodeError:
            pass
        
        # V1格式解析
        decoder = json.JSONDecoder()
        idx = 0
        
        while idx < len(content):
            while idx < len(content) and content[idx] in ' \t\n\r':
                idx += 1
            if idx >= len(content):
                break
            
            try:
                obj, end = decoder.raw_decode(content, idx)
                curve = self._parse_object(obj)
                if curve:
                    yield curve
                idx = end
            except json.JSONDecodeError:
                idx += 1
    
    def _parse_object(self, obj: dict) -> TighteningCurve:
        """解析单个JSON对象"""
        if 'model' not in obj:
            return None
        
        model = obj['model']
        results = model.get('results', [])
        
        if not results:
            return None
        
        result = results[0]
        curves_data = result.get('curves', [])
        
        if not curves_data:
            return None
        
        # 提取点数据
        points_data = curves_data[0].get('data', {}).get('points', [])
        points = [
            CurvePoint(
                torque=p.get('torque', 0),
                angle=p.get('angle', 0),
                current=p.get('current', 0),
                marker=p.get('marker', 0)
            )
            for p in points_data
        ]
        
        return TighteningCurve(
            id=obj.get('id', ''),
            points=points,
            report=model.get('report', 'UNKNOWN'),
            result_number=model.get('resultNumber', 0),
            station=model.get('stationName', ''),
            tool=model.get('toolName', ''),
            metadata=obj.get('metadata', {})
        )


class ParserFactory:
    """解析器工厂"""
    
    _parsers = {
        'json': RavenDBJSONParser,
        # 'csv': CSVParser,
        # 'pkl': PickleParser,
    }
    
    @classmethod
    def get_parser(cls, format: str):
        """获取解析器"""
        if format not in cls._parsers:
            raise ValueError(f"不支持的格式: {format}")
        return cls._parsers[format]()
    
    @classmethod
    def detect_format(cls, filepath: str) -> str:
        """自动检测文件格式"""
        ext = Path(filepath).suffix.lower()
        format_map = {
            '.json': 'json',
            '.csv': 'csv',
            '.pkl': 'pkl',
            '.pickle': 'pkl'
        }
        return format_map.get(ext, 'unknown')
```

### 4.4 配置管理设计

```yaml
# config.yaml

# 数据配置
data:
  default_curve_length: 500
  channels:
    - torque
    - angle
    - current
    - marker

# 聚类配置
clustering:
  default_algorithm: kmeans
  algorithms:
    kmeans:
      n_clusters: 5
      random_state: 42
    dtw_hierarchical:
      n_clusters: 5
      method: ward
      dtw_window: 0.1
    dbscan:
      eps: 0.5
      min_samples: 5
    gmm:
      n_components: 5

# 模型配置
model:
  default_model: cnn
  models:
    cnn:
      input_channels: 4
      hidden_channels: [32, 64, 128]
    lstm:
      input_size: 4
      hidden_size: 64
      num_layers: 2
    transformer:
      d_model: 128
      nhead: 8
      num_layers: 4

# 训练配置
training:
  epochs: 50
  batch_size: 16
  learning_rate: 0.001
  validation_split: 0.2

# 路径配置
paths:
  data_dir: data
  output_dir: output
  cache_dir: cache
  models_dir: models
  projects_dir: projects

# 离线部署
offline:
  wheel_dir: offline_packages
  log_level: INFO
```

```python
# src/config/settings.py

from pathlib import Path
from typing import Dict, Any
import yaml


class Settings:
    """全局配置管理"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: str = None):
        """加载配置"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config.yaml'
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def data(self) -> Dict:
        return self._config.get('data', {})
    
    @property
    def clustering(self) -> Dict:
        return self._config.get('clustering', {})
    
    @property
    def model(self) -> Dict:
        return self._config.get('model', {})
    
    @property
    def paths(self) -> Dict:
        return self._config.get('paths', {})


# 全局配置实例
settings = Settings()
```

### 4.5 接口设计规范

#### 4.5.1 服务接口

```python
# 统一的服务接口设计

from typing import Protocol, List, Dict, Any
from abc import abstractmethod


class DataService(Protocol):
    """数据服务接口"""
    
    def load(self, path: str, format: str = 'auto') -> List[TighteningCurve]:
        """加载数据"""
        ...
    
    def save(self, curves: List[TighteningCurve], path: str) -> str:
        """保存数据"""
        ...


class AnalysisService(Protocol):
    """分析服务接口"""
    
    def analyze(self, curves: List[TighteningCurve], **kwargs) -> Dict[str, Any]:
        """执行分析"""
        ...
    
    def export(self, results: Dict, output_path: str) -> str:
        """导出结果"""
        ...


class ModelService(Protocol):
    """模型服务接口"""
    
    def train(self, curves: List[TighteningCurve], labels: List[str], **kwargs) -> Dict:
        """训练模型"""
        ...
    
    def predict(self, curves: List[TighteningCurve]) -> List[str]:
        """预测"""
        ...
    
    def save_model(self, path: str) -> str:
        """保存模型"""
        ...
    
    def load_model(self, path: str):
        """加载模型"""
        ...
```

---

## 5. 重构实施计划

### 5.1 阶段规划

```
Phase 1: 基础重构（2周）
├── 统一数据解析模块
├── 抽取公共特征提取
├── 建立配置管理
└── 添加基础测试

Phase 2: 架构优化（2周）
├── 实现分层架构
├── 算法策略模式
├── 服务层重构
└── UI与业务解耦

Phase 3: 功能完善（1周）
├── 完善测试覆盖
├── API文档编写
├── 性能优化
└── 日志系统

Phase 4: 部署优化（1周）
├── 离线包优化
├── 安装脚本完善
├── 用户手册更新
└── 验收测试
```

### 5.2 详细任务分解

#### Phase 1: 基础重构

| 任务ID | 描述 | 优先级 | 工时 | 依赖 |
|--------|------|--------|------|------|
| P1-001 | 创建统一数据解析模块 | P0 | 4h | - |
| P1-002 | 合并特征提取逻辑 | P0 | 4h | P1-001 |
| P1-003 | 建立配置管理框架 | P0 | 2h | - |
| P1-004 | 迁移硬编码路径到配置 | P1 | 2h | P1-003 |
| P1-005 | 创建基础测试框架 | P0 | 2h | - |
| P1-006 | 数据解析单元测试 | P1 | 3h | P1-001, P1-005 |
| P1-007 | 特征提取单元测试 | P1 | 3h | P1-002, P1-005 |

#### Phase 2: 架构优化

| 任务ID | 描述 | 优先级 | 工时 | 依赖 |
|--------|------|--------|------|------|
| P2-001 | 创建领域层实体类 | P0 | 3h | P1-002 |
| P2-002 | 实现算法基类与策略模式 | P0 | 4h | P2-001 |
| P2-003 | 重构聚类算法 | P0 | 6h | P2-002 |
| P2-004 | 重构分类模型 | P0 | 4h | P2-002 |
| P2-005 | 创建服务层 | P0 | 4h | P2-003, P2-004 |
| P2-006 | 重构Streamlit UI | P1 | 6h | P2-005 |
| P2-007 | 集成测试 | P1 | 4h | P2-006 |

#### Phase 3: 功能完善

| 任务ID | 描述 | 优先级 | 工时 | 依赖 |
|--------|------|--------|------|------|
| P3-001 | 提高测试覆盖率至80% | P1 | 6h | P2-007 |
| P3-002 | API文档编写 | P1 | 4h | P2-005 |
| P3-003 | 性能优化（DTW并行等） | P2 | 4h | P2-003 |
| P3-004 | 添加日志系统 | P2 | 2h | - |
| P3-005 | 错误处理完善 | P1 | 3h | - |

#### Phase 4: 部署优化

| 任务ID | 描述 | 优先级 | 工时 | 依赖 |
|--------|------|--------|------|------|
| P4-001 | 离线安装包优化 | P1 | 4h | P3-001 |
| P4-002 | 安装脚本完善 | P1 | 2h | P4-001 |
| P4-003 | 用户手册更新 | P1 | 3h | - |
| P4-004 | 验收测试 | P0 | 4h | P4-001 |
| P4-005 | 发布准备 | P0 | 2h | P4-004 |

### 5.3 风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| 重构引入Bug | 高 | 中 | 充分的单元测试 + 集成测试 |
| 功能回退 | 中 | 低 | 保留旧代码分支，渐进式迁移 |
| 性能下降 | 中 | 低 | 性能基准测试 |
| 用户习惯改变 | 低 | 中 | UI保持一致，提供迁移指南 |

---

## 6. 代码重构示例

### 6.1 数据解析重构

**重构前（分散在多个文件）：**

```python
# dtw_cluster_full.py
def parse_anord_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    parts = content.split('}{')
    curves = []
    for i, part in enumerate(parts):
        # ... 50行实现代码
    return curves

# cluster_analysis.py
def parse_curves(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    parts = content.split('}{')
    curves = []
    for i, part in enumerate(parts):
        # ... 类似但略有不同的实现
    return curves
```

**重构后（统一模块）：**

```python
# src/infrastructure/parsers/json_parser.py

class RavenDBJSONParser:
    """统一的JSON解析器"""
    
    def parse_file(self, filepath: str) -> List[TighteningCurve]:
        """解析JSON文件，返回标准化实体"""
        path = Path(filepath)
        content = path.read_text(encoding='utf-8')
        return list(self._parse_content(content))
    
    def _parse_content(self, content: str) -> Iterator[TighteningCurve]:
        """支持V1/V2两种格式"""
        # ... 统一实现

# 使用方式
from src.infrastructure.parsers import ParserFactory

parser = ParserFactory.get_parser('json')
curves = parser.parse_file('data/Anord.json')
```

### 6.2 聚类算法重构

**重构前（独立函数）：**

```python
# advanced_clustering.py
def dbscan_clustering(curves, output_dir='output'):
    features, valid_curves = extract_features(curves)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    # ... 具体实现

def gmm_clustering(curves, output_dir='output'):
    features, valid_curves = extract_features(curves)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    # ... 类似重复代码
```

**重构后（策略模式）：**

```python
# src/algorithms/clustering/__init__.py

class ClusteringAlgorithm(ABC):
    """聚类算法基类"""
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'ClusteringAlgorithm':
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

class DBSCANClustering(ClusteringAlgorithm):
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self._model = None
    
    def fit(self, X: np.ndarray) -> 'DBSCANClustering':
        self._model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self._labels = self._model.fit_predict(X)
        return self
    
    def predict(self, X: np.ndarray = None) -> np.ndarray:
        return self._labels

# 使用方式
from src.application.services import ClusteringService

service = ClusteringService()
result = service.cluster(curves, algorithm='dbscan', eps=0.5)
```

### 6.3 UI层解耦重构

**重构前（UI直接调用底层）：**

```python
# streamlit_app/pages/0_工程管理.py

from project_manager import ProjectManager

@st.cache_resource
def get_project_manager():
    return ProjectManager(workspace_dir="projects")

# UI中直接处理业务逻辑
if st.button("开始导入"):
    pm.import_data(project_name, str(temp_path), data_format)
```

**重构后（通过服务层）：**

```python
# ui/streamlit_app/pages/project_management.py

from src.application.services import ProjectService
from ui.streamlit_app.components import ProjectList, DataImporter

class ProjectManagementPage:
    """工程管理页面"""
    
    def __init__(self):
        self.project_service = ProjectService()
    
    def render(self):
        st.title("📁 工程管理")
        
        # 渲染统计卡片
        self._render_statistics()
        
        # 渲染工程列表组件
        ProjectList(self.project_service).render()
        
        # 渲染数据导入组件
        DataImporter(self.project_service).render()
    
    def _render_statistics(self):
        stats = self.project_service.get_statistics()
        # ...

# 页面入口
if __name__ == "__main__":
    ProjectManagementPage().render()
```

### 6.4 配置管理重构

**重构前（硬编码）：**

```python
# 多处硬编码
filepath='API/Anord.json'
output_dir='output/advanced_clustering'
target_length=500
n_clusters=5
```

**重构后（配置驱动）：**

```python
# src/config/settings.py

from src.config import settings

# 使用配置
data_path = settings.get('paths.data_dir') / 'Anord.json'
output_dir = settings.get('paths.output_dir') / 'clustering'
curve_length = settings.get('data.default_curve_length')
n_clusters = settings.get('clustering.algorithms.kmeans.n_clusters')
```

---

## 7. 企业内网部署方案

### 7.1 离线部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                    企业内网环境                               │
│                                                             │
│  ┌─────────────────┐     ┌─────────────────┐               │
│  │   用户工作站     │────▶│   应用服务器     │               │
│  │  (浏览器访问)    │     │  (Streamlit)    │               │
│  └─────────────────┘     └────────┬────────┘               │
│                                   │                         │
│                    ┌──────────────┴──────────────┐          │
│                    │                             │          │
│           ┌────────▼────────┐         ┌────────▼────────┐  │
│           │   数据存储       │         │   模型存储       │  │
│           │   (JSON/CSV)    │         │   (.pth files)  │  │
│           └─────────────────┘         └─────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              离线依赖包仓库                            │  │
│  │  offline_packages/                                    │  │
│  │  ├── numpy-2.4.3-py3-none-any.whl                    │  │
│  │  ├── pandas-2.3.3-py3-none-any.whl                   │  │
│  │  ├── torch-2.10.0-cp313-win_amd64.whl                │  │
│  │  └── ... (所有依赖)                                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 安装包结构

```
tightening-ai-offline-v1.1.0.zip
│
├── install.bat                  # 一键安装脚本
├── run.bat                      # 启动脚本
├── train_new_data.bat           # 新数据训练脚本
│
├── src/                         # 源代码
│   ├── application/
│   ├── algorithms/
│   ├── infrastructure/
│   └── config/
│
├── ui/                          # UI代码
│   └── streamlit_app/
│
├── offline_packages/            # 离线依赖包
│   ├── install.bat
│   └── *.whl                   # 所有wheel包
│
├── API/                         # 示例数据
│   └── example_data.json
│
├── models/                      # 预训练模型
│   └── cnn_model.pth
│
├── config.yaml                  # 配置文件
├── requirements.txt             # 依赖清单
└── README.txt                   # 安装说明
```

### 7.3 安装脚本

```batch
@echo off
chcp 65001 >nul
echo ========================================
echo 拧紧曲线智能分析系统 - 离线安装
echo ========================================

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.9+
    pause
    exit /b 1
)

:: 安装离线依赖
echo.
echo [1/3] 安装离线依赖包...
cd offline_packages
call install.bat
cd ..

:: 创建必要目录
echo.
echo [2/3] 创建目录结构...
if not exist "data" mkdir data
if not exist "output" mkdir output
if not exist "projects" mkdir projects
if not exist "cache" mkdir cache

:: 验证安装
echo.
echo [3/3] 验证安装...
python -c "import torch; import streamlit; import numpy; print('所有依赖安装成功!')"

echo.
echo ========================================
echo 安装完成！
echo.
echo 使用方法:
echo   1. 双击 run.bat 启动应用
echo   2. 浏览器访问 http://localhost:8501
echo ========================================
pause
```

### 7.4 运行要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Windows 10 | Windows 10/11 |
| Python | 3.9 | 3.11+ |
| 内存 | 8GB | 16GB+ |
| 磁盘 | 5GB | 10GB+ |
| CPU | 4核 | 8核+ |
| GPU | 无 | CUDA兼容（可选） |

---

## 附录

### A. 测试覆盖要求

```python
# tests/unit/test_parsers.py

import pytest
from src.infrastructure.parsers import RavenDBJSONParser

class TestRavenDBJSONParser:
    
    @pytest.fixture
    def parser(self):
        return RavenDBJSONParser()
    
    def test_parse_v1_format(self, parser, sample_v1_data):
        """测试V1格式解析"""
        curves = parser.parse_file(sample_v1_data)
        assert len(curves) > 0
        assert all(hasattr(c, 'points') for c in curves)
    
    def test_parse_v2_format(self, parser, sample_v2_data):
        """测试V2格式解析"""
        curves = parser.parse_file(sample_v2_data)
        assert len(curves) > 0
    
    def test_parse_empty_file(self, parser, empty_file):
        """测试空文件"""
        curves = parser.parse_file(empty_file)
        assert curves == []
    
    def test_parse_malformed_json(self, parser, malformed_file):
        """测试错误格式"""
        # 应该跳过错误记录，不抛异常
        curves = parser.parse_file(malformed_file)
        assert isinstance(curves, list)
```

### B. API文档模板

```python
def cluster_curves(
    curves: List[TighteningCurve],
    algorithm: str = 'kmeans',
    n_clusters: int = 5,
    **kwargs
) -> ClusteringResult:
    """
    对拧紧曲线执行聚类分析。

    Parameters
    ----------
    curves : List[TighteningCurve]
        要聚类的曲线列表。
    algorithm : str, default='kmeans'
        聚类算法名称。可选值：
        - 'kmeans': K-Means聚类
        - 'dtw_hierarchical': DTW距离+层次聚类
        - 'dbscan': DBSCAN密度聚类
        - 'gmm': 高斯混合模型
    n_clusters : int, default=5
        聚类数量（部分算法适用）。
    **kwargs : dict
        算法特定参数。

    Returns
    -------
    ClusteringResult
        聚类结果，包含标签、统计信息等。

    Raises
    ------
    ValueError
        当algorithm不在支持列表中时。

    Examples
    --------
    >>> from src.application.services import ClusteringService
    >>> service = ClusteringService()
    >>> result = service.cluster(curves, algorithm='dtw_hierarchical', n_clusters=3)
    >>> print(result.labels)
    [0, 1, 0, 2, ...]
    """
```

---

## 总结

本架构设计文档识别了当前系统的主要问题：

1. **代码重复** - 数据解析、特征提取在多处重复实现
2. **职责不清** - 根目录脚本混乱，缺少统一入口
3. **扩展受限** - 缺少算法抽象，新增功能需要大量修改
4. **配置分散** - 硬编码路径和参数

推荐的重构方案：

1. **分层架构** - 表现层、应用层、领域层、基础设施层
2. **策略模式** - 算法可插拔，易于扩展
3. **配置管理** - 统一的YAML配置
4. **统一入口** - 清晰的命令行和Web入口

实施计划分为4个阶段，预计6周完成，优先级清晰，风险可控。

---

*文档结束*