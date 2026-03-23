"""
Dataset Model - with JSON column_names serialization
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import json


# ============ CurveData ============
@dataclass
class CurveData:
    """Curve data for compatibility."""
    curve_id: str = ""
    torque: List[float] = field(default_factory=list)
    angle: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, curve_id: str = "", torque: List[float] = None, angle: List[float] = None, 
                 metadata: Dict[str, Any] = None, **kwargs):
        self.curve_id = curve_id
        self.torque = torque if torque is not None else []
        self.angle = angle if angle is not None else []
        self.metadata = metadata if metadata is not None else {}
        # Compatibility with kwargs
        self.id = kwargs.get('id', curve_id)
    
    @property
    def max_torque(self) -> float:
        return max(self.torque) if self.torque else 0.0
    
    @property
    def max_angle(self) -> float:
        return max(self.angle) if self.angle else 0.0
    
    def to_dict(self) -> dict:
        return {
            "curve_id": self.curve_id,
            "torque": self.torque,
            "angle": self.angle,
            "metadata": self.metadata,
            "max_torque": self.max_torque,
            "max_angle": self.max_angle
        }


# ============ DatasetConfig ============
@dataclass
class DatasetConfig:
    """Dataset configuration for compatibility."""
    id: str = ""
    project_id: str = ""
    name: str = ""
    column_names: List[str] = field(default_factory=list)
    version: str = ""
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    record_count: int = 0
    
    def __init__(self, id: str = "", project_id: str = "", name: str = "", column_names: List[str] = None, **kwargs):
        # 从kwargs中提取可能的参数
        self.id = kwargs.get('id', id)
        self.project_id = kwargs.get('project_id', project_id)
        self.name = kwargs.get('name', name)
        self.column_names = kwargs.get('column_names', column_names) if kwargs.get('column_names') else (column_names if column_names is not None else [])
        # 提取其他可能的参数
        self.version = kwargs.get('version', '')
        self.description = kwargs.get('description', None)
        self.created_at = kwargs.get('created_at', None)
        self.record_count = kwargs.get('record_count', 0)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "name": self.name,
            "column_names": self.column_names,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "record_count": self.record_count
        }


# ============ Dataset (for backward compatibility) ============
@dataclass
class Dataset:
    """Dataset model for backwards compatibility."""
    id: str = ""
    project_id: str = ""
    name: str = ""
    config: Optional[DatasetConfig] = None
    curves: List[CurveData] = field(default_factory=list)
    record_count: int = 0
    
    def __init__(self, id: str = "", project_id: str = "", name: str = "", config: DatasetConfig = None, 
                 curves: List[CurveData] = None, record_count: int = 0, **kwargs):
        self.id = kwargs.get('id', id)
        self.project_id = kwargs.get('project_id', project_id)
        self.name = kwargs.get('name', name)
        self.config = kwargs.get('config', config)
        self.curves = kwargs.get('curves', curves) if kwargs.get('curves') else (curves if curves is not None else [])
        self.record_count = kwargs.get('record_count', record_count)
        
        if self.config is None:
            self.config = DatasetConfig(id=self.id, project_id=self.project_id, name=self.name)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "name": self.name,
            "config": self.config.to_dict() if self.config else {},
            "curves": [c.to_dict() for c in self.curves],
            "record_count": self.record_count
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Dataset":
        config = None
        if data.get('config'):
            config = DatasetConfig(
                id=data['config'].get('id', ''),
                project_id=data['config'].get('project_id', ''),
                name=data['config'].get('name', '')
            )
        curves = []
        for c in data.get('curves', []):
            curves.append(CurveData(
                curve_id=c.get('curve_id', ''),
                torque=c.get('torque', []),
                angle=c.get('angle', [])
            ))
        return cls(
            id=data['id'],
            project_id=data['project_id'],
            name=data['name'],
            config=config,
            curves=curves,
            record_count=data.get('record_count', 0)
        )
    
    def add_curves(self, curves: List[CurveData]):
        """Add curves to dataset."""
        self.curves.extend(curves)
        self.record_count = len(self.curves)
        if self.config:
            self.config.record_count = self.record_count
    
    def get_curve_by_id(self, curve_id: str) -> Optional[CurveData]:
        """Get a curve by ID."""
        for curve in self.curves:
            if curve.curve_id == curve_id:
                return curve
        return None
    
    @property
    def stats(self) -> dict:
        """Get dataset statistics."""
        return {
            "id": self.id,
            "curve_count": len(self.curves),
            "record_count": len(self.curves)
        }


# ============ DatasetRepository ============
class DatasetRepository:
    """Dataset repository for compatibility."""
    def __init__(self, base_path: Path = None, db_manager=None):
        self.base_path = base_path or Path("data/datasets")
        self.db = db_manager
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, dataset: Dataset):
        """Save a dataset."""
        import json
        dataset_path = self.base_path / f"{dataset.id}.json"
        with open(dataset_path, 'w') as f:
            json.dump(dataset.to_dict(), f)
    
    def get(self, dataset_id: str) -> Optional[Dataset]:
        """Get a dataset by ID."""
        import json
        dataset_path = self.base_path / f"{dataset_id}.json"
        if dataset_path.exists():
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            return Dataset.from_dict(data)
        return None
    
    def delete(self, dataset_id: str) -> bool:
        """Delete a dataset."""
        import json
        dataset_path = self.base_path / f"{dataset_id}.json"
        if dataset_path.exists():
            dataset_path.unlink()
            return True
        return False
    
    def get_by_project(self, project_id: str) -> List[Dataset]:
        """Get datasets by project ID."""
        datasets = []
        for dataset_file in self.base_path.glob("*.json"):
            try:
                with open(dataset_file, 'r') as f:
                    data = json.load(f)
                if data.get('project_id') == project_id:
                    datasets.append(Dataset.from_dict(data))
            except:
                continue
        return datasets
