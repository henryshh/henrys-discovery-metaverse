"""
Model Module
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import json


@dataclass
class ModelConfig:
    """Model configuration for compatibility."""
    id: str = ""
    name: str = ""
    project_id: str = ""
    model_type: str = ""
    algorithm: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, id: str = "", name: str = "", project_id: str = "", model_type: str = "", 
                 algorithm: str = "", parameters: Dict[str, Any] = None, **kwargs):
        self.id = kwargs.get('id', id)
        self.name = kwargs.get('name', name)
        self.project_id = kwargs.get('project_id', project_id)
        self.model_type = kwargs.get('model_type', model_type)
        self.algorithm = kwargs.get('algorithm', algorithm)
        self.parameters = kwargs.get('parameters', parameters) if kwargs.get('parameters') else (parameters if parameters is not None else {})
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "project_id": self.project_id,
            "model_type": self.model_type,
            "algorithm": self.algorithm,
            "parameters": self.parameters
        }


class Model:
    """Model class for compatibility."""
    def __init__(self, id: str = "", project_id: str = "", config: ModelConfig = None, **kwargs):
        self.id = kwargs.get('id', id)
        self.project_id = kwargs.get('project_id', project_id)
        if config is None:
            config = ModelConfig(**kwargs)
        self.config = config
        self.trained = False
    
    def train(self, data):
        """Train the model."""
        self.trained = True
    
    def predict(self, data):
        """Make predictions."""
        if not self.trained:
            raise ValueError("Model not trained yet")
        return []
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "name": self.config.name,
            "config": self.config.to_dict(),
            "trained": self.trained
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Model":
        config = None
        if data.get('config'):
            config = ModelConfig(
                id=data['config'].get('id', ''),
                name=data['config'].get('name', ''),
                project_id=data['config'].get('project_id', ''),
                model_type=data['config'].get('model_type', ''),
                algorithm=data['config'].get('algorithm', ''),
                parameters=data['config'].get('parameters', {})
            )
        return cls(
            id=data['id'],
            project_id=data['project_id'],
            config=config
        )
    
    @property
    def stats(self) -> dict:
        """Get model statistics."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "name": self.config.name,
            "model_type": self.config.model_type,
            "algorithm": self.config.algorithm
        }


class ModelManager:
    """Model manager for ML model storage and loading."""
    
    def __init__(self, models_path: str = "data/models"):
        self.models_path = models_path
        self._models: Dict[str, Any] = {}
    
    def register_model(self, name: str, model: Any):
        """Register a model in memory."""
        self._models[name] = model
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get a registered model."""
        return self._models.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self._models.keys())
    
    def remove_model(self, name: str) -> bool:
        """Remove a registered model."""
        if name in self._models:
            del self._models[name]
            return True
        return False


class ModelRepository:
    """Model repository for compatibility."""
    def __init__(self, base_path: Path = None, db_manager=None):
        self.base_path = base_path or Path("data/models")
        self.db = db_manager
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, model: Model):
        """Save a model."""
        import json
        model_path = self.base_path / f"{model.id}.json"
        with open(model_path, 'w') as f:
            json.dump(model.to_dict(), f)
    
    def get(self, model_id: str) -> Optional[Model]:
        """Get a model by ID."""
        import json
        model_path = self.base_path / f"{model_id}.json"
        if model_path.exists():
            with open(model_path, 'r') as f:
                data = json.load(f)
            return Model.from_dict(data)
        return None
    
    def delete(self, model_id: str) -> bool:
        """Delete a model."""
        import json
        model_path = self.base_path / f"{model_id}.json"
        if model_path.exists():
            model_path.unlink()
            return True
        return False
    
    def get_by_project(self, project_id: str) -> List[Model]:
        """Get models by project ID."""
        models = []
        for model_file in self.base_path.glob("*.json"):
            try:
                with open(model_file, 'r') as f:
                    data = json.load(f)
                if data.get('project_id') == project_id:
                    models.append(Model.from_dict(data))
            except:
                continue
        return models
