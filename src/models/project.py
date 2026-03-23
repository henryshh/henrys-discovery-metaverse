"""
Project Model
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class ProjectConfig:
    """Project configuration for compatibility."""
    id: str = ""
    name: str = ""
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, id: str = "", name: str = "", description: str = None, metadata: Dict[str, Any] = None, **kwargs):
        self.id = kwargs.get('id', id)
        self.name = kwargs.get('name', name)
        self.description = kwargs.get('description', description)
        self.metadata = kwargs.get('metadata', metadata) if kwargs.get('metadata') else (metadata if metadata is not None else {})
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata
        }


@dataclass
class Project:
    """Project model."""
    id: str = ""
    name: str = ""
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    datasets: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Optional[ProjectConfig] = None
    
    def __init__(self, id: str = "", name: str = "", description: str = None, created_at: datetime = None, 
                 datasets: List[str] = None, metadata: Dict[str, Any] = None, config: ProjectConfig = None, **kwargs):
        # 为了兼容性，从kwargs中提取可能的参数
        self.id = kwargs.get('id', id)
        self.name = kwargs.get('name', name)
        self.description = kwargs.get('description', description)
        self.created_at = kwargs.get('created_at', created_at)
        self.datasets = kwargs.get('datasets', datasets) if kwargs.get('datasets') else (datasets if datasets is not None else [])
        self.metadata = kwargs.get('metadata', metadata) if kwargs.get('metadata') else (metadata if metadata is not None else {})
        self.config = kwargs.get('config', config)
        
        if self.config is None:
            self.config = ProjectConfig(id=self.id, name=self.name, description=self.description, metadata=self.metadata)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "datasets": self.datasets,
            "metadata": self.metadata,
            "config": self.config.to_dict() if self.config else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        """Create from dictionary."""
        created_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])
        
        config = None
        if data.get('config'):
            config = ProjectConfig(
                id=data['config'].get('id', ''),
                name=data['config'].get('name', ''),
                description=data['config'].get('description'),
                metadata=data['config'].get('metadata', {})
            )
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            created_at=created_at,
            datasets=data.get("datasets", []),
            metadata=data.get("metadata", {}),
            config=config
        )
    
    def add_dataset(self, dataset_id: str):
        """Add a dataset to the project."""
        if dataset_id not in self.datasets:
            self.datasets.append(dataset_id)
    
    def remove_dataset(self, dataset_id: str):
        """Remove a dataset from the project."""
        if dataset_id in self.datasets:
            self.datasets.remove(dataset_id)


class ProjectRepository:
    """Project repository for compatibility."""
    def __init__(self, base_path: Path = None, db_manager=None):
        self.base_path = base_path or Path("data/projects")
        self.db = db_manager
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, project: Project):
        """Save a project."""
        import json
        project_path = self.base_path / f"{project.id}.json"
        with open(project_path, 'w') as f:
            json.dump(project.to_dict(), f)
    
    def get(self, project_id: str) -> Optional[Project]:
        """Get a project by ID."""
        import json
        project_path = self.base_path / f"{project_id}.json"
        if project_path.exists():
            with open(project_path, 'r') as f:
                data = json.load(f)
            return Project.from_dict(data)
        return None
    
    def delete(self, project_id: str) -> bool:
        """Delete a project."""
        import json
        project_path = self.base_path / f"{project_id}.json"
        if project_path.exists():
            project_path.unlink()
            return True
        return False
    
    def get_all(self) -> List[Project]:
        """Get all projects."""
        projects = []
        for project_file in self.base_path.glob("*.json"):
            try:
                with open(project_file, 'r') as f:
                    data = json.load(f)
                projects.append(Project.from_dict(data))
            except:
                continue
        return projects


# Global instance
_project_repo: Optional[ProjectRepository] = None


def get_project_repo(base_path: Path = None, db_manager=None) -> ProjectRepository:
    """Get or create the global project repository instance."""
    global _project_repo
    if _project_repo is None:
        _project_repo = ProjectRepository(base_path, db_manager)
    return _project_repo


def clear_project_repo():
    """Clear project repository for testing."""
    global _project_repo
    _project_repo = None
