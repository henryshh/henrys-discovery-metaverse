"""
Storage Manager - Local file storage with 100MB limit
"""
import os
import hashlib
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any


class StorageManager:
    """Local file storage manager with size limits."""
    
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    def __init__(self, base_path: str = "data/uploads", base_dir: str = None):
        # 支持base_dir参数作为别名
        if base_dir is not None:
            base_path = base_dir
        self.base_path = Path(base_path)
        self.base_dir = self.base_path  # 添加此属性以支持旧版API
        self.datasets_dir = self.base_path / "datasets"
        self.vectors_dir = self.base_path / "vectors"
        self.uploads_dir = self.base_path / "uploads"
        self.projects_path = self.base_path / "projects"
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create necessary directories."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.projects_path.mkdir(parents=True, exist_ok=True)
    
    def _get_project_path(self, project_id: str) -> Path:
        """Get project-specific upload path."""
        return self.projects_path / project_id
    
    def _generate_file_hash(self, file_path: str) -> str:
        """Generate SHA256 hash for a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def save_file(self, project_id: str, file_path: str, filename: Optional[str] = None) -> str:
        """
        Save a file to project-specific directory with size validation.
        
        Args:
            project_id: Project identifier
            file_path: Source file path
            filename: Custom filename (optional, defaults to original)
        
        Returns:
            Relative path to saved file
        
        Raises:
            ValueError: If file exceeds 100MB limit
            FileNotFoundError: If source file doesn't exist
        """
        # ✅ 关键：文件大小检查
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {file_size} bytes (max: {self.MAX_FILE_SIZE})"
            )
        
        # Create project directory if needed
        project_path = self._get_project_path(project_id)
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename with hash
        original_name = os.path.basename(file_path)
        file_hash = self._generate_file_hash(file_path)
        
        if filename:
            name_parts = os.path.splitext(filename)
            safe_name = f"{name_parts[0]}_{file_hash}{name_parts[1]}"
        else:
            name_parts = os.path.splitext(original_name)
            safe_name = f"{name_parts[0]}_{file_hash}{name_parts[1]}"
        
        # Save file
        dest_path = project_path / safe_name
        shutil.copy2(file_path, dest_path)
        
        return str(dest_path.relative_to(self.base_path))
    
    def load_file(self, relative_path: str) -> bytes:
        """Load a file from storage."""
        full_path = self.base_path / relative_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {relative_path}")
        
        with open(full_path, "rb") as f:
            return f.read()
    
    def delete_file(self, relative_path: str) -> bool:
        """Delete a file from storage."""
        full_path = self.base_path / relative_path
        
        if full_path.exists():
            full_path.unlink()
            return True
        return False
    
    def get_file_info(self, relative_path: str) -> Optional[Dict[str, Any]]:
        """Get file information."""
        full_path = self.base_path / relative_path
        
        if not full_path.exists():
            return None
        
        stat = full_path.stat()
        return {
            "path": str(relative_path),
            "size": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime
        }
    
    def get_project_files(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all files for a project."""
        project_path = self._get_project_path(project_id)
        
        if not project_path.exists():
            return []
        
        files = []
        for file_path in project_path.iterdir():
            stat = file_path.stat()
            files.append({
                "name": file_path.name,
                "path": str(file_path.relative_to(self.base_path)),
                "size": stat.st_size,
                "created": stat.st_ctime
            })
        
        return sorted(files, key=lambda x: x["created"], reverse=True)
    
    def cleanup_old_files(self, days: int = 30):
        """Delete files older than specified days."""
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 60 * 60)
        
        for project_path in self.projects_path.iterdir():
            if project_path.is_dir():
                for file_path in project_path.iterdir():
                    if file_path.is_file():
                        if file_path.stat().st_mtime < cutoff_time:
                            try:
                                file_path.unlink()
                            except Exception:
                                pass

    def save_dataset(self, project_id: str, file: Path) -> str:
        """Save dataset file for a project."""
        project_path = self.projects_path / project_id
        project_path.mkdir(parents=True, exist_ok=True)
        
        # 生成唯一文件名
        import hashlib
        with open(file, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        dest_filename = f"{file.stem}_{file_hash}{file.suffix}"
        dest_path = project_path / dest_filename
        
        # 复制文件
        import shutil
        shutil.copy2(file, dest_path)
        
        return dest_path.name

    def get_dataset_path(self, project_id: str, dataset_id: str) -> Path:
        """Get the path for a specific dataset."""
        return self.projects_path / project_id / dataset_id

    def get_dataset_info(self, project_id: str, dataset_id: str):
        """Get dataset information."""
        dataset_path = self.get_dataset_path(project_id, dataset_id)
        if dataset_path.exists():
            stat = dataset_path.stat()
            return {
                "name": dataset_path.name,
                "size": stat.st_size,
                "modified": stat.st_mtime
            }
        return None

    def list_datasets(self, project_id: str):
        """List all datasets for a project."""
        project_path = self.projects_path / project_id
        if not project_path.exists():
            return []
        
        datasets = []
        for file_path in project_path.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                datasets.append({
                    "id": file_path.name,
                    "name": file_path.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
        return datasets

    def delete_dataset(self, project_id: str, dataset_id: str) -> bool:
        """Delete a dataset."""
        dataset_path = self.get_dataset_path(project_id, dataset_id)
        if dataset_path.exists():
            try:
                dataset_path.unlink()
                return True
            except Exception:
                return False
        return False

    def save_curves_json(self, project_id: str, dataset_id: str, curves_data: list):
        """Save curves data as JSON."""
        dataset_path = self.get_dataset_path(project_id, dataset_id)
        curves_path = dataset_path.with_suffix('.curves.json')
        
        import json
        with open(curves_path, 'w') as f:
            json.dump(curves_data, f)

    def load_curves_json(self, project_id: str, dataset_id: str) -> list:
        """Load curves data from JSON."""
        dataset_path = self.get_dataset_path(project_id, dataset_id)
        curves_path = dataset_path.with_suffix('.curves.json')
        
        import json
        if curves_path.exists():
            with open(curves_path, 'r') as f:
                return json.load(f)
        return []


# 兼容性别名
LocalStorage = StorageManager


# 为兼容性添加一些类
class DatasetConfig:
    """Dataset configuration for compatibility."""
    pass


class CurveData:
    """Curve data for compatibility."""
    pass


class ProjectConfig:
    """Project configuration for compatibility."""
    pass


class ModelConfig:
    """Model configuration for compatibility."""
    pass


class Model:
    """Model class for compatibility."""
    pass


class ProjectRepository:
    """Project repository for compatibility."""
    pass
