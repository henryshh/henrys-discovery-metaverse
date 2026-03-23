"""
Unit tests for Projects
"""
import pytest
from pathlib import Path
import shutil


@pytest.fixture
def clean_repo():
    """清理测试仓库"""
    test_dir = Path("test_project_temp")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(exist_ok=True)
    
    # 清理全局仓库实例
    import src.models.project as project_module
    project_module._project_repo = None
    
    yield test_dir
    
    if test_dir.exists():
        shutil.rmtree(test_dir)


def test_project_config_creation():
    """测试工程配置创建"""
    from src.models.project import ProjectConfig
    from datetime import datetime
    
    config = ProjectConfig(
        name="Test Project",
        description="Test description"
    )
    
    assert config.name == "Test Project"
    assert config.description == "Test description"


def test_project_creation():
    """测试工程创建"""
    from src.models.project import Project, ProjectConfig
    
    config = ProjectConfig(name="Test Project")
    project = Project(id="test_001", config=config)
    
    assert project.id == "test_001"
    assert project.config.name == "Test Project"
    assert len(project.datasets) == 0


def test_project_add_dataset():
    """测试添加数据集"""
    from src.models.project import Project, ProjectConfig
    
    config = ProjectConfig(name="Test Project")
    project = Project(id="test_001", config=config)
    
    project.add_dataset("ds_001")
    assert "ds_001" in project.datasets
    
    project.add_dataset("ds_002")
    assert len(project.datasets) == 2


def test_project_remove_dataset():
    """测试移除数据集"""
    from src.models.project import Project, ProjectConfig
    
    config = ProjectConfig(name="Test Project")
    project = Project(id="test_001", config=config)
    
    project.add_dataset("ds_001")
    project.add_dataset("ds_002")
    
    project.remove_dataset("ds_001")
    assert "ds_001" not in project.datasets
    assert "ds_002" in project.datasets


def test_project_repo_save_get(clean_repo):
    """测试工程仓库保存和获取"""
    from src.models.project import Project, ProjectConfig, ProjectRepository
    
    repo = ProjectRepository(base_path=clean_repo / "projects")
    
    config = ProjectConfig(name="Test Project")
    project = Project(id="repo_001", config=config)
    
    repo.save(project)
    
    retrieved = repo.get("repo_001")
    assert retrieved is not None
    assert retrieved.id == "repo_001"


def test_project_repo_delete(clean_repo):
    """测试工程仓库删除"""
    from src.models.project import Project, ProjectConfig, ProjectRepository
    
    repo = ProjectRepository(base_path=clean_repo / "projects")
    
    config = ProjectConfig(name="Test Project")
    project = Project(id="delete_001", config=config)
    
    repo.save(project)
    assert repo.get("delete_001") is not None
    
    result = repo.delete("delete_001")
    assert result is True
    assert repo.get("delete_001") is None
