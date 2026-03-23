"""
Unit tests for Models
"""
import pytest
from pathlib import Path
import shutil


@pytest.fixture
def clean_repo():
    """清理测试仓库"""
    test_dir = Path("test_model_temp")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(exist_ok=True)
    
    # 清理全局仓库实例
    import src.models.model as model_module
    if hasattr(model_module, '_model_repo'):
        model_module._model_repo = None
    
    yield test_dir
    
    if test_dir.exists():
        shutil.rmtree(test_dir)


def test_model_config_creation():
    """测试模型配置创建"""
    from src.models.model import ModelConfig
    from datetime import datetime
    
    config = ModelConfig(
        name="Test Model",
        model_type="clustering",
        algorithm="kmeans"
    )
    
    assert config.name == "Test Model"
    assert config.model_type == "clustering"
    assert config.algorithm == "kmeans"


def test_model_creation():
    """测试模型创建"""
    from src.models.model import Model, ModelConfig
    
    config = ModelConfig(name="Test Model")
    model = Model(
        id="model_001",
        project_id="proj_001",
        config=config
    )
    
    assert model.id == "model_001"
    assert model.project_id == "proj_001"
    assert model.config.name == "Test Model"


def test_model_stats():
    """测试模型统计"""
    from src.models.model import Model, ModelConfig
    
    config = ModelConfig(name="Test Model", algorithm="kmeans")
    model = Model(
        id="model_001",
        project_id="proj_001",
        config=config
    )
    
    stats = model.stats
    
    assert stats["id"] == "model_001"
    assert stats["algorithm"] == "kmeans"


def test_model_repo_save_get(clean_repo):
    """测试模型仓库保存和获取"""
    from src.models.model import Model, ModelConfig, ModelRepository
    
    repo = ModelRepository(base_path=clean_repo / "projects")
    
    config = ModelConfig(name="Test Model")
    model = Model(id="model_001", project_id="proj_001", config=config)
    
    repo.save(model)
    
    retrieved = repo.get("model_001")
    assert retrieved is not None
    assert retrieved.id == "model_001"


def test_model_repo_get_by_project(clean_repo):
    """测试根据工程获取模型"""
    from src.models.model import Model, ModelConfig, ModelRepository
    
    repo = ModelRepository(base_path=clean_repo / "projects")
    
    config = ModelConfig(name="Model 1")
    model1 = Model(id="model_001", project_id="proj_001", config=config)
    
    config2 = ModelConfig(name="Model 2")
    model2 = Model(id="model_002", project_id="proj_001", config=config2)
    
    repo.save(model1)
    repo.save(model2)
    
    models = repo.get_by_project("proj_001")
    assert len(models) == 2


def test_model_repo_delete(clean_repo):
    """测试模型仓库删除"""
    from src.models.model import Model, ModelConfig, ModelRepository
    
    repo = ModelRepository(base_path=clean_repo / "projects")
    
    config = ModelConfig(name="Test Model")
    model = Model(id="model_001", project_id="proj_001", config=config)
    
    repo.save(model)
    assert repo.get("model_001") is not None
    
    result = repo.delete("model_001")
    assert result is True
    assert repo.get("model_001") is None
