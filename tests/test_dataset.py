"""
Unit tests for Datasets
"""
import pytest
from pathlib import Path
import shutil


@pytest.fixture
def clean_repo():
    """清理测试仓库"""
    test_dir = Path("test_dataset_temp")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(exist_ok=True)
    
    # 清理全局仓库实例
    import src.models.dataset as dataset_module
    if hasattr(dataset_module, '_dataset_repo'):
        dataset_module._dataset_repo = None
    
    yield test_dir
    
    if test_dir.exists():
        shutil.rmtree(test_dir)


def test_dataset_config_creation():
    """测试数据集配置创建"""
    from src.models.dataset import DatasetConfig
    from datetime import datetime
    
    config = DatasetConfig(
        name="Test Dataset",
        version="1.0.0"
    )
    
    assert config.name == "Test Dataset"
    assert config.version == "1.0.0"


def test_curve_data_creation():
    """测试曲线数据创建"""
    from src.models.dataset import CurveData
    
    curve = CurveData(
        curve_id="c1",
        torque=[1.0, 2.0, 3.0],
        angle=[0.0, 1.0, 2.0]
    )
    
    assert curve.curve_id == "c1"
    assert len(curve.torque) == 3
    assert curve.max_torque == 3.0
    assert curve.max_angle == 2.0


def test_dataset_creation():
    """测试数据集创建"""
    from src.models.dataset import Dataset, DatasetConfig, CurveData
    
    config = DatasetConfig(name="Test Dataset")
    dataset = Dataset(
        id="ds_001",
        project_id="proj_001",
        config=config
    )
    
    assert dataset.id == "ds_001"
    assert dataset.project_id == "proj_001"
    assert len(dataset.curves) == 0


def test_dataset_add_curves():
    """测试添加曲线"""
    from src.models.dataset import Dataset, DatasetConfig, CurveData
    
    config = DatasetConfig(name="Test Dataset")
    dataset = Dataset(id="ds_001", project_id="proj_001", config=config)
    
    curve1 = CurveData(
        curve_id="c1",
        torque=[1.0, 2.0],
        angle=[0.0, 1.0]
    )
    curve2 = CurveData(
        curve_id="c2",
        torque=[2.0, 3.0],
        angle=[1.0, 2.0]
    )
    
    dataset.add_curves([curve1, curve2])
    
    assert len(dataset.curves) == 2
    assert dataset.config.record_count == 2


def test_dataset_get_curve_by_id():
    """测试根据ID获取曲线"""
    from src.models.dataset import Dataset, DatasetConfig, CurveData
    
    config = DatasetConfig(name="Test Dataset")
    dataset = Dataset(id="ds_001", project_id="proj_001", config=config)
    
    curve = CurveData(
        curve_id="c1",
        torque=[1.0, 2.0],
        angle=[0.0, 1.0]
    )
    dataset.add_curves([curve])
    
    retrieved = dataset.get_curve_by_id("c1")
    assert retrieved is not None
    assert retrieved.curve_id == "c1"
    
    not_found = dataset.get_curve_by_id("c2")
    assert not_found is None


def test_dataset_stats():
    """测试数据集统计"""
    from src.models.dataset import Dataset, DatasetConfig, CurveData
    
    config = DatasetConfig(name="Test Dataset")
    dataset = Dataset(id="ds_001", project_id="proj_001", config=config)
    
    curve = CurveData(
        curve_id="c1",
        torque=[1.0, 2.0],
        angle=[0.0, 1.0]
    )
    dataset.add_curves([curve])
    
    stats = dataset.stats
    
    assert stats["id"] == "ds_001"
    assert stats["curve_count"] == 1
    assert stats["record_count"] == 1


def test_dataset_repo_save_get(clean_repo):
    """测试数据集仓库保存和获取"""
    from src.models.dataset import Dataset, DatasetConfig, DatasetRepository
    
    repo = DatasetRepository(base_path=clean_repo / "projects")
    
    config = DatasetConfig(name="Test Dataset")
    dataset = Dataset(id="ds_001", project_id="proj_001", config=config)
    
    repo.save(dataset)
    
    retrieved = repo.get("ds_001")
    assert retrieved is not None
    assert retrieved.id == "ds_001"


def test_dataset_repo_get_by_project(clean_repo):
    """测试根据工程获取数据集"""
    from src.models.dataset import Dataset, DatasetConfig, DatasetRepository
    
    repo = DatasetRepository(base_path=clean_repo / "projects")
    
    config = DatasetConfig(name="Test Dataset 1")
    dataset1 = Dataset(id="ds_001", project_id="proj_001", config=config)
    
    config2 = DatasetConfig(name="Test Dataset 2")
    dataset2 = Dataset(id="ds_002", project_id="proj_001", config=config2)
    
    repo.save(dataset1)
    repo.save(dataset2)
    
    datasets = repo.get_by_project("proj_001")
    assert len(datasets) == 2


def test_dataset_repo_delete(clean_repo):
    """测试数据集仓库删除"""
    from src.models.dataset import Dataset, DatasetConfig, DatasetRepository
    
    repo = DatasetRepository(base_path=clean_repo / "projects")
    
    config = DatasetConfig(name="Test Dataset")
    dataset = Dataset(id="ds_001", project_id="proj_001", config=config)
    
    repo.save(dataset)
    assert repo.get("ds_001") is not None
    
    result = repo.delete("ds_001")
    assert result is True
    assert repo.get("ds_001") is None
