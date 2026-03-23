"""
Unit tests for Services
"""
import pytest
import tempfile
import os
import shutil
import json


@pytest.fixture
def clean_repo():
    """清理测试仓库"""
    test_dir = tempfile.mkdtemp(prefix="test_service_")
    
    yield test_dir
    
    if os.path.exists(test_dir):
        try:
            shutil.rmtree(test_dir)
        except PermissionError:
            pass


def test_project_service_create(clean_repo):
    """测试项目服务创建"""
    from src.services.project_service import ProjectService
    
    project_service = ProjectService()
    
    project = project_service.create_project(
        name="Test Project",
        description="Test description"
    )
    
    assert project is not None
    assert project.name == "Test Project"


def test_project_service_get(clean_repo):
    """测试项目服务获取"""
    from src.services.project_service import ProjectService
    
    project_service = ProjectService()
    
    # 创建项目
    project = project_service.create_project(
        name="Test Project",
        description="Test description"
    )
    
    # 获取项目
    retrieved = project_service.get_project(project.id)
    assert retrieved is not None
    assert retrieved.name == "Test Project"


def test_project_service_update(clean_repo):
    """测试项目服务更新"""
    from src.services.project_service import ProjectService
    
    project_service = ProjectService()
    
    # 创建项目
    project = project_service.create_project(
        name="Test Project",
        description="Original description"
    )
    
    # 更新项目
    updated = project_service.update_project(
        project_id=project.id,
        name="Updated Project",
        description="Updated description"
    )
    
    assert updated is not None
    assert updated.name == "Updated Project"
    assert updated.description == "Updated description"


def test_project_service_delete(clean_repo):
    """测试项目服务删除"""
    from src.services.project_service import ProjectService
    
    project_service = ProjectService()
    
    # 创建项目
    project = project_service.create_project(
        name="Test Project",
        description="To be deleted"
    )
    
    # 删除项目
    result = project_service.delete_project(project.id)
    assert result is True
    
    # 确认已删除
    retrieved = project_service.get_project(project.id)
    assert retrieved is None


def test_dataset_service_import(clean_repo):
    """测试数据集服务导入"""
    from src.services.dataset_service import DatasetService
    
    dataset_service = DatasetService()
    
    # 创建数据集
    dataset_id = dataset_service.create_dataset(
        project_id="proj_001",
        name="Test Dataset",
        column_names=["col1", "col2"]
    )
    
    assert dataset_id is not None
    
    # 获取数据集
    dataset = dataset_service.get_dataset(dataset_id)
    assert dataset is not None
    assert dataset["name"] == "Test Dataset"


def test_dataset_service_get_stats(clean_repo):
    """测试数据集服务统计"""
    from src.services.dataset_service import DatasetService
    
    dataset_service = DatasetService()
    
    # 创建数据集
    dataset_id = dataset_service.create_dataset(
        project_id="proj_001",
        name="Test Dataset",
        column_names=["col1", "col2", "col3"]
    )
    
    # 获取统计
    stats = dataset_service.get_dataset_statistics(dataset_id)
    assert stats is not None
    assert stats["column_count"] == 3
