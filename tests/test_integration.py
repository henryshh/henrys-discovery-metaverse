"""
Integration tests for tightening-ai
"""
import pytest
import tempfile
import shutil
import json
import os
import sys


@pytest.fixture
def clean_workspace():
    """清理测试工作区"""
    test_dir = tempfile.mkdtemp(prefix="test_integration_")
    os.chdir(test_dir)
    
    # 清理全局数据库管理器
    from src.core.database import reset_db_manager
    reset_db_manager()
    
    yield test_dir
    
    # 清理
    os.chdir("..")
    if os.path.exists(test_dir):
        try:
            shutil.rmtree(test_dir)
        except PermissionError:
            pass


def test_full_project_workflow(clean_workspace):
    """测试完整工程工作流"""
    from src.services.project_service import ProjectService
    from src.services.dataset_service import DatasetService
    from src.models.dataset import Dataset, CurveData
    
    project_service = ProjectService()
    
    # 创建工程
    project = project_service.create_project(
        name="Workflow Test"
    )
    assert project is not None
    assert project.name == "Workflow Test"
    
    # 创建数据集
    dataset_service = DatasetService()
    dataset_id = dataset_service.create_dataset(
        project_id=project.id,
        name="Workflow Data",
        column_names=["id", "value"]
    )
    assert dataset_id is not None
    
    # 获取数据集
    dataset = dataset_service.get_dataset(dataset_id)
    assert dataset is not None
    assert dataset["name"] == "Workflow Data"


def test_dataset_crud(clean_workspace):
    """测试数据集 CRUD"""
    from src.services.dataset_service import DatasetService
    
    dataset_service = DatasetService()
    
    # 创建数据集
    dataset_id = dataset_service.create_dataset(
        project_id="proj_001",
        name="Test Dataset",
        column_names=["col1", "col2"]
    )
    assert dataset_id is not None
    
    # 读取数据集
    dataset = dataset_service.get_dataset(dataset_id)
    assert dataset is not None
    assert dataset["name"] == "Test Dataset"
    
    # 更新数据集
    updated = dataset_service.update_dataset(
        dataset_id=dataset_id,
        name="Updated Dataset"
    )
    assert updated is not None
    assert updated["name"] == "Updated Dataset"
    
    # 删除数据集
    result = dataset_service.delete_dataset(dataset_id)
    assert result is True


def test_project_dataset_relationship(clean_workspace):
    """测试工程和数据集关系"""
    from src.services.project_service import ProjectService
    from src.services.dataset_service import DatasetService
    
    project_service = ProjectService()
    dataset_service = DatasetService()
    
    # 创建工程
    project = project_service.create_project(
        name="Test Project"
    )
    assert project is not None
    
    # 创建多个数据集
    dataset_id1 = dataset_service.create_dataset(
        project_id=project.id,
        name="Dataset 1",
        column_names=["col1"]
    )
    dataset_id2 = dataset_service.create_dataset(
        project_id=project.id,
        name="Dataset 2",
        column_names=["col2"]
    )
    assert dataset_id1 is not None
    assert dataset_id2 is not None
    
    # 获取数据集列表
    datasets = dataset_service.list_datasets(project_id=project.id)
    assert len(datasets) >= 2


def test_get_all_projects(clean_workspace):
    """测试获取所有工程"""
    from src.services.project_service import ProjectService
    
    project_service = ProjectService()
    
    # 创建多个工程
    project1 = project_service.create_project(
        name="Project 1"
    )
    project2 = project_service.create_project(
        name="Project 2"
    )
    
    # 获取工程列表
    projects = project_service.list_projects()
    assert len(projects) >= 2


def test_save_load_dataset(clean_workspace):
    """测试保存和加载数据集"""
    from src.services.dataset_service import DatasetService
    from src.models.dataset import Dataset, CurveData
    
    dataset_service = DatasetService()
    
    # 创建数据集
    dataset_id = dataset_service.create_dataset(
        project_id="proj_001",
        name="Saved Dataset",
        column_names=["id", "value"]
    )
    
    # 获取数据集
    dataset = dataset_service.get_dataset(dataset_id)
    assert dataset is not None
    assert dataset["name"] == "Saved Dataset"
