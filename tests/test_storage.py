"""
Unit tests for Storage
"""
import pytest
from pathlib import Path
import shutil
import json
import os


@pytest.fixture
def clean_storage():
    """清理测试存储"""
    test_dir = Path("test_storage_temp")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(exist_ok=True)
    
    yield test_dir
    
    if test_dir.exists():
        shutil.rmtree(test_dir)


def test_local_storage_creation(clean_storage):
    """测试本地存储创建"""
    from src.core.storage import LocalStorage
    
    storage = LocalStorage(base_dir=clean_storage)
    
    assert storage.base_dir == clean_storage
    assert storage.datasets_dir.exists()
    assert storage.vectors_dir.exists()
    assert storage.uploads_dir.exists()


def test_local_storage_save_dataset(clean_storage):
    """测试保存数据集文件"""
    from src.core.storage import LocalStorage
    
    storage = LocalStorage(base_dir=clean_storage)
    
    # 创建测试文件
    test_file = clean_storage / "test_data.json"
    with open(test_file, 'w') as f:
        json.dump({"data": "test"}, f)
    
    # 保存
    dataset_id = storage.save_dataset(
        project_id="storage_test",
        file=test_file
    )
    
    assert dataset_id is not None
    
    # 获取路径
    dataset_path = storage.get_dataset_path("storage_test", dataset_id)
    assert dataset_path.exists()


def test_local_storage_get_dataset_info(clean_storage):
    """测试获取数据集信息"""
    from src.core.storage import LocalStorage
    
    storage = LocalStorage(base_dir=clean_storage)
    
    # 创建测试文件
    test_file = clean_storage / "test_data.json"
    with open(test_file, 'w') as f:
        json.dump({"data": "test"}, f)
    
    # 保存
    dataset_id = storage.save_dataset(
        project_id="info_test",
        file=test_file
    )
    
    info = storage.get_dataset_info("info_test", dataset_id)
    assert info is not None
    assert info["project_id"] == "info_test"
    assert info["dataset_id"] == dataset_id


def test_local_storage_list_datasets(clean_storage):
    """测试列出数据集"""
    from src.core.storage import LocalStorage
    
    storage = LocalStorage(base_dir=clean_storage)
    
    # 创建测试文件
    test_file1 = clean_storage / "test1.json"
    with open(test_file1, 'w') as f:
        json.dump({"data": "test1"}, f)
    
    test_file2 = clean_storage / "test2.json"
    with open(test_file2, 'w') as f:
        json.dump({"data": "test2"}, f)
    
    # 保存
    dataset_id1 = storage.save_dataset(
        project_id="list_test",
        file=test_file1
    )
    dataset_id2 = storage.save_dataset(
        project_id="list_test",
        file=test_file2
    )
    
    datasets = storage.list_datasets("list_test")
    assert len(datasets) >= 1


def test_local_storage_save_curves(clean_storage):
    """测试保存曲线数据"""
    from src.core.storage import LocalStorage
    
    storage = LocalStorage(base_dir=clean_storage)
    
    curves_data = [
        {
            "curve_id": "c1",
            "torque": [1.0, 2.0, 3.0],
            "angle": [0.0, 1.0, 2.0]
        }
    ]
    
    storage.save_curves_json(
        project_id="curve_test",
        dataset_id="curve_ds",
        curves_data=curves_data
    )
    
    # 加载验证
    loaded = storage.load_curves_json("curve_test", "curve_ds")
    assert len(loaded) == 1
    assert loaded[0]["curve_id"] == "c1"


def test_local_storage_delete_dataset(clean_storage):
    """测试删除数据集"""
    from src.core.storage import LocalStorage
    
    storage = LocalStorage(base_dir=clean_storage)
    
    # 创建测试文件
    test_file = clean_storage / "test_data.json"
    with open(test_file, 'w') as f:
        json.dump({"data": "test"}, f)
    
    # 保存
    dataset_id = storage.save_dataset(
        project_id="delete_test",
        file=test_file
    )
    
    dataset_path = storage.get_dataset_path("delete_test", dataset_id)
    assert dataset_path.exists()
    
    result = storage.delete_dataset("delete_test", dataset_id)
    assert result is True
    assert not dataset_path.exists()


def test_local_storage_file_size_limit(clean_storage):
    """测试文件大小限制"""
    from src.core.storage import LocalStorage
    
    storage = LocalStorage(base_dir=clean_storage)
    
    # 创建测试文件（150MB）
    test_file = clean_storage / "test_large.json"
    with open(test_file, 'wb') as f:
        f.write(b'\0' * (150 * 1024 * 1024))
    
    # 应该抛出异常
    with pytest.raises(ValueError, match="File too large"):
        storage.save_dataset(
            project_id="size_test",
            file=test_file
        )
    
    # 清理大文件
    if test_file.exists():
        os.remove(test_file)
