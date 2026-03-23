"""
Unit tests for Core modules
"""
import pytest
from pathlib import Path
import shutil
import json


@pytest.fixture
def clean_test_dir():
    """清理测试目录"""
    test_dir = Path("test_core_temp")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(exist_ok=True)
    
    yield test_dir
    
    if test_dir.exists():
        shutil.rmtree(test_dir)


def test_database_manager_creation(clean_test_dir):
    """测试数据库管理器创建"""
    from src.core.database import DatabaseManager
    
    db_manager = DatabaseManager(data_dir=clean_test_dir)
    
    assert db_manager.data_dir == clean_test_dir
    assert clean_test_dir.exists()


def test_database_get_project_db(clean_test_dir):
    """测试获取工程数据库"""
    from src.core.database import DatabaseManager
    
    db_manager = DatabaseManager(data_dir=clean_test_dir)
    
    conn = db_manager.get_project_db("test_db")
    
    assert conn is not None
    conn.close()


def test_database_init_tables(clean_test_dir):
    """测试初始化表"""
    from src.core.database import DatabaseManager
    
    db_manager = DatabaseManager(data_dir=clean_test_dir)
    
    # 创建连接会自动初始化表
    conn = db_manager.get_project_db("init_test")
    cursor = conn.cursor()
    
    # 检查表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    assert "projects" in tables
    assert "datasets" in tables
    assert "curves" in tables
    assert "models" in tables
    
    conn.close()


def test_database_saveProject(clean_test_dir):
    """测试保存工程"""
    from src.core.database import DatabaseManager
    
    db_manager = DatabaseManager(data_dir=clean_test_dir)
    
    db_manager.save_project(
        project_id="save_test",
        name="Test Project",
        description="Test"
    )
    
    conn = db_manager.get_project_db("save_test")
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM projects WHERE id='save_test'")
    row = cursor.fetchone()
    
    assert row is not None
    assert row[0] == "Test Project"
    
    conn.close()


def test_storage_creation(clean_test_dir):
    """测试本地存储创建"""
    from src.core.storage import LocalStorage
    
    storage = LocalStorage(base_dir=clean_test_dir)
    
    assert storage.base_dir == clean_test_dir
    assert storage.datasets_dir.exists()
    assert storage.vectors_dir.exists()
    assert storage.uploads_dir.exists()


def test_storage_saveDataset(clean_test_dir):
    """测试保存数据集文件"""
    from src.core.storage import LocalStorage
    
    storage = LocalStorage(base_dir=clean_test_dir)
    
    # 创建测试文件
    test_file = clean_test_dir / "test_data.json"
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


def test_storage_file_size_limit(clean_test_dir):
    """测试文件大小限制"""
    from src.core.storage import LocalStorage
    
    storage = LocalStorage(base_dir=clean_test_dir)
    
    # 创建测试文件（150MB）
    test_file = clean_test_dir / "test_large.json"
    with open(test_file, 'wb') as f:
        f.write(b'\0' * (150 * 1024 * 1024))
    
    # 应该抛出异常
    import pytest
    with pytest.raises(ValueError, match="File too large"):
        storage.save_dataset(
            project_id="size_test",
            file=test_file
        )
    
    # 清理大文件
    import os
    if test_file.exists():
        os.remove(test_file)


def test_storage_saveCurves(clean_test_dir):
    """测试保存曲线数据"""
    from src.core.storage import LocalStorage
    
    storage = LocalStorage(base_dir=clean_test_dir)
    
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


def test_cache_manager_creation(clean_test_dir):
    """测试缓存管理器创建"""
    from src.core.cache import CacheManager
    
    cache = CacheManager(cache_dir=clean_test_dir / "cache")
    
    assert cache.cache_dir.exists()
    assert cache.cache is not None


def test_cache_manager_setGet(clean_test_dir):
    """测试缓存设置和获取"""
    from src.core.cache import CacheManager
    
    cache = CacheManager(cache_dir=clean_test_dir / "cache")
    
    # 设置值
    result = cache.set("test_key", {"data": "test_value"})
    assert result is True
    
    # 获取值
    value = cache.get("test_key")
    assert value is not None
    assert value["data"] == "test_value"


def test_vector_store_creation(clean_test_dir):
    """测试向量存储创建"""
    try:
        from src.core.vector_store import VectorStore
    except ImportError:
        pytest.skip("FAISS not available, skipping vector store tests")
    
    try:
        store = VectorStore(
            project_id="vector_test",
            dimension=4,
            data_dir=clean_test_dir
        )
        
        assert store.project_id == "vector_test"
        assert store.dimension == 4
        assert clean_test_dir.exists()
    except NotImplementedError as e:
        pytest.skip(f"FAISS not available: {e}")


def test_vector_store_addSearch(clean_test_dir):
    """测试向量添加和搜索"""
    try:
        from src.core.vector_store import VectorStore
        import numpy as np
    except ImportError:
        pytest.skip("FAISS not available, skipping vector store tests")
    
    try:
        store = VectorStore(
            project_id="vector_search",
            dimension=4,
            data_dir=clean_test_dir
        )
        
        # 添加一些向量
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        ids = ["v1", "v2", "v3"]
        
        store.add_vectors(vectors, ids)
        
        # 搜索
        query = np.array([1.0, 0.1, 0.1, 0.1])
        results = store.search(query, k=2)
        
        assert len(results) >= 1
        assert results[0].id == "v1"
    except NotImplementedError as e:
        pytest.skip(f"FAISS not available: {e}")


def test_vector_store_saveLoad(clean_test_dir):
    """测试向量存储保存和加载"""
    try:
        from src.core.vector_store import VectorStore
        import numpy as np
    except ImportError:
        pytest.skip("FAISS not available, skipping vector store tests")
    
    try:
        store = VectorStore(
            project_id="vector_io",
            dimension=3,
            data_dir=clean_test_dir
        )
        
        # 添加向量
        vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        ids = ["a", "b"]
        store.add_vectors(vectors, ids)
        
        # 保存
        store.save()
        
        # 创建新实例并加载
        new_store = VectorStore(
            project_id="vector_io",
            dimension=3,
            data_dir=clean_test_dir
        )
        
        assert new_store.load()
        
        # 搜索验证
        results = new_store.search(np.array([1.0, 0.0, 0.0]), k=1)
        assert len(results) >= 1
    except NotImplementedError as e:
        pytest.skip(f"FAISS not available: {e}")
