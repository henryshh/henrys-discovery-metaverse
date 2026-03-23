"""
Unit tests for Vector Store
"""
import pytest
from pathlib import Path
import shutil


@pytest.fixture
def clean_test_dir():
    """清理测试目录"""
    test_dir = Path("test_vector_temp")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(exist_ok=True)
    
    yield test_dir
    
    if test_dir.exists():
        shutil.rmtree(test_dir)


def test_vector_store_creation(clean_test_dir):
    """测试向量存储创建"""
    try:
        from src.core.vector_store import VectorStore
    except ImportError:
        pytest.skip("FAISS not available, skipping vector store tests")
    
    store = VectorStore(
        project_id="vector_test",
        dimension=4,
        data_dir=clean_test_dir
    )
    
    assert store.project_id == "vector_test"
    assert store.dimension == 4
    assert clean_test_dir.exists()


def test_vector_store_add_vectors(clean_test_dir):
    """测试向量添加"""
    try:
        from src.core.vector_store import VectorStore
        import numpy as np
    except ImportError:
        pytest.skip("FAISS not available, skipping vector store tests")
    
    store = VectorStore(
        project_id="vector_add",
        dimension=3,
        data_dir=clean_test_dir
    )
    
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    ids = ["v1", "v2", "v3"]
    
    store.add_vectors(vectors, ids)
    
    assert store.get_vector_count() == 3


def test_vector_store_search(clean_test_dir):
    """测试向量搜索"""
    try:
        from src.core.vector_store import VectorStore
        import numpy as np
    except ImportError:
        pytest.skip("FAISS not available, skipping vector store tests")
    
    store = VectorStore(
        project_id="vector_search",
        dimension=3,
        data_dir=clean_test_dir
    )
    
    # 添加向量
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    ids = ["v1", "v2", "v3"]
    store.add_vectors(vectors, ids)
    
    # 搜索
    query = np.array([1.0, 0.1, 0.1])
    results = store.search(query, k=2)
    
    assert len(results) >= 1
    assert results[0].id == "v1"


def test_vector_store_save_load(clean_test_dir):
    """测试向量存储保存和加载"""
    try:
        from src.core.vector_store import VectorStore
        import numpy as np
    except ImportError:
        pytest.skip("FAISS not available, skipping vector store tests")
    
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
    
    assert new_store.load() is True
    
    # 搜索验证
    results = new_store.search(np.array([1.0, 0.0, 0.0]), k=1)
    assert len(results) >= 1
    assert results[0].id == "a"


def test_vector_store_clear(clean_test_dir):
    """测试向量存储清空"""
    try:
        from src.core.vector_store import VectorStore
        import numpy as np
    except ImportError:
        pytest.skip("FAISS not available, skipping vector store tests")
    
    store = VectorStore(
        project_id="vector_clear",
        dimension=3,
        data_dir=clean_test_dir
    )
    
    # 添加向量
    vectors = np.array([[1.0, 0.0, 0.0]])
    ids = ["v1"]
    store.add_vectors(vectors, ids)
    
    assert store.get_vector_count() == 1
    
    # 清空
    store.clear()
    
    assert store.get_vector_count() == 0
