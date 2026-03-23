"""
Unit tests for Cache
"""
import pytest
from pathlib import Path
import shutil


@pytest.fixture
def clean_cache():
    """清理测试缓存"""
    test_dir = Path("test_cache_temp")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(exist_ok=True)
    
    yield test_dir
    
    if test_dir.exists():
        shutil.rmtree(test_dir)


def test_cache_manager_creation(clean_cache):
    """测试缓存管理器创建"""
    from src.core.cache import CacheManager
    
    cache = CacheManager(cache_dir=clean_cache / "cache")
    
    assert cache.cache_dir.exists()
    assert cache.cache is not None


def test_cache_set_get(clean_cache):
    """测试缓存设置和获取"""
    from src.core.cache import CacheManager
    
    cache = CacheManager(cache_dir=clean_cache / "cache")
    
    # 设置值
    result = cache.set("test_key", {"data": "test_value"})
    assert result is True
    
    # 获取值
    value = cache.get("test_key")
    assert value is not None
    assert value["data"] == "test_value"


def test_cache_delete(clean_cache):
    """测试缓存删除"""
    from src.core.cache import CacheManager
    
    cache = CacheManager(cache_dir=clean_cache / "cache")
    
    cache.set("test_key", "test_value")
    assert cache.get("test_key") is not None
    
    result = cache.delete("test_key")
    assert result is True
    assert cache.get("test_key") is None


def test_cache_get_or_set(clean_cache):
    """测试缓存获取或设置"""
    from src.core.cache import CacheManager
    
    cache = CacheManager(cache_dir=clean_cache / "cache")
    
    call_count = [0]
    
    def factory():
        call_count[0] += 1
        return {"computed": True}
    
    # 第一次调用，应该执行 factory
    value1 = cache.get_or_set("computed_key", factory)
    assert call_count[0] == 1
    assert value1["computed"] is True
    
    # 第二次调用，应该使用缓存
    value2 = cache.get_or_set("computed_key", factory)
    assert call_count[0] == 1  # 未再次调用
    assert value2["computed"] is True


def test_cache_project_stats(clean_cache):
    """测试缓存工程统计"""
    from src.core.cache import CacheManager
    
    cache = CacheManager(cache_dir=clean_cache / "cache")
    
    stats = {"total": 100, "active": 50}
    result = cache.cache_project_stats("proj_001", stats, timeout=60)
    assert result is True
    
    cached = cache.get_cached_project_stats("proj_001")
    assert cached is not None
    assert cached["total"] == 100


def test_cache_dataset_summary(clean_cache):
    """测试缓存数据集摘要"""
    from src.core.cache import CacheManager
    
    cache = CacheManager(cache_dir=clean_cache / "cache")
    
    summary = {"record_count": 1000, "curve_count": 100}
    result = cache.cache_dataset_summary("proj_001", "ds_001", summary, timeout=60)
    assert result is True
    
    cached = cache.get_cached_dataset_summary("proj_001", "ds_001")
    assert cached is not None
    assert cached["record_count"] == 1000


def test_cache_curve_stats(clean_cache):
    """测试缓存曲线统计"""
    from src.core.cache import CacheManager
    
    cache = CacheManager(cache_dir=clean_cache / "cache")
    
    stats = {"max_torque": 100.0, "max_angle": 360.0}
    result = cache.cache_curve_stats("proj_001", "ds_001", "c1", stats, timeout=60)
    assert result is True
    
    cached = cache.get_cached_curve_stats("proj_001", "ds_001", "c1")
    assert cached is not None
    assert cached["max_torque"] == 100.0


def test_cache_json(clean_cache):
    """测试 JSON 缓存"""
    from src.core.cache import CacheManager
    
    cache = CacheManager(cache_dir=clean_cache / "cache")
    
    data = {"key": "value", "number": 123}
    result = cache.set_json("json_key", data, timeout=60)
    assert result is True
    
    cached = cache.get_json("json_key")
    assert cached is not None
    assert cached["key"] == "value"
    assert cached["number"] == 123
