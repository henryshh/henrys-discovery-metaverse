"""
Cache Manager - Disk-based caching with diskcache
"""
import os
import pickle
from pathlib import Path
from typing import Any, Optional, Dict
from datetime import datetime


class CacheManager:
    """Disk-based cache manager using diskcache."""
    
    def __init__(self, cache_path: str = "data/cache", cache_dir: str = None):
        # 支持cache_dir参数作为别名
        if cache_dir is not None:
            cache_path = cache_dir
        self.cache_path = Path(cache_path)
        self.cache_dir = self.cache_path  # 应该是一个Path对象，不是字符串
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self._cache = None
        self._init_cache()
    
    def _init_cache(self):
        """Initialize diskcache."""
        try:
            from diskcache import Cache
            self._cache = Cache(str(self.cache_path))
            # 公开的cache属性，用于兼容性
            self.cache = self._cache
        except ImportError:
            print("Warning: diskcache not installed. Caching disabled.")
            print("Install with: pip install diskcache")
            self._cache = None
            self.cache = None
    
    def is_available(self) -> bool:
        """Check if caching is available."""
        return self._cache is not None
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        if not self._cache:
            return None
        
        try:
            return self._cache.get(key)
        except Exception:
            return None
    
    def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """Set a value in cache with expiration (default 1 hour)."""
        if not self._cache:
            return False
        
        try:
            self._cache.set(key, value, expire=expire)
            return True
        except Exception:
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        if not self._cache:
            return False
        
        try:
            self._cache.delete(key)
            return True
        except Exception:
            return False
    
    def clear(self):
        """Clear all cache entries."""
        if self._cache:
            try:
                self._cache.clear()
            except Exception:
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._cache:
            return {
                "available": False,
                "size": 0,
                "count": 0
            }
        
        try:
            size = self._cache.volume()
            count = len(self._cache)
            return {
                "available": True,
                "size": size,
                "count": count
            }
        except Exception:
            return {
                "available": True,
                "size": 0,
                "count": 0
            }

    def get_detailed_stats(self) -> Dict[str, Any]:
        """修复统计功能"""
        try:
            return {
                "size": len(self._cache) if self._cache else 0,
                "hits": getattr(self, '_hits', 0),
                "misses": getattr(self, '_misses', 0)
            }
        except Exception as e:
            return {"error": str(e)}

    def get_or_set(self, key: str, factory, timeout: int = 3600) -> Any:
        """获取或设置缓存值，如果不存在则执行factory函数"""
        value = self.get(key)
        if value is None:
            value = factory()
            self.set(key, value, expire=timeout)
        return value

    def cache_project_stats(self, project_id: str, stats: Dict, timeout: int = 60) -> bool:
        """缓存项目统计"""
        key = f"project_stats:{project_id}"
        return self.set(key, stats, expire=timeout)

    def get_cached_project_stats(self, project_id: str) -> Optional[Dict]:
        """获取缓存的项目统计"""
        key = f"project_stats:{project_id}"
        return self.get(key)

    def cache_dataset_summary(self, project_id: str, dataset_id: str, summary: Dict, timeout: int = 60) -> bool:
        """缓存数据集摘要"""
        key = f"dataset_summary:{project_id}:{dataset_id}"
        return self.set(key, summary, expire=timeout)

    def get_cached_dataset_summary(self, project_id: str, dataset_id: str) -> Optional[Dict]:
        """获取缓存的数据集摘要"""
        key = f"dataset_summary:{project_id}:{dataset_id}"
        return self.get(key)

    def cache_curve_stats(self, project_id: str, dataset_id: str, curve_id: str, stats: Dict, timeout: int = 60) -> bool:
        """缓存曲线统计"""
        key = f"curve_stats:{project_id}:{dataset_id}:{curve_id}"
        return self.set(key, stats, expire=timeout)

    def get_cached_curve_stats(self, project_id: str, dataset_id: str, curve_id: str) -> Optional[Dict]:
        """获取缓存的曲线统计"""
        key = f"curve_stats:{project_id}:{dataset_id}:{curve_id}"
        return self.get(key)

    def set_json(self, key: str, data: Dict, timeout: int = 60) -> bool:
        """设置JSON数据"""
        import json
        try:
            json_str = json.dumps(data)
            return self.set(key, json_str, expire=timeout)
        except Exception:
            return False

    def get_json(self, key: str) -> Optional[Dict]:
        """获取JSON数据"""
        import json
        try:
            json_str = self.get(key)
            if json_str is None:
                return None
            return json.loads(json_str)
        except Exception:
            return None


# Global instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(cache_path: str = "data/cache") -> CacheManager:
    """Get or create the global cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(cache_path)
    return _cache_manager


def reset_cache_manager():
    """Reset the global cache manager."""
    global _cache_manager
    if _cache_manager is not None:
        _cache_manager.clear()
        _cache_manager = None
