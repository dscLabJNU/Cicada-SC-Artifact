from typing import Dict, Any, Optional, TypeVar, Generic
import torch.nn as nn
from threading import Lock

T = TypeVar('T', bound=nn.Module)


class LayerCache:
    """通用层级缓存管理器，使用单例模式确保全局唯一"""
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LayerCache, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._caches: Dict[str, Dict[str, nn.Module]] = {}
        self._initialized = True

    def register_cache(self, cache_name: str):
        """注册新的缓存类型"""
        if cache_name not in self._caches:
            self._caches[cache_name] = {}

    def get_cache_key(self, **kwargs) -> str:
        """生成缓存键"""
        return "_".join(f"{k}_{v}" for k, v in sorted(kwargs.items()))

    def get_or_create(self, cache_name: str, creator_fn: callable, **kwargs) -> nn.Module:
        """获取或创建层

        Args:
            cache_name: 缓存类型名称
            creator_fn: 创建层的函数
            **kwargs: 层的参数

        Returns:
            nn.Module: 缓存的或新创建的层
        """
        if cache_name not in self._caches:
            self.register_cache(cache_name)

        cache = self._caches[cache_name]
        cache_key = self.get_cache_key(**kwargs)

        if cache_key not in cache:
            cache[cache_key] = creator_fn(**kwargs)

        return cache[cache_key]

    def clear_cache(self, cache_name: Optional[str] = None):
        """清除指定或所有缓存"""
        if cache_name is None:
            self._caches.clear()
        elif cache_name in self._caches:
            self._caches[cache_name].clear()

    def get_cache_info(self) -> Dict[str, int]:
        """获取缓存信息"""
        return {name: len(cache) for name, cache in self._caches.items()}
