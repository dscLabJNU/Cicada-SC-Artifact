import abc
import torch
from typing import Dict, Optional
from parameter import ParameterCache
from model_config import ModelConfig
from logger_config import get_logger

logger = get_logger(__name__)


class ModelCacheStrategy(abc.ABC):
    """模型缓存策略的抽象基类"""

    @abc.abstractmethod
    def initialize(self) -> None:
        """初始化策略所需的资源"""
        pass

    @abc.abstractmethod
    def add_model(self, model: torch.nn.Module, model_name: str, weights_key: str = "") -> None:
        """添加模型到缓存中"""
        pass

    @abc.abstractmethod
    def get_model_state_dict(self, model_name: str) -> Dict[str, torch.Tensor]:
        """获取模型的state_dict"""
        pass

    @abc.abstractmethod
    def get_layer_state(self, model_name: str, layer_name: str) -> Optional[torch.Tensor]:
        """获取特定层的参数"""
        pass


class TetrisStrategy(ModelCacheStrategy):
    def __init__(self, cache_size: int = 2 * 1024 * 1024 * 1024, persistent: bool = False):
        """
        初始化 Tetris 策略。

        参数：
        - cache_size (int): 缓存大小（以字节为单位）。
        - persistent (bool): 是否使用持久化缓存。
        """
        logger.info("Initializing TetrisStrategy...")
        self.cache = ParameterCache(max_size=cache_size, persistent=persistent)
        self.models_manifest = {}
        self.config = ModelConfig('models_config.yaml')
        self.model_fn_map = {}
        self.weights_fn_map = {}

    def initialize(self) -> None:
        # 初始化model_fn_map和weights_fn_map
        self.model_fn_map.update(self.config.get_model_fn_map())
        self.weights_fn_map.update(self.config.get_weights_fn_map())

    def add_model(self, model: torch.nn.Module, model_name: str, weights_key: str = "") -> None:
        """添加模型到缓存中，如果模型已存在则跳过。"""
        if model_name in self.models_manifest:
            logger.info(f"Model '{model_name}' already in cache, skipping...")
            return

        state_dict = model.state_dict()
        self.models_manifest[model_name] = {
            "parameters": {}
        }

        for param_name, param_tensor in state_dict.items():
            # 检查参数是否已经在缓存中
            param_id = self.cache.compute_param_id(param_name, model_name)
            if not self.cache.has_parameter(param_id):
                self.cache.store_parameter(param_id, param_tensor)
            self.models_manifest[model_name]["parameters"][param_name] = param_id

        logger.info(f"Model '{model_name}' added to cache successfully.")

    def get_model_state_dict(self, model_name: str) -> Dict[str, torch.Tensor]:
        if model_name not in self.models_manifest:
            raise ValueError(f"Model '{model_name}' not loaded in cache.")

        param_map = self.models_manifest[model_name]["parameters"]
        return {
            param_name: self.cache.get_parameter(param_id)
            for param_name, param_id in param_map.items()
        }

    def get_layer_state(self, model_name: str, layer_name: str) -> Optional[torch.Tensor]:
        if model_name not in self.models_manifest:
            raise ValueError(f"Model '{model_name}' not loaded in cache.")

        param_map = self.models_manifest[model_name]["parameters"]
        if layer_name not in param_map:
            raise ValueError(
                f"Layer '{layer_name}' not found in model '{model_name}'.")

        param_id = param_map[layer_name]
        return self.cache.get_parameter(param_id)