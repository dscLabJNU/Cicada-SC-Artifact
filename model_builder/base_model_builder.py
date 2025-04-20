from abc import ABC, abstractmethod
import torch.nn as nn


class BaseModelBuilder(ABC):
    """模型构建器的基类"""

    @abstractmethod
    def init_base_model(self):
        """初始化基础模型"""
        pass

    @abstractmethod
    def make_layer(self, *args, **kwargs):
        """创建模型层"""
        pass

    @abstractmethod
    def get_model(self, model_name: str) -> nn.Module:
        """获取完整模型"""
        pass
