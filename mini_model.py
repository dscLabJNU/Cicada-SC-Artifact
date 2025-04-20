import torch
from logger_config import get_logger
from model_builder.model_factory import ModelFactory

# 获取 logger
logger = get_logger(__name__)


class MiniModel(torch.nn.Module):
    _instance = None  # 类变量用于存储单例实例

    def __new__(cls, *args, **kwargs):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info("Created new MiniModel instance")
        return cls._instance

    def __init__(self, model_types=[]):
        """
        初始化MiniModel
        Args:
            model_types: 可选，指定要初始化的模型类型列表，如 ['resnet', 'vgg']
        """
        # 防止重复初始化
        if not hasattr(self, 'initialized'):
            super().__init__()
            self.builders = {}
            self.initialized = True
        self.add_model_types(model_types)

    def add_model_types(self, model_types: list) -> None:
        """
        添加新的模型类型支持

        Args:
            model_types: 要添加的模型类型列表，如 ['vgg', 'vision_transformer']
        """
        for model_name in model_types:
            model_type = self._get_model_type(model_name)
            if model_type not in self.builders:
                logger.info(f"初始化 {model_type} 模型构建器")
                self.builders[model_type] = ModelFactory.get_builder(
                    model_type)
        logger.info(f"当前支持的模型构建器: {list(self.builders.keys())}")

    def get_model(self, model_name: str) -> torch.nn.Module:
        """获取指定模型"""
        model_type = self._get_model_type(model_name)
        if model_type not in self.builders:
            self.builders[model_type] = ModelFactory.get_builder(model_type)
        return self.builders[model_type].get_model(model_name)

    def _get_model_type(self, model_name: str) -> str:
        logger.info(f"model_name: {model_name}")
        """从模型名称中提取模型类型"""
        if model_name.startswith('resnet'):
            return 'resnet'
        elif model_name.startswith('vgg'):
            return 'vgg'
        elif model_name.startswith('vit'):
            return 'vit'
        else:
            raise ValueError(f"Unsupported model: {model_name}")
