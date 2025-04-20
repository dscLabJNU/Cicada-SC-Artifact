from typing import Dict, Type
from .base_model_builder import BaseModelBuilder
from .resnet_pipeline_builder import ResNetPipelineBuilder
from .vit_pipeline_builder import ViTPipelineBuilder
from .vgg_pipeline_builder import VGGPipelineBuilder
from .resnet_builder import ResNetBuilder
from .vit_builder import ViTBuilder
from .vgg_builder import VGGBuilder


class ModelFactory:
    """模型工厂类"""

    _builders: Dict[str, Type[BaseModelBuilder]] = {
        'resnet': ResNetBuilder,
        'resnet_pipeline': ResNetPipelineBuilder,
        'vgg': VGGBuilder,
        'vgg_pipeline': VGGPipelineBuilder,
        'vit': ViTBuilder,
        'vit_pipeline': ViTPipelineBuilder,
    }

    @classmethod
    def get_builder(cls, model_type: str, use_pipeline: bool = False) -> BaseModelBuilder:
        """获取对应的模型构建器

        Args:
            model_type (str): 模型类型
            use_pipeline (bool): 是否使用 pipeline 方式加载
        """
        if use_pipeline:
            builder_type = f"{model_type}_pipeline"
        else:
            builder_type = model_type

        builder_class = cls._builders.get(builder_type.lower())
        if not builder_class:
            raise ValueError(f"Unsupported model type: {builder_type}")
        return builder_class()

    @classmethod
    def register_builder(cls, model_type: str, builder: Type[BaseModelBuilder]):
        """注册新的模型构建器"""
        cls._builders[model_type.lower()] = builder
