import yaml
import torchvision.models as models
import torch
from logger_config import get_logger

logger = get_logger(__name__)



class ModelConfig:
    """
    模型配置类，负责加载和解析 YAML 配置文件，
    并提供模型构造函数映射和权重枚举映射。
    """

    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        self.model_fn_map = {}
        self.weights_fn_map = {}
        self.model_to_default_weight = {}  # 添加默认权重映射字典
        self.load_config()

    def load_config(self):
        """从 YAML 文件加载模型和权重映射"""
        try:
            with open(self.yaml_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(
                f"Failed to load YAML config from '{self.yaml_path}': {e}")
            raise

        families = config.get('families', {})

        for family, models_list in families.items():
            for model_info in models_list:
                name = model_info['name']
                model_name = model_info['model']
                weight_str = model_info['weights']

                # 添加默认权重映射
                self.model_to_default_weight[name] = weight_str

                # 尝试从 torchvision.models 获取
                model_fn = getattr(models, model_name, None)
                if model_fn is None:
                    logger.warning(
                        f"Model '{model_name}' not found in torchvision.models")
                    continue

                self.model_fn_map[name] = model_fn

                # 获取权重枚举（仅针对 torchvision 模型）
                if not weight_str.startswith('huggingface:'):
                    try:
                        weight_class_name, weight_attr = weight_str.split('.')
                        weight_class = getattr(models, weight_class_name)
                        weight_enum = getattr(weight_class, weight_attr)
                        self.weights_fn_map[weight_str] = weight_enum
                    except AttributeError as e:
                        logger.error(
                            f"Failed to get weight enum for '{weight_str}': {e}")
                        continue

    def get_model_fn_map(self):
        """
        返回模型构造函数映射。
        """
        return self.model_fn_map

    def get_weights_fn_map(self):
        """
        返回权重枚举映射。
        """
        return self.weights_fn_map

    def get_default_weight(self):
        """获取模型的默认权重键"""
        return self.model_to_default_weight
