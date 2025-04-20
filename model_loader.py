import torch
from model_config import ModelConfig
from logger_config import get_logger
logger = get_logger(__name__)


# 加载模型配置
yaml_path = 'models_config.yaml'  # 确保路径正确
model_config = ModelConfig(yaml_path)
model_fn_map = model_config.get_model_fn_map()
weights_fn_map = model_config.get_weights_fn_map()


def load_model_from_source(model_name: str, weights_key: str = "") -> torch.nn.Module:
    """从各种来源加载模型"""
    logger.info(f"Loading model: {model_name} with weights: {weights_key}")
    return _load_from_local(model_name, weights_key)

def _load_from_local(model_name: str, weights_key: str = "") -> torch.nn.Module:
    """从本地加载模型"""
    # 获取模型构造函数
    model_class = model_fn_map.get(model_name, None)
    if model_class is None:
        raise ValueError(f"Model '{model_name}' not found in model_fn_map.")

    kwargs = {}
    if model_name == 'vit_h_14':
        # 特殊处理 vit_h_14
        kwargs["image_size"] = 518

    if weights_key:
        weights = weights_fn_map.get(weights_key, None)
        if weights is None:
            raise ValueError(
                f"Weights key '{weights_key}' not found in weights_fn_map.")
        return model_class(weights=weights, **kwargs)
    else:
        return model_class(**kwargs)
