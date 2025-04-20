import logging
from torchvision import models
from logger_config import get_logger
import torch
import sys
import io
from typing import Union, Dict, Any, Optional, List
import numpy as np
import re
import yaml
from pathlib import Path
import json
import time

logger = get_logger(__name__)


class MemoryUtils:
    """内存计算和模型分析工具类"""

    @staticmethod
    def get_numpy_size(arr: np.ndarray) -> int:
        """
        计算NumPy数组的完整内存占用（字节）

        Args:
            arr (np.ndarray): 要计算大小的NumPy数组

        Returns:
            int: 数组的总内存占用（字节）
        """
        # 数据缓冲区大小
        size = arr.nbytes
        # 数组对象本身的开销（包含了基本的元数据如dtype,flags等）
        size += sys.getsizeof(arr) - arr.nbytes  # 减去nbytes避免重复计算
        # shape和strides的开销（它们是元组）
        size += sys.getsizeof(arr.shape)
        size += sys.getsizeof(arr.strides)
        return size

    @staticmethod
    def get_tensor_size(tensor: torch.Tensor) -> int:
        """
        计算PyTorch张量的内存占用（字节）

        Args:
            tensor (torch.Tensor): 要计算大小的PyTorch张量

        Returns:
            int: 张量的总内存占用（字节）
        """
        return tensor.numel() * tensor.element_size()

    @staticmethod
    def get_list_size(data: list) -> int:
        """
        计算列表的内存占用（字节），包括列表本身和所有元素

        Args:
            data (list): 要计算大小的列表

        Returns:
            int: 列表的总内存占用（字节）
        """
        size = sys.getsizeof(data)  # 列表对象本身的大小

        # 计算列表中每个元素的大小
        for item in data:
            if item is None:
                size += sys.getsizeof(None)
            elif isinstance(item, (int, float)):
                size += sys.getsizeof(item)
            elif isinstance(item, list):
                size += MemoryUtils.get_list_size(item)
            elif isinstance(item, np.ndarray):
                size += MemoryUtils.get_numpy_size(item)
            elif isinstance(item, torch.Tensor):
                size += MemoryUtils.get_tensor_size(item)
            else:
                size += sys.getsizeof(item)
        return size

    @staticmethod
    def get_compressed_data_size(data: Union[np.ndarray, dict, list]) -> int:
        """
        计算压缩数据的总内存占用（字节）

        Args:
            data: 压缩后的数据结构（可以是数组、字典或列表）

        Returns:
            int: 数据的总内存占用（字节）
        """
        if isinstance(data, np.ndarray):
            return MemoryUtils.get_numpy_size(data)
        elif isinstance(data, dict):
            total_size = 0
            for value in data.values():
                if isinstance(value, (np.ndarray, dict, list)):
                    total_size += MemoryUtils.get_compressed_data_size(value)
            return total_size
        elif isinstance(data, list):
            return MemoryUtils.get_list_size(data)
        return 0

    @staticmethod
    def bytes_to_mb(bytes_size: int) -> float:
        """
        将字节转换为兆字节

        Args:
            bytes_size (int): 字节大小

        Returns:
            float: 兆字节大小
        """
        return bytes_size / (1024 * 1024)

    @staticmethod
    def get_model_memory_usage(model: torch.nn.Module) -> Dict[str, Any]:
        """
        计算模型的内存使用情况，针对不同数据类型进行优化计算

        Args:
            model: PyTorch 模型

        Returns:
            Dict: 包含内存使用信息的字典
        """
        """计算模型的内存使用情况，分别计算各个部分"""
        # 1. 计算参数大小
        param_size = 0
        for p in model.parameters():
            if p.dtype == torch.bool:
                param_size += p.nelement() // 8 + (1 if p.nelement() % 8 else 0)
            else:
                param_size += p.nelement() * p.element_size()

        # 2. 计算缓冲区大小
        buffer_size = sum(b.nelement() * b.element_size()
                          for b in model.buffers())

        # 3. 计算序列化大小 - 修改这部分逻辑
        try:
            # 只序列化状态字典而不是整个模型
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            base_serialized_size = buffer.tell()

            # 计算额外的序列化开销
            empty_buffer = io.BytesIO()
            torch.save({}, empty_buffer)
            serialization_overhead = empty_buffer.tell()

            # 实际的序列化大小应该减去基本开销
            base_serialized_size = max(
                0, base_serialized_size - serialization_overhead)

        except Exception as e:
            logger.warning(f"计算序列化大小时出错: {e}")
            base_serialized_size = 0

        # 4. 计算基础实例大小（不包括参数和缓冲区）
        def get_base_size(obj, seen=None):
            if seen is None:
                seen = set()

            obj_id = id(obj)
            if obj_id in seen:
                return 0

            seen.add(obj_id)
            size = sys.getsizeof(obj)

            if isinstance(obj, dict):
                size += sum(get_base_size(k, seen) + get_base_size(v, seen)
                            for k, v in obj.items())
            elif isinstance(obj, (list, tuple, set, frozenset)):
                size += sum(get_base_size(item, seen) for item in obj)
            elif isinstance(obj, torch.Tensor):
                # 这里可能与之前的参数和缓冲区计算重复
                # 我们应该跳过已经计算过的参数和缓冲区张量
                if id(obj) not in {id(p) for p in model.parameters()} and \
                   id(obj) not in {id(b) for b in model.buffers()}:
                    if obj.dtype == torch.bool:
                        size += obj.nelement() // 8 + (1 if obj.nelement() % 8 else 0)
                    elif obj.dtype in [torch.float16, torch.bfloat16]:
                        size += obj.nelement() * 2
                    elif obj.dtype in [torch.int8, torch.uint8]:
                        size += obj.nelement()
                    elif obj.dtype in [torch.int16, torch.short]:
                        size += obj.nelement() * 2
                    elif obj.dtype in [torch.int32, torch.int]:
                        size += obj.nelement() * 4
                    elif obj.dtype in [torch.int64, torch.long]:
                        size += obj.nelement() * 8
                    elif obj.dtype in [torch.float32, torch.float]:
                        size += obj.nelement() * 4
                    elif obj.dtype in [torch.float64, torch.double]:
                        size += obj.nelement() * 8
                    else:
                        size += obj.nelement() * obj.element_size()
            if hasattr(obj, '__dict__'):
                size += get_base_size(obj.__dict__, seen)
            if hasattr(obj, '__slots__'):
                size += sum(get_base_size(getattr(obj, slot), seen)
                            for slot in obj.__slots__ if hasattr(obj, slot))

            return size

        base_instance_size = get_base_size(model)

        return {
            'parameters': {
                'count': sum(p.nelement() for p in model.parameters()),
                'size_bytes': param_size,
                'size_mb': param_size / (1024 * 1024)
            },
            'buffers': {
                'size_bytes': buffer_size,
                'size_mb': buffer_size / (1024 * 1024)
            },
            'serialized': {
                'size_bytes': base_serialized_size,
                'size_mb': base_serialized_size / (1024 * 1024)
            },
            'instance': {
                'size_bytes': base_instance_size,
                'size_mb': base_instance_size / (1024 * 1024)
            },
            'total': {
                'size_bytes': param_size + buffer_size + base_instance_size,  # 不需要计算序列化后的大小
                'size_mb': (param_size + buffer_size + base_instance_size) / (1024 * 1024)
            }
        }

    @staticmethod
    def format_size(size_info: Dict[str, Any], detailed: bool = False) -> str:
        """格式化大小信息为易读的字符串"""
        if detailed:
            return (
                f"参数量: {size_info['parameters']['count']:,} "
                f"({size_info['parameters']['size_mb']:.2f} MB) | "
                f"缓冲区: {size_info['buffers']['size_mb']:.2f} MB | "
                f"序列化: {size_info['serialized']['size_mb']:.2f} MB | "
                f"实例: {size_info['instance']['size_mb']:.2f} MB | "
                f"总计: {size_info['total']['size_mb']:.2f} MB"
            )
        return (
            f"参数量: {size_info['parameters']['count']:,} | "
            f"总内存: {size_info['total']['size_mb']:.2f} MB"
        )

    @staticmethod
    def compare_model_structures(model1: torch.nn.Module, model2: torch.nn.Module,
                                 name1: str = "Model1", name2: str = "Model2",
                                 prefix: str = "") -> bool:
        """
        详细比较两个模型的结构和参数

        Args:
            model1: 第一个模型
            model2: 第二个模型
            name1: 第一个模型的名称
            name2: 第二个模型的名称
            prefix: 日志输出的前缀

        Returns:
            bool: 如果发现差异返回 True，否则返回 False
        """
        has_difference = False

        # 比较模型的参数数量
        model1_params = sum(p.numel() for p in model1.parameters())
        model2_params = sum(p.numel() for p in model2.parameters())
        if model1_params != model2_params:
            has_difference = True
            logger.warning(f"\n{prefix}参数数量不同:")
            logger.warning(f"{name1}: {model1_params:,}")
            logger.warning(f"{name2}: {model2_params:,}")

        # 获取所有模块，过滤掉空字符串
        model1_modules = {name: module for name,
                          module in model1.named_modules() if name}
        model2_modules = {name: module for name,
                          module in model2.named_modules() if name}

        # 比较模块集合
        model1_names = set(model1_modules.keys())
        model2_names = set(model2_modules.keys())

        # 检查是否有缺失的层
        missing_layers = model1_names - model2_names
        extra_layers = model2_names - model1_names
        if missing_layers:
            has_difference = True
            logger.warning(f"\n{prefix}{name2} 缺失的层: {missing_layers}")
        if extra_layers:
            has_difference = True
            logger.warning(f"\n{prefix}{name2} 多余的层: {extra_layers}")

        # 对于共同存在的层，比较其属性
        common_layers = model1_names & model2_names
        for name in common_layers:
            m1_module = model1_modules[name]
            m2_module = model2_modules[name]

            # 检查层的参数形状
            m1_params = {name: p.shape for name,
                         p in m1_module.named_parameters()}
            m2_params = {name: p.shape for name,
                         p in m2_module.named_parameters()}

            if m1_params.keys() != m2_params.keys():
                has_difference = True
                logger.warning(f"\n{prefix}层 {name} 的参数名称不同:")
                logger.warning(f"{name1}: {list(m1_params.keys())}")
                logger.warning(f"{name2}: {list(m2_params.keys())}")
                continue

            for param_name in m1_params:
                if m1_params[param_name] != m2_params[param_name]:
                    has_difference = True
                    logger.warning(
                        f"\n{prefix}层 {name} 的参数 {param_name} 形状不同:")
                    logger.warning(f"{name1}: {m1_params[param_name]}")
                    logger.warning(f"{name2}: {m2_params[param_name]}")

            # 检查重要属性
            important_attrs = ['in_channels', 'out_channels', 'kernel_size',
                               'stride', 'padding', 'groups', 'dilation']
            for attr in important_attrs:
                if hasattr(m1_module, attr) and hasattr(m2_module, attr):
                    m1_value = getattr(m1_module, attr)
                    m2_value = getattr(m2_module, attr)
                    if m1_value != m2_value:
                        has_difference = True
                        logger.warning(f"\n{prefix}层 {name} 的属性 {attr} 不同:")
                        logger.warning(f"{name1}: {m1_value}")
                        logger.warning(f"{name2}: {m2_value}")

        # 如果没有发现任何差异，输出确认信息
        if not has_difference:
            logger.info(f"{prefix}模型结构完全相同")

        return has_difference

    @staticmethod
    def get_current_memory_usage() -> float:
        """
        获取当前进程的内存使用情况（MB）

        Returns:
            float: 当前内存使用量（MB）
        """
        try:
            import psutil
            import os

            # 获取当前进程
            process = psutil.Process(os.getpid())

            # 获取内存信息（字节）
            memory_info = process.memory_info()

            # 转换为MB并返回
            return memory_info.rss / (1024 * 1024)  # RSS: Resident Set Size

        except ImportError:
            logger.warning("psutil 模块未安装，无法获取内存使用情况")
            return 0
        except Exception as e:
            logger.error(f"获取内存使用情况时出错: {str(e)}")
            return 0


def get_weights_instance(model_id: str, weights_key: str):
    """
    将字符串形式的权重键转换为实际的 Weights 实例
    例如: "ResNet50_Weights.DEFAULT" -> models.ResNet50_Weights.DEFAULT

    Args:
        model_id (str): 模型ID，如 "resnet50"
        weights_key (str): 权重键，如 "ResNet50_Weights.DEFAULT"

    Returns:
        WeightsEnum: 权重实例，如果解析失败则返回 None
    """
    if not weights_key:
        return None
    logger.info(f"model_id: {model_id}, weights_key: {weights_key}")

    try:
        # 分解权重键，例如 "ResNet50_Weights.DEFAULT" -> ["ResNet50_Weights", "DEFAULT"]
        weights_class_name, weights_enum = weights_key.split('.')

        # 获取权重类，例如 ResNet50_Weights
        weights_class = getattr(models, weights_class_name)
        if not weights_class:
            logger.error(f"Weight class not found: {weights_class_name}")
            return None
        weights_instance = getattr(weights_class, weights_enum)
        logger.info(f"weights_instance: {weights_instance}")
        return weights_instance

    except Exception as e:
        logger.error(f"Failed to get weights instance: {e}")
        return None


def verify_model_state_dict(server_state_dict: dict, reference_state_dict: dict, model_name: str) -> bool:
    """
    验证服务器返回的 state_dict 是否与参考模型的 state_dict 一致。

    Args:
        server_state_dict (dict): 从服务器获取的模型 state_dict。
        reference_state_dict (dict): 参考模型的 state_dict。
        model_name (str): 模型的名称（如 'resnet50_default' 或 'bert-base-uncased_default'）。

    Returns:
        bool: 验证是否通过
    """
    # 验证一致性
    all_match = True
    differing_layers = []
    # 验证参数数量
    if len(server_state_dict) != len(reference_state_dict):
        logger.error(
            f"Parameter count mismatch: server({len(server_state_dict)}) != reference({len(reference_state_dict)})")
        return False

    for key in server_state_dict:
        server_tensor = server_state_dict[key]
        client_tensor = reference_state_dict.get(key)

        if client_tensor is None:
            all_match = False
            differing_layers.append(f"{key} (missing in reference)")
            continue

        # 确保数据类型匹配
        if server_tensor.dtype != client_tensor.dtype:
            logger.warning(
                f"Converting {key} from {server_tensor.dtype} to {client_tensor.dtype}")
            server_tensor = server_tensor.to(client_tensor.dtype)

        try:
            if not torch.allclose(server_tensor, client_tensor, atol=1e-4):
                all_match = False
                differing_layers.append(f"{key} (values mismatch)")
        except Exception as e:
            all_match = False
            differing_layers.append(f"{key} (comparison failed: {str(e)})")

    if all_match:
        logger.info(f"All parameters match exactly for model '{model_name}'.")
    else:
        logger.error(f"Some parameters differ for model '{model_name}'.")
        logger.error(f"Total differing layers: {len(differing_layers)}")
        logger.error("Differing layers:")
        for layer in differing_layers:
            logger.error(f" - {layer}")

    return all_match


def load_densenet_state_dict(model, state_dict):
    """特殊处理 DenseNet 的权重加载"""
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    model.load_state_dict(state_dict)


def load_model_configs(family_name: str = None) -> List[Dict[str, str]]:
    """
    从yaml文件加载模型配置

    Args:
        family_name: 可选，指定要加载的模型家族名称

    Returns:
        list: 模型配置列表
    """
    config_path = Path("models_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if family_name:
        if family_name not in config['families']:
            raise ValueError(f"不支持的模型家族: {family_name}")
        family_configs = config['families'][family_name]
        return [
            {
                "model_nickname": model["name"],
                "weights_key": model["weights"],
                "model_name": model["model"]
            }
            for model in family_configs
        ]
    return config['families']


def split_weights_by_layer(model_name: str, weights_key: str, cache_dir: str = "persistent_cache") -> Dict[str, str]:
    """
    将模型权重按层拆分并保存到缓存目录，使用模块化方式存储

    Args:
        model_name (str): 模型名称，如 'resnet50'
        weights_key (str): 权重标识，如 'ResNet50_Weights.DEFAULT'
        cache_dir (str): 缓存目录路径

    Returns:
        Dict[str, str]: 层名称到权重文件路径的映射
    """
    try:
        # 创建缓存目录
        cache_path = Path(cache_dir) / model_name
        cache_path.mkdir(parents=True, exist_ok=True)

        # 加载完整模型和权重
        weights = get_weights_instance(model_name, weights_key)
        if weights is None:
            raise ValueError(f"无法加载权重: {weights_key}")

        # 创建模型实例
        model = getattr(models, model_name)(weights=weights)

        # 创建层名称映射
        layer_names = {}
        for idx, (name, layer) in enumerate(model.named_modules()):
            # 只保留第一级子模块且名称非空
            if name.count('.') < 1 and name:
                layer_names[layer] = name

        # 保存每个层的权重
        layer_weights = {}
        for layer, name in layer_names.items():
            if len(list(layer.parameters())) > 0:
                # 获取包含所有子模块参数的完整state_dict
                layer_state = layer.state_dict()
                save_path = cache_path / f"{name}.pt"
                torch.save(layer_state, save_path)
                layer_weights[name] = str(save_path)

        # 为没有参数的层创建空的权重文件
        no_param_layers = ['relu', 'maxpool', 'avgpool']
        for layer_name in no_param_layers:
            if layer_name not in layer_weights:  # 只处理尚未保存的层
                weight_path = cache_path / f"{layer_name}.pt"
                torch.save({}, weight_path)  # 保存空的state_dict
                layer_weights[layer_name] = str(weight_path)

        # 创建元数据文件
        metadata = {
            'model_name': model_name,
            'weights_key': weights_key,
            'layer_files': layer_weights,
            'creation_time': time.time()
        }

        with open(cache_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"模型 {model_name} 的权重已按层拆分并保存到 {cache_path}")
        return layer_weights

    except Exception as e:
        logger.error(f"拆分权重时出错: {str(e)}")
        raise


def get_layer_weights_path(model_name: str, layer_name: str, cache_dir: str = "persistent_cache") -> Optional[str]:
    """
    获取特定层的权重文件路径

    Args:
        model_name (str): 模型名称
        layer_name (str): 层名称
        cache_dir (str): 缓存目录路径

    Returns:
        Optional[str]: 权重文件路径，如果不存在则返回 None
    """
    try:
        cache_path = Path(cache_dir) / model_name
        weight_path = cache_path / f"{layer_name}.pt"

        if weight_path.exists():
            return str(weight_path)
        return None

    except Exception as e:
        logger.error(f"获取层权重路径时出错: {str(e)}")
        return None
