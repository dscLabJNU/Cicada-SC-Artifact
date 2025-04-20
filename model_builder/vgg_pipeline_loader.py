from .pipeline_loader import PipelineLoader, LayerLoadingState
import torch.nn as nn
from typing import Iterator, Tuple, Dict, Optional, List
from utils import get_logger
import time
from dataclasses import dataclass
import os

logger = get_logger(__name__)


@dataclass
class LayerStats:
    """层统计信息"""
    layer_name: str
    create_time: float = 0.0      # 创建时间
    init_weight_time: float = 0.0  # 权重初始化时间
    weights_time: float = 0.0     # 权重加载时间
    compute_time: float = 0.0     # 计算时间
    wait_time: float = 0.0        # 等待时间
    total_time: float = 0.0       # 总时间
    start_time: float = 0.0       # 开始时间
    end_time: float = 0.0         # 结束时间


class VGGPipelineLoader(PipelineLoader):
    def __init__(self, model_name: str):
        """初始化 VGG pipeline 加载器

        Args:
            model_name (str): 模型名称
        """
        self.model_name = model_name
        self.loading_state = LayerLoadingState()

        # 从环境变量读取配置
        self.use_miniloader = os.environ.get(
            'USE_MINILOADER', 'False').lower() == 'true'

        # 层统计信息
        self.layer_stats: Dict[str, LayerStats] = {}

        # VGG 配置
        self.features_config = self._get_vgg_config(model_name)
        self.classifier_config = [
            ('linear1', 4096),
            ('relu', None),
            ('dropout1', 0.5),
            ('linear2', 4096),
            ('relu', None),
            ('dropout2', 0.5),
            ('linear3', 1000)
        ]

    def _get_vgg_config(self, model_name: str) -> List[Tuple[str, Optional[int]]]:
        """获取 VGG 特征提取层配置"""
        configs = {
            'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }

        if model_name not in configs:
            raise ValueError(f"Unsupported VGG model: {model_name}")

        # 转换配置为层名称和参数
        config = []
        in_channels = 3
        layer_idx = 1

        for v in configs[model_name]:
            if v == 'M':
                config.append((f'maxpool{layer_idx}', None))
                layer_idx += 1
            else:
                config.append((f'conv{layer_idx}', v))
                config.append((f'relu{layer_idx}', None))
                in_channels = v
                layer_idx += 1

        return config

    def has_weights(self, layer_name: str) -> bool:
        """检查层是否有权重需要加载"""
        non_weight_layers = {}
        return layer_name not in non_weight_layers

    def init_weight(self, layer: nn.Module, layer_name: str):
        """初始化层权重"""
        weights_time_start = time.time()
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

        logger.debug(
            f"Init weight for layer: {layer_name} time: {(time.time() - weights_time_start)*1000:.2f}ms")

    def create_layer(self, layer_name: str) -> Optional[Tuple[nn.Module, Dict]]:
        """创建单个层实例，与拆分后的权重文件对应"""
        try:
            structure_start_ts = time.perf_counter()
            layer = None

            if layer_name == 'features':
                # 创建特征提取模块（包含所有卷积层和ReLU）
                layers = []
                in_channels = 3
                for layer_cfg in self.features_config:
                    name, param = layer_cfg
                    if name.startswith('conv'):
                        conv = nn.Conv2d(in_channels, param,
                                         kernel_size=3, padding=1)
                        layers += [conv, nn.ReLU(inplace=True)]
                        in_channels = param
                    elif name.startswith('maxpool'):
                        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                layer = nn.Sequential(*layers)  # 对应features.pt的完整结构

            elif layer_name == 'avgpool':
                layer = nn.AdaptiveAvgPool2d((7, 7))  # 无参数层，对应空文件

            elif layer_name == 'classifier':
                # 创建分类器模块（包含全连接层）
                layers = []
                in_features = 512 * 7 * 7
                for name, param in self.classifier_config:
                    if name.startswith('linear'):
                        layers.append(nn.Linear(in_features, param))
                        in_features = param
                    elif name.startswith('relu'):
                        layers.append(nn.ReLU(inplace=True))
                    elif name.startswith('dropout'):
                        layers.append(nn.Dropout(p=param))
                layer = nn.Sequential(*layers)  # 对应classifier.pt的完整结构

            if layer is None:
                raise ValueError(f"Unknown layer name: {layer_name}")

            structure_end_ts = time.perf_counter()
            structure_time = structure_end_ts - structure_start_ts

            init_weight_time = 0.0
            init_weight_start_ts = time.perf_counter()
            # 只在非 miniloader 模式下执行权重初始化
            if not self.use_miniloader:
                self.init_weight(layer, layer_name)
            init_weight_end_ts = time.perf_counter()
            init_weight_time = init_weight_end_ts - init_weight_start_ts

            # 初始化层统计信息
            self.layer_stats[layer_name] = LayerStats(
                layer_name=layer_name,
                create_time=structure_time,
                start_time=structure_start_ts,
                init_weight_time=init_weight_time,
                end_time=structure_end_ts
            )
            # 如果是没有权重的层，直接标记为已加载
            if not self.has_weights(layer_name):
                self.loading_state.mark_layer_loaded(layer_name)

            # 构造时间戳信息字典
            structure_ts_info = {
                "structure_start_ts": structure_start_ts,
                "structure_end_ts": structure_end_ts,
                "init_weight_start_ts": init_weight_start_ts,
                "init_weight_end_ts": init_weight_end_ts,
            }

            return layer, structure_ts_info

        except Exception as e:
            logger.error(f"创建层 {layer_name} 时出错: {str(e)}")
            return None

    def get_layer_sequence(self) -> Iterator[Tuple[str, nn.Module, Dict[str, float]]]:
        """按顺序返回 VGG的层序列"""
        layer_names = ['features', 'avgpool', 'classifier']
        for layer_name in layer_names:
            layer_info = self.create_layer(layer_name)
            if layer_info:
                layer, structure_ts_info = layer_info
                yield layer_name, layer, structure_ts_info

    def record_layer_compute_time(self, layer_name: str, compute_time: float):
        """记录层的计算时间"""
        if layer_name in self.layer_stats:
            stats = self.layer_stats[layer_name]
            stats.compute_time = compute_time
            stats.end_time = time.perf_counter()
            stats.total_time = stats.end_time - stats.start_time

    def record_layer_weights_time(self, layer_name: str, weights_time: float):
        """记录层的权重加载时间"""
        if layer_name in self.layer_stats:
            self.layer_stats[layer_name].weights_time = weights_time

    def record_layer_wait_time(self, layer_name: str, wait_time: float):
        """记录层的等待时间"""
        if layer_name in self.layer_stats:
            self.layer_stats[layer_name].wait_time = wait_time

    def get_layer_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有层的统计信息"""
        return {
            name: {
                'create_time_ms': stats.create_time * 1000,
                'weights_time_ms': stats.weights_time * 1000,
                'compute_time_ms': stats.compute_time * 1000,
                'wait_time_ms': stats.wait_time * 1000,
                'total_time_ms': stats.total_time * 1000,
                'pipeline_efficiency': (stats.create_time + stats.weights_time + stats.compute_time) / stats.total_time
                if stats.total_time > 0 else 0
            }
            for name, stats in self.layer_stats.items()
        }

    def print_layer_stats(self):
        """打印层统计信息"""
        stats = self.get_layer_stats()
        logger.info(f"\n{self.model_name} 模型层统计信息:")

        for layer_name, layer_stats in stats.items():
            logger.info(f"\n层 {layer_name}:")
            logger.info(f"- 创建时间: {layer_stats['create_time_ms']:.2f}ms")
            logger.info(f"- 权重加载: {layer_stats['weights_time_ms']:.2f}ms")
            logger.info(f"- 计算时间: {layer_stats['compute_time_ms']:.2f}ms")
            logger.info(f"- 等待时间: {layer_stats['wait_time_ms']:.2f}ms")
            logger.info(f"- 总时间: {layer_stats['total_time_ms']:.2f}ms")
            logger.info(f"- 流水线效率: {layer_stats['pipeline_efficiency']:.2%}")
