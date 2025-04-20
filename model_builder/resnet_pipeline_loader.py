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


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNetPipelineLoader(PipelineLoader):
    def __init__(self, model_name: str, block_type, layers: list):
        """初始化 ResNet pipeline 加载器

        Args:
            model_name (str): 模型名称
            block_type: block 类型 (BasicBlock 或 Bottleneck)
            layers (list): 每层的 block 数量
        """
        self.model_name = model_name
        self.block_type = block_type
        self.layers = layers
        self.loading_state = LayerLoadingState()

        # 从环境变量读取配置
        self.use_miniloader = os.environ.get(
            'USE_MINILOADER', 'False').lower() == 'true'

        # ResNet 基本参数
        self.inplanes = 64
        self.base_width = 64  # 默认 base_width
        if 'wide' in model_name:
            self.base_width = 64 * 2  # wide ResNet 的 base_width 加倍

        self.groups = 1
        self.dilation = 1
        self.norm_layer = nn.BatchNorm2d

        # 层统计信息
        self.layer_stats: Dict[str, LayerStats] = {}

    def has_weights(self, layer_name: str) -> bool:
        """检查层是否有权重需要加载"""
        # 这些层没有需要加载的权重
        no_weight_layers = {}
        return layer_name not in no_weight_layers

    def init_weight(self, layer: nn.Module, layer_name: str):
        weights_time_start = time.time()
        # 初始化权重 - 参考ResNet实现
        if isinstance(layer, nn.Sequential):
            # 对于Sequential层（layer1-4），遍历其中的所有模块
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            # 对于单个层
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)

        logger.debug(
            f"Init weight for layer: {layer_name} time: {(time.time() - weights_time_start)*1000:.2f}ms")

    def create_layer(self, layer_name: str) -> Optional[Tuple[nn.Module, float]]:
        """创建单个层实例"""
        try:
            structure_start_ts = time.perf_counter()

            # 根据层名称创建对应的层
            if layer_name == "conv1":
                layer = nn.Conv2d(3, 64, kernel_size=7,
                                  stride=2, padding=3, bias=False)
            elif layer_name == "bn1":
                layer = self.norm_layer(64)
            elif layer_name == "relu":
                layer = nn.ReLU(inplace=True)
            elif layer_name == "maxpool":
                layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            elif layer_name == "avgpool":
                layer = nn.AdaptiveAvgPool2d((1, 1))
            elif layer_name == "fc":
                layer = nn.Linear(512 * self.block_type.expansion, 1000)
            elif layer_name.startswith("layer"):
                layer_idx = int(layer_name[-1])
                current_planes = 64 * (2 ** (layer_idx-1))
                layer = self._make_layer(
                    current_planes,
                    self.layers[layer_idx-1],
                    stride=1 if layer_idx == 1 else 2
                )
            else:
                raise ValueError(f"Unknown layer name: {layer_name}")
            structure_end_ts = time.perf_counter()
            structure_time = structure_end_ts - structure_start_ts

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
                init_weight_time=init_weight_time
            )

            # 如果是没有权重的层，直接标记为已加载
            if not self.has_weights(layer_name):
                self.loading_state.mark_layer_loaded(layer_name)
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
        """按顺序返回ResNet的层序列"""
        layer_names = [
            "conv1", "bn1", "relu", "maxpool",
            "layer1", "layer2", "layer3", "layer4",
            "avgpool", "fc"
        ]

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
            stats.end_time = time.time()
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

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """创建ResNet层"""
        norm_layer = self.norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * self.block_type.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes *
                        self.block_type.expansion, stride),
                norm_layer(planes * self.block_type.expansion),
            )

        layers = []
        layers.append(
            self.block_type(
                self.inplanes, planes, stride, downsample,
                self.groups, self.base_width, self.dilation, norm_layer
            )
        )

        self.inplanes = planes * self.block_type.expansion
        for _ in range(1, blocks):
            layers.append(
                self.block_type(
                    self.inplanes, planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )

        return nn.Sequential(*layers)

    def is_layer_ready(self, layer_name: str) -> bool:
        """检查层是否准备就绪"""
        return self.loading_state.is_layer_loaded(layer_name)
