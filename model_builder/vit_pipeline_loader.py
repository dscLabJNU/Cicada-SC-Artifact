from .pipeline_loader import PipelineLoader, LayerLoadingState
import torch.nn as nn
from typing import Iterator, Tuple, Dict, Optional, List
from utils import get_logger
import time
from dataclasses import dataclass
from torchvision.models.vision_transformer import MLPBlock, EncoderBlock, Encoder
import torch
from functools import partial
from collections import OrderedDict
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


class ViTPipelineLoader(PipelineLoader):
    def __init__(self, model_name: str, config: Dict):
        """初始化 ViT pipeline 加载器

        Args:
            model_name (str): 模型名称
            config (Dict): 模型配置
        """
        self.model_name = model_name
        self.config = config
        self.loading_state = LayerLoadingState()

        # 计算序列长度
        self.seq_length = (config['image_size'] //
                           config['patch_size']) ** 2 + 1

        # 层统计信息
        self.layer_stats: Dict[str, LayerStats] = {}

        # 从环境变量读取配置
        self.use_miniloader = os.environ.get(
            'USE_MINILOADER', 'False').lower() == 'true'

    def init_weight(self, layer: nn.Module, layer_name: str):
        weights_time_start = time.perf_counter()
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        logger.debug(
            f"Init weight for layer: {layer_name} time: {(time.perf_counter() - weights_time_start)*1000:.2f}ms")

    def create_layer(self, layer_name: str) -> Optional[Tuple[nn.Module, Dict]]:
        """创建指定的层"""
        try:
            structure_start_ts = time.perf_counter()
            layer = None

            # 参数层封装器
            class ParameterWrapper(nn.Module):
                def __init__(self, param: nn.Parameter):
                    super().__init__()
                    self.param = param

                def expand(self, batch_size: int, dim1: int, dim2: int):
                    return self.param.expand(batch_size, dim1, dim2)

                def forward(self, x):
                    return x + self.param  # 适用于pos_embed和class_token的加法操作

            # 创建层逻辑
            if layer_name == "conv_proj":
                layer = nn.Conv2d(
                    in_channels=3,
                    out_channels=self.config['hidden_dim'],
                    kernel_size=self.config['patch_size'],
                    stride=self.config['patch_size'],
                )
            elif layer_name == "class_token":
                param = nn.Parameter(torch.empty(
                    1, 1, self.config['hidden_dim']).normal_(std=0.02))
                layer = ParameterWrapper(param)  # 封装Parameter
            elif layer_name == "pos_embed":
                param = nn.Parameter(torch.empty(
                    1, self.seq_length, self.config['hidden_dim']).normal_(std=0.02))
                layer = ParameterWrapper(param)  # 封装Parameter
            elif layer_name == "encoder":
                layer = Encoder(
                    seq_length=self.seq_length,
                    num_layers=self.config['num_layers'],
                    num_heads=self.config['num_heads'],
                    hidden_dim=self.config['hidden_dim'],
                    mlp_dim=self.config['mlp_dim'],
                    dropout=self.config.get('dropout', 0.0),
                    attention_dropout=self.config.get(
                        'attention_dropout', 0.0),
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    init_weights=False
                )
            elif layer_name == "heads":
                # 重构分类头构建逻辑
                heads_layers = OrderedDict()
                representation_size = self.config.get('representation_size')

                if representation_size:
                    heads_layers["pre_logits"] = nn.Linear(
                        self.config['hidden_dim'],
                        representation_size
                    )
                    heads_layers["act"] = nn.Tanh()
                    heads_layers["head"] = nn.Linear(
                        representation_size,
                        self.config['num_classes']
                    )
                else:
                    heads_layers["head"] = nn.Linear(
                        self.config['hidden_dim'],
                        self.config['num_classes']
                    )

                layer = nn.Sequential(heads_layers)

            if layer is None:
                raise ValueError(f"Unknown layer name: {layer_name}")

            structure_end_ts = time.perf_counter()
            structure_time = structure_end_ts - structure_start_ts

            init_weight_start_ts = time.perf_counter()
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

            # 构造时间戳信息字典
            structure_ts_info = {
                "structure_start_ts": structure_start_ts,
                "structure_end_ts": structure_end_ts,
                "init_weight_start_ts": init_weight_start_ts,
                "init_weight_end_ts": init_weight_end_ts,
            }

            # 如果是没有权重的层，直接标记为已加载
            if not self.has_weights(layer_name):
                self.loading_state.mark_layer_loaded(layer_name)

            return layer, structure_ts_info

        except Exception as e:
            logger.error(f"创建层 {layer_name} 时出错: {str(e)}")
            return None

    def get_layer_sequence(self) -> Iterator[Tuple[str, nn.Module, float]]:
        """按顺序返回ViT的层序列"""
        layer_names = [
            "conv_proj",
            "class_token",
            # "pos_embed",
            "encoder",
            "heads"
        ]

        for layer_name in layer_names:
            result = self.create_layer(layer_name)
            if result:
                layer, structure_ts_info = result
                yield layer_name, layer, structure_ts_info

    def has_weights(self, layer_name: str) -> bool:
        """检查层是否有权重需要加载"""
        no_weight_layers = {"class_token", "pos_embed"}
        return layer_name not in no_weight_layers

    def is_layer_ready(self, layer_name: str) -> bool:
        """检查层是否准备就绪"""
        return self.loading_state.is_layer_loaded(layer_name)
