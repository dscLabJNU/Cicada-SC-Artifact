from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Any, Dict
import torch.nn as nn


class SyncPipelineTask:
    def __init__(self, layer_name: str, layer: nn.Module, structure_ts_info: Dict[str, float]):
        self.layer_name = layer_name
        self.layer = layer
        self.submit_time_ts = structure_ts_info['submit_time_ts']

        self.structure_start_ts = structure_ts_info['structure_start_ts']
        self.structure_end_ts = structure_ts_info['structure_end_ts']

        self.init_weight_start_ts = structure_ts_info['init_weight_start_ts']
        self.init_weight_end_ts = structure_ts_info['init_weight_end_ts']

        self.weight_start_ts = structure_ts_info['weight_start_ts']
        self.weight_end_ts = structure_ts_info['weight_end_ts']

        self.weight_preload_start_ts = None
        self.weight_preload_end_ts = None

        self.compute_start_ts = structure_ts_info['compute_start_ts']
        self.compute_end_ts = structure_ts_info['compute_end_ts']

        self.compute_start_ts = structure_ts_info['compute_start_ts']
        self.compute_end_ts = structure_ts_info['compute_end_ts']

        self.end_time_ts = structure_ts_info['end_time_ts']


class PipelineLoader(ABC):
    """模型流水线加载器基类"""

    @abstractmethod
    def get_layer_sequence(self) -> Iterator[Tuple[str, nn.Module]]:
        """获取层的加载顺序"""
        pass

    # @abstractmethod
    # def is_layer_ready(self, layer_name: str) -> bool:
    #     """检查层是否准备就绪"""
    #     pass


class LayerLoadingState:
    """层加载状态管理"""

    def __init__(self):
        self.loaded_layers: Dict[str, bool] = {}
        self.current_loading_layer: str = None

    def mark_layer_loaded(self, layer_name: str):
        self.loaded_layers[layer_name] = True

    def is_layer_loaded(self, layer_name: str) -> bool:
        return self.loaded_layers.get(layer_name, False)
