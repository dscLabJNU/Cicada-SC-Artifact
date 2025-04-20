import os
import torch.nn as nn
import torchvision.models as models
from typing import Iterator, Tuple, Dict, Optional, List, Union
from .resnet_pipeline_loader import ResNetPipelineLoader
from weights_loader import WeightsLoader
import time
from logger_config import get_logger
import threading
import torch
from .pipeline_executor import PipelineExecutor, PipelineTask
from .pipeline_loader import SyncPipelineTask

logger = get_logger(__name__)


class ResNetPipelineBuilder():
    """专门用于Pipeline方式加载ResNet的构建器"""

    def __init__(self):
        self.pipeline_loaders = {}
        self.weights_loader = WeightsLoader()
        # 记录待加载权重的层 {layer_name: weights_key}
        self.pending_weights: Dict[str, str] = {}
        # 用于存储已加载的权重
        self.loaded_weights: Dict[str, dict] = {}
        # 用于同步的事件字典
        self.weight_events: Dict[str, threading.Event] = {}
        # 添加性能统计属性
        self.layer_stats: List[Dict[str, Union[str, float]]] = []

        # 从环境变量读取配置
        self.record_stats = os.environ.get(
            'RECORD_LAYER_STATS', 'False').lower() == 'true'
        self.enable_preload = os.environ.get(
            'ENABLE_PRELOAD', 'False').lower() == 'true'
        self.use_miniloader = os.environ.get(
            'USE_MINILOADER', 'False').lower() == 'true'

        # 构建日志文件名
        mode = os.environ.get('PIPELINE_MODE', 'sync')  # 从环境变量获取模式
        preload_status = 'preload' if self.enable_preload else 'nopreload'
        mini_status = 'mini' if self.use_miniloader else 'normal'
        self.stats_filename = f'logs/layer_stats_{mode}_{preload_status}_{mini_status}.csv'

    def create_pipeline_loader(self, model_name: str) -> ResNetPipelineLoader:
        """创建流水线加载器"""
        if model_name not in self.pipeline_loaders:
            # 获取模型配置
            configs = {
                'resnet18': ([2, 2, 2, 2], 'basic'),
                'resnet34': ([3, 4, 6, 3], 'basic'),
                'resnet50': ([3, 4, 6, 3], 'bottleneck'),
                'resnet101': ([3, 4, 23, 3], 'bottleneck'),
                'resnet152': ([3, 8, 36, 3], 'bottleneck'),
                'wide_resnet50_2': ([3, 4, 6, 3], 'bottleneck'),
                'wide_resnet101_2': ([3, 4, 23, 3], 'bottleneck')
            }

            if model_name not in configs:
                raise ValueError(f"Unsupported ResNet model: {model_name}")

            layers, block_type = configs[model_name]
            block = models.resnet.BasicBlock if block_type == 'basic' else models.resnet.Bottleneck

            # 创建加载器
            self.pipeline_loaders[model_name] = ResNetPipelineLoader(
                model_name=model_name,
                block_type=block,
                layers=layers
            )

        return self.pipeline_loaders[model_name]

    def _run_inference(self, layer_name: str, layer: nn.Module, current_output: torch.Tensor) -> Optional[torch.Tensor]:
        """执行单层推理

        Args:
            layer_name (str): 层名称
            layer (nn.Module): 层实例
            current_output (torch.Tensor): 当前输入张量
            loader (ResNetPipelineLoader): 加载器实例

        Returns:
            Optional[torch.Tensor]: 推理结果，如果失败则返回 None
        """
        if current_output is None:
            return None

        with torch.no_grad():
            try:
                if layer_name == 'fc':
                    current_output = current_output.view(
                        current_output.size(0), -1)

                output = layer(current_output)
                logger.info(
                    f"Layer {layer_name} output shape: {output.shape}")
                return output

            except Exception as e:
                logger.warning(
                    f"Layer {layer_name} inference failed: {str(e)}")
                return None

    def load_model_pipeline_sync(self, model_name: str, weights_key: str = None, test_input: Optional[torch.Tensor] = None) -> Iterator[Tuple[str, nn.Module]]:
        """同步流水线方式加载模型"""
        try:
            loader = self.create_pipeline_loader(model_name)
            current_output = test_input.clone() if test_input is not None else None

            submit_time_ts = time.perf_counter()
            for layer_name, layer, structure_ts_info in loader.get_layer_sequence():
                # 使用 weights_loader 加载层权重
                weight_start_ts = time.perf_counter()
                weights = self.weights_loader.load_layer_weights(
                    model_name=model_name,
                    layer_name=layer_name
                )

                if weights:
                    layer.load_state_dict(weights)
                weight_end_ts = time.perf_counter()
                weight_time_ms = (weight_end_ts - weight_start_ts) * 1000
                structure_ts_info["weight_start_ts"] = weight_start_ts
                structure_ts_info["weight_end_ts"] = weight_end_ts

                compute_start_ts = time.perf_counter()
                # 执行推理
                if current_output is not None:
                    current_output = self._run_inference(
                        layer_name, layer, current_output)
                compute_end_ts = time.perf_counter()
                compute_time_ms = (compute_end_ts - compute_start_ts) * 1000

                structure_ts_info["compute_start_ts"] = compute_start_ts
                structure_ts_info["compute_end_ts"] = compute_end_ts
                structure_ts_info['submit_time_ts'] = submit_time_ts
                structure_ts_info['end_time_ts'] = time.perf_counter()

                # 仅在需要时记录统计信息
                if self.record_stats:
                    structure_time_ms = (structure_ts_info["structure_end_ts"] -
                                         structure_ts_info["structure_start_ts"]) * 1000
                    init_weight_time_ms = (structure_ts_info["init_weight_end_ts"] -
                                           structure_ts_info["init_weight_start_ts"]) * 1000
                    self.layer_stats.append({
                        'model_name': model_name,
                        'layer_name': layer_name,
                        'structure_time_ms': structure_time_ms,
                        'init_weight_time_ms': init_weight_time_ms,
                        'weight_time_ms': weight_time_ms,
                        'compute_time_ms': compute_time_ms,
                    })

                    logger.info(
                        f"层 {layer_name} - 结构构建耗时: {structure_time_ms:.2f}ms, 加载权重耗时: {weight_time_ms:.2f}ms, 计算耗时: {compute_time_ms:.2f}ms")

                yield SyncPipelineTask(
                    layer_name=layer_name,
                    layer=layer,
                    structure_ts_info=structure_ts_info,
                )

        except Exception as e:
            logger.error(f"同步加载模型 {model_name} 时出错: {str(e)}")
            raise

    def load_model_pipeline_async(self, model_name: str, weights_key: Optional[str] = None, test_input: Optional[torch.Tensor] = None) -> Iterator[PipelineTask]:
        """使用并行流水线方式加载模型"""
        try:
            # 创建加载器
            loader = self.create_pipeline_loader(model_name)

            # 获取层名称序列
            layer_names = [
                "conv1", "bn1", "relu", "maxpool",
                "layer1", "layer2", "layer3", "layer4",
                "avgpool", "fc"
            ]

            # 创建流水线执行器
            pipeline = PipelineExecutor(
                loader=loader,
                weights_loader=self.weights_loader,
                layer_names=layer_names
            )

            # 准备输入张量字典，用于跟踪每层的输入
            layer_inputs = {}
            if test_input is not None:
                layer_inputs[layer_names[0]] = test_input.clone()

            # 提交所有任务到流水线
            for layer_name in layer_names:
                pipeline.submit_task(
                    layer_name=layer_name,
                    input_tensor=layer_inputs.get(layer_name)
                )

            # 获取结果并处理
            completed_layers = 0
            while completed_layers < len(layer_names):
                layer_result = pipeline.get_result(timeout=30)
                if layer_result and not layer_result.error:
                    completed_layers += 1
                    yield layer_result

            # 关闭流水线
            pipeline.shutdown()

        except Exception as e:
            logger.error(f"异步加载模型 {model_name} 时出错: {str(e)}")
            raise

    def apply_pending_weights(self, timeout: Optional[float] = None) -> bool:
        """应用所有待加载的权重

        Args:
            timeout: 等待权重加载的超时时间（秒），None表示一直等待

        Returns:
            bool: 是否所有权重都成功加载并应用
        """
        try:
            if not self.pending_weights:
                return True

            logger.info("开始应用待加载的权重...")
            success = self.weights_loader.wait_for_weights(timeout)

            if not success:
                logger.warning("等待权重加载超时")
                return False

            # 应用已加载的权重
            for layer_name, weights_key in self.pending_weights.items():
                weights = self.weights_loader.get_layer_weights(layer_name)
                if weights:
                    layer = self.get_layer(layer_name)
                    if layer:
                        start = time.time()
                        layer.load_state_dict(weights)
                        logger.info(
                            f"层 {layer_name} 权重应用完成，耗时: {time.time() - start:.2f}s")
                    else:
                        logger.warning(f"找不到层 {layer_name}")
                else:
                    logger.warning(f"层 {layer_name} 的权重加载失败pipeline_builder")
                    return False

            self.pending_weights.clear()
            return True

        except Exception as e:
            logger.error(f"应用权重时出错: {str(e)}")
            return False

    def get_layer(self, layer_name: str) -> Optional[nn.Module]:
        """获取指定名称的层实例"""
        for loader in self.pipeline_loaders.values():
            if hasattr(loader, 'layers') and layer_name in loader.layers:
                return loader.layers[layer_name]
        return None

    def get_model(self, model_name: str) -> nn.Module:
        """重写父类方法，抛出异常提示使用pipeline方法"""
        raise NotImplementedError(
            "ResNetPipelineBuilder 不支持直接获取完整模型，请使用 load_model_pipeline 方法进行流水线加载"
        )

    def save_layer_stats_to_csv(self) -> None:
        """将层统计信息保存到CSV文件"""
        try:
            import csv
            import os

            # 确保日志目录存在
            os.makedirs('logs', exist_ok=True)

            file_exists = os.path.exists(self.stats_filename)

            # 以追加模式打开文件
            with open(self.stats_filename, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'model_name', 'layer_name', 'create_time_ms', 'weights_time_ms'
                ])

                # 如果文件不存在，写入表头
                if not file_exists:
                    writer.writeheader()

                # 写入当前模型的层统计信息
                writer.writerows(self.layer_stats)

            logger.info(f"层统计信息已追加到: {self.stats_filename}")

        except Exception as e:
            logger.error(f"保存层统计信息时出错: {str(e)}")

    def __del__(self):
        """确保清理资源"""
        if hasattr(self, 'weights_loader'):
            self.weights_loader.shutdown()
