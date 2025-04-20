import torch.nn as nn
from typing import Iterator, Tuple, Dict, Optional, List, Union
from .vit_pipeline_loader import ViTPipelineLoader
from weights_loader import WeightsLoader
import time
from logger_config import get_logger
import threading
import torch
from .pipeline_executor import PipelineExecutor, PipelineTask
from .pipeline_loader import SyncPipelineTask
import os

logger = get_logger(__name__)


class ViTPipelineBuilder:
    """专门用于Pipeline方式加载ViT的构建器"""

    def __init__(self):
        self.pipeline_loaders = {}
        self.weights_loader = WeightsLoader()
        self.pending_weights: Dict[str, str] = {}
        self.loaded_weights: Dict[str, dict] = {}
        self.weight_events: Dict[str, threading.Event] = {}
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

    def create_pipeline_loader(self, model_name: str) -> ViTPipelineLoader:
        """创建流水线加载器"""
        if model_name not in self.pipeline_loaders:
            # 获取模型配置
            configs = {
                'vit_b_16': {
                    'image_size': 224,
                    'patch_size': 16,
                    'num_layers': 12,
                    'num_heads': 12,
                    'hidden_dim': 768,
                    'mlp_dim': 3072,
                    'num_classes': 1000,
                    'representation_size': None
                },
                'vit_b_32': {
                    'image_size': 224,
                    'patch_size': 32,
                    'num_layers': 12,
                    'num_heads': 12,
                    'hidden_dim': 768,
                    'mlp_dim': 3072,
                    'dropout': 0.0,
                    'attention_dropout': 0.0,
                    'num_classes': 1000,
                    'representation_size': None
                },
                'vit_l_16': {
                    'image_size': 224,
                    'patch_size': 16,
                    'num_layers': 24,
                    'num_heads': 16,
                    'hidden_dim': 1024,
                    'mlp_dim': 4096,
                    'dropout': 0.0,
                    'attention_dropout': 0.0,
                    'num_classes': 1000,
                    'representation_size': None
                },
                'vit_h_14': {
                    'image_size': 518,  # 修改为正确的图像尺寸
                    'patch_size': 14,
                    'num_layers': 32,
                    'num_heads': 16,
                    'hidden_dim': 1280,
                    'mlp_dim': 5120,
                    'dropout': 0.0,
                    'attention_dropout': 0.0,
                    'num_classes': 1000,
                    'representation_size': None
                }
            }

            if model_name not in configs:
                raise ValueError(f"Unsupported ViT model: {model_name}")

            # 创建加载器
            self.pipeline_loaders[model_name] = ViTPipelineLoader(
                model_name=model_name,
                config=configs[model_name]
                )

        return self.pipeline_loaders[model_name]

    def load_model_pipeline_async(self, model_name: str, weights_key: Optional[str] = None, test_input: Optional[torch.Tensor] = None) -> Iterator[PipelineTask]:
        """使用并行流水线方式加载模型"""
        try:
            loader = self.create_pipeline_loader(model_name)

            layer_names = [
                "conv_proj",
                "class_token",
                # "pos_embed",
                "encoder",
                "heads"
            ]

            pipeline = PipelineExecutor(
                loader=loader,
                weights_loader=self.weights_loader,
                layer_names=layer_names
            )

            # 准备输入张量字典
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

    def load_model_pipeline_sync(self, model_name: str, weights_key: str = None, test_input: Optional[torch.Tensor] = None) -> Iterator[Tuple[str, nn.Module]]:
        """同步流水线方式加载模型"""
        try:
            loader = self.create_pipeline_loader(model_name)
            current_output = test_input.clone() if test_input is not None else None

            submit_time_ts = time.perf_counter()
            for layer_name, layer, structure_ts_info in loader.get_layer_sequence():
                # 使用 weights_loader 加载层权重
                weights = None
                weight_start_ts = time.perf_counter()
                if loader.has_weights(layer_name):
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

    def _run_inference(self, layer_name: str, layer: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
        """执行ViT特定层推理"""
        with torch.no_grad():
            try:
                # 特殊处理卷积投影层
                if layer_name == "conv_proj":
                    B, C, H, W = input_tensor.shape
                    output = layer(input_tensor).flatten(2).transpose(1, 2)
                # 处理class token拼接
                elif layer_name == "class_token":
                    # 添加class token到patch嵌入结果中
                    class_token = layer.expand(
                        input_tensor.shape[0], -1, -1)  # [B, 1, D]
                    output = torch.cat(
                        [class_token, input_tensor], dim=1)  # [B, N+1, D]
                # 处理分类头输入
                elif layer_name == "heads":
                    output = layer(input_tensor[:, 0])  # 使用class token
                else:
                    output = layer(input_tensor)

                logger.debug(f"层 {layer_name} 输出形状: {output.shape}")
                return output

            except Exception as e:
                logger.warning(f"层 {layer_name} 推理失败: {str(e)}")
                return None

    def save_layer_stats_to_csv(self) -> None:
        """保存ViT层统计信息到CSV"""
        try:
            import csv
            os.makedirs('logs', exist_ok=True)

            file_exists = os.path.exists(self.stats_filename)
            with open(self.stats_filename, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'model_name', 'layer_name', 'create_time', 'weights_time',
                    'compute_time', 'total_time'
                ])

                if not file_exists:
                    writer.writeheader()

                writer.writerows(self.layer_stats)

            logger.info(f"ViT层统计已保存至: {self.stats_filename}")

        except Exception as e:
            logger.error(f"保存统计信息失败: {str(e)}")

    def get_layer(self, layer_name: str) -> Optional[nn.Module]:
        """获取指定ViT层实例"""
        for loader in self.pipeline_loaders.values():
            if hasattr(loader, 'layers') and layer_name in loader.layers:
                return loader.layers[layer_name]
        return None

    def __del__(self):
        """资源清理"""
        if hasattr(self, 'weights_loader'):
            self.weights_loader.shutdown()
        logger.info("ViT流水线构建器资源已释放")
