import torch
import threading
from queue import Queue
from typing import Dict, Optional, Set
from logger_config import get_logger
import time
import utils
from pathlib import Path

logger = get_logger(__name__)


class WeightsLoader:
    """异步权重加载器"""

    def __init__(self):
        self.weights_queue = Queue()
        self.loaded_weights: Dict[str, Dict] = {}
        self.loading_weights: Set[str] = set()  # 记录正在加载的层
        self.loading_thread = threading.Thread(
            target=self._weights_loading_worker, daemon=True)
        self.loading_thread.start()
        self.current_model = None  # 添加当前模型名称
        self.cache_dir = "persistent_cache"  # 添加缓存目录
        # 添加事件字典用于同步
        self.weight_events: Dict[str, threading.Event] = {}

    def load_layer_weights(self, model_name: str, layer_name: str, cache_dir: str = "persistent_cache") -> Optional[Dict[str, torch.Tensor]]:
        """加载特定层的权重

        Args:
            model_name (str): 模型名称
            layer_name (str): 层名称
            cache_dir (str): 缓存目录路径

        Returns:
            Optional[Dict[str, torch.Tensor]]: 层权重字典，如果加载失败则返回 None
        """
        try:
            # 获取权重文件路径
            cache_path = Path(cache_dir) / model_name
            weight_path = cache_path / f"{layer_name}.pt"

            if not weight_path.exists():
                logger.warning(f"未找到层 {layer_name} 的权重文件, {weight_path} 不存在")
                return None

            # 加载权重
            weights = torch.load(weight_path, weights_only=True)
            if not isinstance(weights, dict):
                logger.error(f"层 {layer_name} 的权重文件格式错误")
                return None

            return weights

        except Exception as e:
            logger.error(f"加载层 {layer_name} 权重时出错: {str(e)}")
            return None

    def start_model_weights_loading(self, model_name: str):
        """预先开始加载模型权重"""
        self.current_model = model_name  # 设置当前模型名称

    def _weights_loading_worker(self):
        """后台权重加载工作线程"""
        while True:
            try:
                layer_name, layer, weights_key = self.weights_queue.get()
                if layer_name is None:  # 停止信号
                    break

                if layer_name not in self.loaded_weights and layer_name not in self.loading_weights:
                    self.loading_weights.add(layer_name)
                    start = time.time()

                    weights = self.load_layer_weights(
                        model_name=self.current_model,
                        layer_name=layer_name,
                        cache_dir=self.cache_dir
                    )

                    if weights is not None:
                        try:
                            layer.load_state_dict(weights)
                            logger.info(
                                f"层 {layer_name} 权重加载并应用完成，耗时: {(time.time() - start)*1000:.2f}ms")
                        except Exception as e:
                            logger.error(f"应用层 {layer_name} 权重时出错: {str(e)}")
                    else:
                        logger.error(f"层 {layer_name} 权重加载失败")

                    self.loading_weights.remove(layer_name)
                    # 设置完成事件
                    if layer_name in self.weight_events:
                        self.weight_events[layer_name].set()

                self.weights_queue.task_done()
            except Exception as e:
                logger.error(
                    f"_weights_loading_worker 处理层 {layer_name} 时出错: {str(e)}")
                if layer_name in self.loading_weights:
                    self.loading_weights.remove(layer_name)
                if layer_name in self.weight_events:
                    self.weight_events[layer_name].set()
                self.weights_queue.task_done()

    def queue_layer_weights(self, layer_name: str, layer: torch.nn.Module, weights_key: Optional[str]):
        """将层添加到权重加载队列"""
        if weights_key:
            # 创建该层的完成事件
            self.weight_events[layer_name] = threading.Event()
            self.weights_queue.put((layer_name, layer, weights_key))

            # 为层添加等待方法
            def wait_for_weights(timeout=None, layer_name=layer_name):
                """等待权重加载完成"""
                return self.weight_events[layer_name].wait(timeout)

            layer.wait_for_weights = wait_for_weights

    def get_layer_weights(self, layer_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """获取已加载的层权重"""
        try:
            weight_path = utils.get_layer_weights_path(
                model_name=self.current_model,
                layer_name=layer_name
            )

            if not weight_path:
                logger.warning(f"未找到层 {layer_name} 的权重文件")
                return None

            weights = torch.load(weight_path, weights_only=True)
            if not isinstance(weights, dict):
                logger.error(f"层 {layer_name} 的权重文件格式错误")
                return None

            return weights

        except Exception as e:
            logger.error(f"获取层 {layer_name} 权重时出错: {str(e)}")
            return None

    def wait_for_weights(self, timeout: Optional[float] = None) -> bool:
        """等待所有权重加载完成"""
        try:
            start = time.time()
            self.weights_queue.join()
            if timeout:
                # 检查是否所有事件都已设置
                for event in self.weight_events.values():
                    remaining = timeout - (time.time() - start)
                    if remaining <= 0 or not event.wait(timeout=remaining):
                        return False
            else:
                # 无超时等待所有事件
                for event in self.weight_events.values():
                    event.wait()
            return True
        except Exception as e:
            logger.error(f"等待权重加载完成时出错: {str(e)}")
            return False

    def shutdown(self):
        """关闭加载器"""
        self.weights_queue.put((None, None, None))
        self.loading_thread.join()
        # 清理事件
        self.weight_events.clear()
