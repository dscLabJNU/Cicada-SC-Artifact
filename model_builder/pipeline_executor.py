from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Dict, Any
import threading
import time
import torch.nn as nn
import torch
from logger_config import get_logger
import traceback
import os

logger = get_logger(__name__)

# model_name -> layer_name -> weight_end_ts, weight_start_ts, weight_preload_start_ts, weight_preload_end_ts
completed_weight_tasks = defaultdict(dict)


@dataclass
class PipelineTask:
    """流水线任务"""
    task_id: int  # 新增唯一任务标识
    layer_name: str
    last_task_id: Optional[int] = None  # 新增前序任务ID
    layer: Optional[nn.Module] = None
    weights: Optional[Dict] = None
    input_tensor: Optional[torch.Tensor] = None
    output_tensor: Optional[torch.Tensor] = None

    # 任务提交时间点
    submit_time_ts: float = None

    # Stage 1: 结构构建时间
    structure_start_ts: float = None
    structure_end_ts: float = None

    init_weight_start_ts: float = None
    init_weight_end_ts: float = None

    # Stage 2: 权重加载时间
    weight_start_ts: float = None
    weight_end_ts: float = None
    # 优先级管理相关属性
    priority: str = "NORMAL"
    expected_weight_completion_time: float = None
    is_weight_completed: bool = False
    io_duration: float = None

    # 权重预加载时间
    weight_preload_start_ts: float = None
    weight_preload_end_ts: float = None

    # Stage 3: 推理执行时间
    compute_start_ts: float = None
    compute_end_ts: float = None

    end_time_ts: float = None
    error: Optional[Exception] = None


class PipelineExecutor:
    """三阶段流水线执行器"""

    def __init__(self, num_threads: int = 2, loader=None, weights_loader=None, layer_names=None):
        # 保存加载器引用和层名称序列
        self.model_name = loader.model_name
        self.loader = loader  # ResNetPipelineLoader
        self.weights_loader = weights_loader  # WeightsLoader
        self.layer_names = layer_names or []  # 层名称序列
        self.current_layer_idx = 0  # 当前处理的层索引

        # 从环境变量读取配置
        self.enable_preload = os.environ.get(
            'ENABLE_PRELOAD', 'False').lower() == 'true'

        # 预加载的权重缓存
        self.preloaded_weights = {} if self.enable_preload else None
        self.preload_lock = threading.Lock() if self.enable_preload else None

        # 初始化线程池和队列
        self.structure_pool = ThreadPoolExecutor(
            max_workers=num_threads, thread_name_prefix="Structure")
        self.weight_pool = ThreadPoolExecutor(
            max_workers=num_threads, thread_name_prefix="Weight")
        self.compute_pool = ThreadPoolExecutor(
            max_workers=num_threads, thread_name_prefix="Compute")

        # 创建阶段之间的队列
        self.structure_queue = Queue()  # 输入队列
        self.weight_queue = Queue()     # 结构到权重的队列
        self.compute_queue = Queue()    # 权重到计算的队列
        self.result_queue = Queue()     # 输出队列

        # 控制标志
        self.running = True
        self.tasks_in_flight = 0
        self.tasks_lock = threading.Lock()

        # 新增任务管理相关属性
        self.task_counter = 0
        self.completed_tasks = []
        self.stats_lock = threading.Lock()

        # 新增结果缓存和锁
        self.output_cache = {}
        self.cache_lock = threading.Lock()

        # 新增优先级管理相关属性
        self.priority_lock = threading.Lock()
        self.weight_operations = {}  # 存储所有进行中的权重加载操作
        self.priority_adjustment_factor = float(
            os.environ.get('PRIORITY_ADJUSTMENT_FACTOR', '1.0'))  # a参数

        # 启动工作线程
        self._start_workers()

    def _start_workers(self):
        """启动各阶段的工作线程"""
        self.structure_thread = threading.Thread(
            target=self._structure_worker, name="Structure-Worker")
        self.weight_thread = threading.Thread(
            target=self._weight_worker, name="Weight-Worker")
        self.compute_thread = threading.Thread(
            target=self._compute_worker, name="Compute-Worker")

        self.structure_thread.start()
        self.weight_thread.start()
        self.compute_thread.start()

    def _structure_worker(self):
        """结构构建阶段工作线程"""
        while self.running:
            try:
                task = self.structure_queue.get(timeout=0.001)
                if task is None:
                    continue

                future = self.structure_pool.submit(
                    self._build_layer_structure, task)
                result = future.result(timeout=30)

                if result and not result.error:
                    self.weight_queue.put(result)
                else:
                    logger.error(f"层 {task.layer_name} 结构构建失败")

            except Empty:
                # 队列为空，继续循环
                continue
            except Exception as e:
                logger.error(f"结构构建工作线程错误: {str(e)}")
                logger.error(traceback.format_exc())  # 打印堆栈跟踪
                continue

    def _weight_worker(self):
        """权重加载阶段工作线程"""
        while self.running:
            try:
                # 尝试从权重队列获取任务
                try:
                    task = self.weight_queue.get(timeout=0.001)
                except Empty:
                    task = None

                if task:
                    # 处理当前任务
                    self._process_weight_task(task)
                elif self.enable_preload and self.current_layer_idx < len(self.layer_names):
                    # 只在启用预加载且队列为空时尝试预加载
                    self._try_preload_next_weights()
                else:
                    continue

            except Exception as e:
                logger.error(f"权重加载工作线程错误: {str(e)}")
                logger.error(traceback.format_exc())
                continue

    def _process_weight_task(self, task: PipelineTask):
        """处理权重加载任务"""
        try:
            task.weight_start_ts = time.perf_counter()

            # 将任务添加到权重操作字典中
            with self.priority_lock:
                self.weight_operations[task.task_id] = task

            # 在加载权重前检查优先级
            priority = self.adjust_retrieval_priority(task)

            if priority == "LOW":
                time.sleep(0.001)  # 短暂暂停以让出资源

            if self.loader.has_weights(task.layer_name):
                if self.enable_preload and task.layer_name in self.preloaded_weights:
                    # 获取预加载的元数据
                    preload_data = self.preloaded_weights.pop(task.layer_name)
                    weights = preload_data['weights']
                    # 记录预加载时间到当前任务
                    task.weight_preload_start_ts = preload_data['weight_preload_start_ts']
                    task.weight_preload_end_ts = preload_data['weight_preload_end_ts']
                else:
                    # 正常加载流程 - 在加载过程中再次检查优先级
                    weights = self.weights_loader.load_layer_weights(
                        model_name=self.loader.model_name,
                        layer_name=task.layer_name
                    )

                    # 加载完成后再次检查优先级（可能已经超时）
                    self.adjust_retrieval_priority(task)

                # 加载权重到层
                task.layer.load_state_dict(weights)

            # 更新当前层索引
            self.current_layer_idx = max(
                self.current_layer_idx,
                self.layer_names.index(task.layer_name)
            )

            # 标记任务已完成
            task.is_weight_completed = True
            completed_weight_tasks[self.model_name][task.layer_name] = {
                "weight_preload_start_ts": task.weight_preload_start_ts,
                "weight_preload_end_ts": task.weight_preload_end_ts,
                "weight_start_ts": task.weight_start_ts,
                "weight_end_ts": task.weight_end_ts,
            }

            # 从权重操作字典中移除任务
            with self.priority_lock:
                self.weight_operations.pop(task.task_id, None)

            task.weight_end_ts = time.perf_counter()
            weight_time = task.weight_end_ts - task.weight_start_ts
            logger.debug(
                f"Stage 2 - 层 {task.layer_name} 权重加载完成，耗时: {weight_time*1000:.2f}ms，优先级: {task.priority}")
            self.compute_queue.put(task)

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"层 {task.layer_name}处理权重任务时出错: {str(e)}")
            task.error = e

            # 从权重操作字典中移除任务
            with self.priority_lock:
                self.weight_operations.pop(task.task_id, None)

            return task

    def _try_preload_next_weights(self):
        """尝试预加载下一层的权重"""
        try:
            with self.preload_lock:
                # 保持当前层索引不变
                next_layer_idx = self.current_layer_idx + 1
                if next_layer_idx >= len(self.layer_names):
                    return

                next_layer_name = self.layer_names[next_layer_idx]
                if not self.loader.has_weights(next_layer_name):
                    return

                # 跳过已预加载的层
                if next_layer_name in self.preloaded_weights:
                    return

                # 创建临时任务记录预加载时间
                weight_preload_start_ts = time.perf_counter()

                # 执行预加载
                weights = self.weights_loader.load_layer_weights(
                    model_name=self.loader.model_name,
                    layer_name=next_layer_name
                )

                weight_preload_end_ts = time.perf_counter()
                weight_preload_time = weight_preload_end_ts - weight_preload_start_ts

                if weights:
                    # 存储预加载时间和权重
                    self.preloaded_weights[next_layer_name] = {
                        'weights': weights,
                        'weight_preload_start_ts': weight_preload_start_ts,
                        'weight_preload_end_ts': weight_preload_end_ts
                    }
                    logger.debug(
                        f"成功预加载 {next_layer_name} 权重，耗时: {weight_preload_time*1000:.2f}ms")

        except Exception as e:
            logger.error(f"预加载权重时出错: {str(e)}")
            logger.error(traceback.format_exc())

    def _compute_worker(self):
        """计算阶段工作线程"""
        while self.running:
            try:
                task = self.compute_queue.get(timeout=0.001)
                if task is None:
                    continue

                # 记录 Stage 3 开始时间
                task.compute_start_ts = time.perf_counter()

                future = self.compute_pool.submit(
                    self._run_layer_inference, task)
                result = future.result(timeout=30)

                if result:
                    # 记录 Stage 3 结束时间
                    result.compute_end_ts = time.perf_counter()
                    compute_time = result.compute_end_ts - result.compute_start_ts
                    logger.debug(
                        f"Stage 3 - 层 {result.layer_name} 推理完成，"
                        f"耗时: {compute_time*1000:.2f}ms"
                    )
                    # 计算总的流水线时间
                    self._calculate_pipeline_stats(result)
                    self.result_queue.put(result)
                    with self.tasks_lock:
                        self.tasks_in_flight -= 1

            except Empty:
                # 队列为空，继续循环
                continue
            except Exception as e:
                logger.error(f"计算工作线程错误: {str(e)}")
                logger.error(traceback.format_exc())  # 打印堆栈跟踪
                continue

    def _run_layer_inference(self, task: PipelineTask) -> Optional[PipelineTask]:
        """执行层推理 (Stage 3)"""
        try:
            # 通过前序任务ID获取输入张量
            if task.last_task_id != 0:
                with self.cache_lock:
                    task.input_tensor = self.output_cache.get(
                        task.last_task_id)
                    # 获取后立即清除缓存
                    self.output_cache.pop(task.last_task_id, None)
                    logger.debug(
                        f"input_tensor of task {task.task_id}: {len(task.input_tensor)}")

            if task.input_tensor is None or task.layer is None:
                logger.error(f"层 {task.layer_name} 推理时输入或层为空")
                return task

            with torch.no_grad():
                if task.layer_name == 'fc':
                    task.input_tensor = task.input_tensor.view(
                        task.input_tensor.size(0), -1)
                elif task.layer_name == "class_token":
                    # 添加class token到patch嵌入结果中
                    task.input_tensor = task.layer.expand(
                        task.input_tensor.shape[0], -1, -1)  # [B, 1, D]
                elif task.layer_name == "classifier":
                    task.input_tensor = task.input_tensor.view(
                        task.input_tensor.size(0), -1)
                task.output_tensor = task.layer(task.input_tensor)

                # 缓存当前结果
                with self.cache_lock:
                    self.output_cache[task.task_id] = task.output_tensor.detach(
                    ).clone()

            return task

        except Exception as e:
            import traceback
            logger.error(f"执行层 {task.layer_name} 推理时出错: {str(e)}")
            logger.error(traceback.format_exc())
            task.error = e
            return task

    def _calculate_pipeline_stats(self, task: PipelineTask):
        """收集统计信息到内存"""
        stats = {
            'task_id': task.task_id,
            'model_name': self.loader.model_name,
            'layer_name': task.layer_name,
            'submit_time_ts': task.submit_time_ts,

            'structure_start_ts': task.structure_start_ts,
            'structure_end_ts': task.structure_end_ts,

            'init_weight_start_ts': task.init_weight_start_ts,
            'init_weight_end_ts': task.init_weight_end_ts,

            'weight_start_ts': task.weight_start_ts,
            'weight_end_ts': task.weight_end_ts,
            "weight_preload_start_ts": task.weight_preload_start_ts,
            "weight_preload_end_ts": task.weight_preload_end_ts,

            'compute_start_ts': task.compute_start_ts,
            'compute_end_ts': task.compute_end_ts,
        }

        with self.stats_lock:
            self.completed_tasks.append(stats)

    def submit_task(self, layer_name: str, input_tensor: Optional[torch.Tensor] = None, io_duration: Optional[float] = None) -> int:
        """提交任务到流水线

        Args:
            layer_name: 层名称
            input_tensor: 输入张量
            io_duration: 估计的I/O操作持续时间(秒)，用于优先级调整

        Returns:
            任务ID
        """
        with self.tasks_lock:
            self.task_counter += 1
            task_id = self.task_counter
            last_task_id = task_id - 1
        logger.debug(f"received task model_name: ({self.model_name})")
        # 如果未提供io_duration，使用默认估计
        if io_duration is None and self.enable_preload:
            logger.debug(
                f"completed_weight_tasks: {completed_weight_tasks[self.model_name]}")
            completed_weight_layers = completed_weight_tasks[self.model_name]

            # 从历史数据中估计，或使用默认值
            if layer_name in completed_weight_layers:
                history_info = completed_weight_layers[layer_name]
                weight_preload_end_ts = history_info['weight_preload_end_ts']
                weight_preload_start_ts = history_info['weight_preload_start_ts']
                weight_start_ts = history_info['weight_start_ts']
                weight_end_ts = history_info['weight_end_ts']
                if weight_preload_end_ts is not None and weight_preload_start_ts is not None:
                    io_duration = weight_preload_end_ts - weight_preload_start_ts
                elif weight_end_ts is not None and weight_start_ts is not None:
                    io_duration = weight_end_ts - weight_start_ts
                else:
                    io_duration = 0.1
            else:
                io_duration = 0.1
        else:
            io_duration = 0.1
        logger.debug(f"任务 {task_id} ({layer_name}) 的I/O持续时间: {io_duration}")
        task = PipelineTask(
            task_id=task_id,
            layer_name=layer_name,
            input_tensor=input_tensor,
            submit_time_ts=time.perf_counter(),
            last_task_id=last_task_id,
            io_duration=io_duration  # 添加I/O持续时间
        )

        with self.tasks_lock:
            self.tasks_in_flight += 1
        self.structure_queue.put(task)

        return task_id

    def get_result(self, timeout: float = None) -> Optional[PipelineTask]:
        """获取计算结果"""
        try:
            result = self.result_queue.get(timeout=timeout)
            result.end_time_ts = time.perf_counter()
            return result
        except Exception:
            return None

    def shutdown(self):
        """关闭流水线执行器"""
        self.running = False
        self.structure_pool.shutdown()
        self.weight_pool.shutdown()
        self.compute_pool.shutdown()

        # 清空未完成任务
        while not self.result_queue.empty():
            self.result_queue.get()

    def __del__(self):
        """析构函数，确保文件被关闭"""
        if hasattr(self, 'log_file'):
            self.log_file.close()

    def _build_layer_structure(self, task: PipelineTask) -> Optional[PipelineTask]:
        """构建层结构 (Stage 1)"""
        try:
            # 从 ResNetPipelineLoader 获取层结构
            layer_info = self.loader.create_layer(task.layer_name)
            if layer_info:
                layer, structure_ts_info = layer_info
                task.layer = layer

                task.structure_start_ts = structure_ts_info["structure_start_ts"]
                task.expected_weight_completion_time = task.structure_start_ts + task.io_duration
                task.structure_end_ts = structure_ts_info["structure_end_ts"]
                structure_time = task.structure_end_ts - task.structure_start_ts

                task.init_weight_start_ts = structure_ts_info["init_weight_start_ts"]
                task.init_weight_end_ts = structure_ts_info["init_weight_end_ts"]
                init_weight_time = task.init_weight_end_ts - task.init_weight_start_ts
                logger.debug(
                    f"Stage 1 - 层 {task.layer_name} 结构构建完成，耗时: {structure_time*1000:.2f}ms, 初始化权重: {init_weight_time*1000:.2f}ms")
                return task
            else:
                raise RuntimeError(f"无法创建层 {task.layer_name}")

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"构建层 {task.layer_name} 结构时出错: {str(e)}")
            task.error = e
            return task

    # def get_task_stats(self) -> list:
    #     """获取排序后的任务统计"""
    #     with self.stats_lock:
    #         return sorted(self.completed_tasks, key=lambda x: x['task_id'])

    def adjust_retrieval_priority(self, task: PipelineTask) -> str:
        """
        调整检索操作优先级

        实现算法1: 当操作超过预期完成时间且未完成时，提高其优先级

        Args:
            task: 当前任务

        Returns:
            task的优先级状态
        """
        if not self.enable_preload:
            return "NORMAL"
        current_duration = time.perf_counter() - task.structure_start_ts

        if current_duration >= task.expected_weight_completion_time and not task.is_weight_completed:
            with self.priority_lock:
                # 对于每个检索操作R，如果R != 当前操作，则阻塞R
                for op_id, op_task in self.weight_operations.items():
                    if op_id != task.task_id:
                        # 暂停低优先级I/O操作
                        logger.debug(
                            f"暂停任务 {op_id} ({op_task.layer_name}) 以优先处理延迟任务 {task.task_id} ({task.layer_name})")
                        op_task.priority = "LOW"

                # 设置当前任务为高优先级
                task.priority = "HIGH"
                logger.info(
                    f"任务 {task.task_id} ({task.layer_name}) 优先级已提升至HIGH - 超过预期完成时间")

        return task.priority
