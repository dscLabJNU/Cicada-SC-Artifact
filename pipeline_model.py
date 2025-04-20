import torch
import torch.nn as nn
from typing import Dict, Optional, List, TextIO, Any
from model_builder.model_factory import ModelFactory
from logger_config import get_logger
import utils
import time
from pathlib import Path
import os
import argparse
import glob
from collections import defaultdict
from model_builder.pipeline_executor import PipelineTask
import pandas as pd

logger = get_logger(__name__)

# 在文件顶部添加常量定义
MAX_INVOCATIONS = 100


class PipelineModel:
    """流水线式模型加载和推理"""

    def __init__(self):
        self.layers: Dict[str, nn.Module] = {}
        self.layer_sequence: List[str] = []
        self.current_output = None
        self.is_eval = True
        self.cache_dir = "persistent_cache"
        self.results = defaultdict(list)

        # 从环境变量读取配置
        self.record_stats = os.environ.get(
            'RECORD_LAYER_STATS', 'False').lower() == 'true'
        self.enable_preload = os.environ.get(
            'ENABLE_PRELOAD', 'False').lower() == 'true'
        self.use_miniloader = os.environ.get(
            'USE_MINILOADER', 'False').lower() == 'true'

    def init_weights(self, model_name: str, weights_key: Optional[str] = None):
        """初始化模型权重，将完整权重拆分到各层"""
        try:
            logger.info(f"开始初始化模型 {model_name} 的权重...")

            # 检查是否已经存在拆分后的权重文件
            cache_path = Path(self.cache_dir)
            model_cache_path = cache_path / model_name
            if model_cache_path.exists() and any(model_cache_path.iterdir()):
                logger.info(f"检测到模型 {model_name} 的缓存权重文件，跳过初始化")
                return True

            # 拆分权重并保存到缓存
            layer_weights = utils.split_weights_by_layer(
                model_name=model_name,
                weights_key=weights_key,
                cache_dir=self.cache_dir
            )

            if not layer_weights:
                logger.warning(f"模型 {model_name} 的权重拆分结果为空")
                return False

            logger.info(f"模型 {model_name} 的权重已成功拆分到 {len(layer_weights)} 个层")
            return True

        except Exception as e:
            logger.error(f"初始化权重时出错: {str(e)}")
            return False

    def inference_model(self, model_name: str, weights_key: Optional[str] = None,
                        async_load: bool = False, test_input: Optional[torch.Tensor] = None):
        """流水线方式加载模型

        Args:
            model_name (str): 模型名称
            weights_key (Optional[str]): 权重标识符
            async_load (bool): 是否使用异步加载方式
            test_input (Optional[torch.Tensor]): 用于测试的输入张量
        """
        try:
            # 确保权重已经初始化
            if weights_key and not self.init_weights(model_name, weights_key):
                raise RuntimeError(f"模型 {model_name} 的权重初始化失败")

            # 获取对应的模型构建器，指定使用 pipeline 方式
            model_type = self._get_model_type(model_name)
            builder = ModelFactory.get_builder(model_type, use_pipeline=True)

            # 根据 async_load 参数选择加载方式
            if async_load:
                logger.info(
                    f"使用异步方式加载模型 {model_name}, 预加载{'启用' if self.enable_preload else '禁用'}")
                load_method = builder.load_model_pipeline_async
            else:
                logger.info(f"使用同步方式加载模型 {model_name}")
                load_method = builder.load_model_pipeline_sync

            start_time = time.perf_counter()
            # 使用选定的方法加载模型
            for layer_result in load_method(model_name, weights_key, test_input):
                self.results[model_name].append(layer_result)
                self.layers[layer_result.layer_name] = layer_result.layer
                self.layer_sequence.append(layer_result.layer_name)

            inference_time = (time.perf_counter() - start_time) * 1000

            # 保存层统计信息
            if self.record_stats:
                builder.save_layer_stats_to_csv()

            return inference_time

        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            raise

    def _get_model_type(self, model_name: str) -> str:
        """从模型名称中提取模型类型"""
        if model_name.startswith('resnet'):
            return 'resnet'
        elif model_name.startswith('wide_resnet'):
            return 'resnet'
        elif model_name.startswith('vgg'):
            return 'vgg'
        elif model_name.startswith('vit'):
            return 'vit'
        elif model_name.startswith('densenet'):
            return 'densenet'
        elif model_name.startswith('mobilenet'):
            return 'mobilenet'
        elif model_name.startswith('efficientnet'):
            return 'efficientnet'
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    @staticmethod
    def _init_logging(family_name: str) -> tuple[TextIO, TextIO]:
        """初始化日志文件和路径

        Args:
            family_name (str): 模型家族名称

        Returns:
            tuple[TextIO, TextIO]: 返回模型评估文件句柄和层性能文件句柄
        """
        # 创建日志目录
        log_dir = './logs/'
        os.makedirs(log_dir, exist_ok=True)

        # 获取环境变量状态
        preload_status = 'preload' if os.environ.get(
            'ENABLE_PRELOAD', 'False').lower() == 'true' else 'nopreload'
        mini_status = 'mini' if os.environ.get(
            'USE_MINILOADER', 'False').lower() == 'true' else 'normal'
        load_mode = os.environ.get('PIPELINE_MODE', 'sync')

        # 生成日志文件路径
        model_eval_path = os.path.join(
            log_dir, f'model_eval_{family_name}_{load_mode}_{preload_status}_{mini_status}.csv')
        layer_perf_path = os.path.join(
            log_dir, f"pipeline_stats_allmodels_{load_mode}_{preload_status}_{mini_status}.csv")

        # 初始化模型评估文件
        model_eval_file = open(model_eval_path, 'a', newline='')

        # 修复表头写入条件：同时检查文件是否存在和是否为空
        if not os.path.exists(model_eval_path) or os.path.getsize(model_eval_path) == 0:
            print(
                'model_name,structure_load_time,'
                'before_param_count,before_param_mb,before_buffer_mb,before_serialized_mb,before_instance_mb,before_total_mb,'
                'after_param_count,after_param_mb,after_buffer_mb,after_serialized_mb,after_instance_mb,after_total_mb',
                file=model_eval_file, flush=True)

        # 初始化层性能数据文件
        layer_perf_file = open(layer_perf_path, 'a', newline='')

        if not os.path.exists(layer_perf_path) or os.path.getsize(layer_perf_path) == 0:
            print(
                "model_name,layer_name,"
                "submit_time_ts,end_time_ts,"
                "structure_start_ts,structure_end_ts,"
                "init_weight_start_ts,init_weight_end_ts,"

                "weight_start_ts,weight_end_ts,"
                "weight_preload_start_ts,weight_preload_end_ts,"

                "compute_start_ts,compute_end_ts,",
                file=layer_perf_file, flush=True)

        return model_eval_file, layer_perf_file

    @staticmethod
    def _run_inference_test(pipeline_model: 'PipelineModel', model_name: str, weights_key: str,
                            layer_perf_file: TextIO) -> tuple[float, dict]:
        """运行模型推理测试

        Args:
            pipeline_model: PipelineModel实例
            model_name (str): 模型名称
            weights_key (str): 权重键值
            layer_perf_file (TextIO): 层性能文件句柄

        Returns:
            tuple[float, dict]: 返回推理时间、加载前内存状态
        """
        # 创建测试输入
        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = 'cpu'
        test_inputs = [torch.randn(1, 3, 224, 224, device=device)]

        # 加载模型并进行推理
        load_mode = os.environ.get('PIPELINE_MODE', 'sync')
        inference_time = pipeline_model.inference_model(
            model_name, weights_key,
            async_load=load_mode == "async",
            test_input=test_inputs[0]
        )

        # 记录层性能数据
        for layer_result in pipeline_model.results[model_name]:
            PipelineModel._record_layer_performance(
                model_name, layer_result, layer_perf_file)

        memory_stats = PipelineModel._calculate_memory_stats(pipeline_model)

        return inference_time, memory_stats

    @staticmethod
    def test_model_family(test_schedule: List[Dict], family_name: str) -> bool:
        """执行基于时间调度的模型测试（重构版本）"""
        model_eval_file = None
        layer_perf_file = None
        current_idx = 0
        success_count = 0

        try:
            # 初始化日志
            model_eval_file, layer_perf_file = PipelineModel._init_logging(
                family_name)
            logger.info(f"开始执行调度测试，共 {len(test_schedule)} 个任务")

            # 按时间顺序处理任务
            while current_idx < len(test_schedule):
                current_task = test_schedule[current_idx]
                now = time.time()

                # 等待直到执行时间到达
                if now < current_task['execute_time']:
                    time.sleep(current_task['execute_time'] - now)
                    continue

                try:
                    # 执行推理测试
                    pipeline_model = PipelineModel()
                    inference_time, before_memory = PipelineModel._run_inference_test(
                        pipeline_model,
                        current_task['model_name'],
                        current_task['weights_key'],
                        layer_perf_file
                    )

                    # 记录带函数上下文的性能指标
                    PipelineModel._record_model_metrics(
                        current_task['model_name'],
                        inference_time,
                        before_memory,
                        model_eval_file
                    )

                    success_count += 1
                    logger.info(
                        f"任务成功 | 模型: {current_task['model_name']} "
                        f"函数: {current_task['func_name']} "
                        f"时间: {time.ctime(current_task['execute_time'])}"
                    )

                except Exception as e:
                    logger.error(
                        f"任务失败 | 模型: {current_task['model_name']} "
                        f"函数: {current_task['func_name']} 错误: {str(e)}"
                    )

                current_idx += 1

            logger.info(f"调度测试完成: {success_count}/{len(test_schedule)} 成功")
            return success_count == len(test_schedule)

        except Exception as e:
            logger.error(f"调度测试异常终止: {str(e)}")
            return False
        finally:
            if model_eval_file:
                model_eval_file.close()
            if layer_perf_file:
                layer_perf_file.close()

    @staticmethod
    def _record_layer_performance(model_name: str, layer_result: PipelineTask, layer_perf_file: TextIO) -> None:
        """记录层性能数据

        Args:
            model_name (str): 模型名称
            layer_result (Any): 层结果对象
            layer_perf_file (TextIO): 层性能文件句柄
            structure_time (float): 结构构建时间
            weight_time (float): 权重加载时间
            compute_time (float): 计算时间
            structure_wait (float): 结构等待时间
            weight_wait (float): 权重等待时间
            compute_wait (float): 计算等待时间
            total_wait_time (float): 总等待时间
            total_pipeline_time (float): 总流水线时间
            pipeline_efficiency (float): 流水线效率
        """
        r = layer_result
        layer_name = r.layer_name
        structure_start_ts = r.structure_start_ts
        structure_end_ts = r.structure_end_ts
        structure_time_ms = (structure_end_ts - structure_start_ts) * 1000

        structure_wait_ms = (r.structure_start_ts - r.submit_time_ts) * 1000

        init_weight_start_ts = r.init_weight_start_ts
        init_weight_end_ts = r.init_weight_end_ts
        init_weight_time_ms = (init_weight_end_ts -
                               init_weight_start_ts) * 1000

        weight_start_ts = r.weight_start_ts
        weight_end_ts = r.weight_end_ts
        weight_time_ms = (weight_end_ts - weight_start_ts) * 1000
        weight_wait_ms = (r.weight_start_ts - r.structure_end_ts) * 1000

        weight_preload_start_ts = r.weight_preload_start_ts or 0
        weight_preload_end_ts = r.weight_preload_end_ts or 0
        weight_preload_time_ms = (weight_preload_end_ts -
                                  weight_preload_start_ts) * 1000

        compute_start_ts = r.compute_start_ts
        compute_end_ts = r.compute_end_ts
        compute_time_ms = (compute_end_ts - compute_start_ts) * 1000
        compute_wait_ms = (r.compute_start_ts - r.weight_end_ts) * 1000

        total_wait_time_ms = structure_wait_ms + weight_wait_ms + compute_wait_ms
        total_active_time_ms = structure_time_ms + \
            weight_time_ms + compute_time_ms + weight_preload_time_ms

        total_pipeline_time_ms = (r.end_time_ts - r.submit_time_ts) * 1000

        if total_pipeline_time_ms > 0:
            pipeline_efficiency = (total_active_time_ms /
                                   total_pipeline_time_ms) * 100
        else:
            raise ValueError(
                f"total_pipeline_time is less than 0, {model_name}, {layer_name}")

        logger.debug(f"""
            层 {layer_name} 流水线统计:
            - Stage 1 (结构构建): {structure_time_ms:.2f}ms (创建: {structure_time_ms:.2f}ms, 初始化权重: {init_weight_time_ms:.2f}ms), 等待: {structure_wait_ms:.2f}ms
            - Stage 2 (权重加载): {weight_time_ms:.2f}ms, 等待: {weight_wait_ms:.2f}ms
            - Stage 3 (推理执行): {compute_time_ms:.2f}ms, 等待: {compute_wait_ms:.2f}ms
            - 预加载时间: {weight_preload_time_ms:.2f}ms
            - 总处理时间: {total_active_time_ms:.2f}ms
            - 总等待时间: {total_wait_time_ms:.2f}ms
            - 总流水线时间: {total_pipeline_time_ms:.2f}ms
            - 流水线效率: {pipeline_efficiency:.2f}%
            """)

        print(
            f"{model_name},{layer_name},"
            f"{r.submit_time_ts},{r.end_time_ts},"
            f"{structure_start_ts},{structure_end_ts},"
            f"{init_weight_start_ts},{init_weight_end_ts},"

            f"{weight_start_ts},{weight_end_ts},"
            f"{weight_preload_start_ts or ''},{weight_preload_end_ts or ''},"

            f"{compute_start_ts},{compute_end_ts},",
            file=layer_perf_file, flush=True)

        # print(
        #     f"{model_name},{layer_name},"
        #     f"{structure_time:.4f},{structure_wait:.4f},"
        #     f"{weight_time:.4f},{weight_wait:.4f},"
        #     f"{compute_time:.4f},{compute_wait:.4f},"
        #     f"{weight_preload_time:.4f},"
        #     f"{total_wait_time:.4f},{total_pipeline_time:.4f},"
        #     f"{pipeline_efficiency:.4f}",
        #     file=layer_perf_file, flush=True)

    @staticmethod
    def _record_model_metrics(model_name: str, inference_time: float,
                              memory_stats: Dict[str, Dict[str, float]],
                              model_eval_file: TextIO) -> None:
        """记录模型性能指标

        Args:
            model_name (str): 模型名称
            inference_time (float): 推理时间
            memory_stats (Dict[str, Dict[str, float]]): 内存统计信息
            model_eval_file (TextIO): 模型评估文件句柄
        """
        print(f'{model_name},{inference_time},'
              f'{memory_stats["parameters"]["count"]},'
              f'{memory_stats["parameters"]["size_mb"]},'
              f'{memory_stats["buffers"]["size_mb"]},'
              f'{memory_stats["serialized"]["size_mb"]},'
              f'{memory_stats["instance"]["size_mb"]},'
              f'{memory_stats["total"]["size_mb"]},'
              # pipeline模式下before和after相同
              f'{memory_stats["parameters"]["count"]},'
              f'{memory_stats["parameters"]["size_mb"]},'
              f'{memory_stats["buffers"]["size_mb"]},'
              f'{memory_stats["serialized"]["size_mb"]},'
              f'{memory_stats["instance"]["size_mb"]},'
              f'{memory_stats["total"]["size_mb"]}',
              file=model_eval_file, flush=True)

    @staticmethod
    def _calculate_memory_stats(pipeline_model: 'PipelineModel') -> Dict[str, Dict[str, float]]:
        """计算内存使用统计

        Args:
            pipeline_model (PipelineModel): 流水线模型实例

        Returns:
            Dict[str, Dict[str, float]]: 内存统计信息
        """
        memory_stats = {
            "parameters": {"count": 0, "size_mb": 0},
            "buffers": {"size_mb": 0},
            "serialized": {"size_mb": 0},
            "instance": {"size_mb": 0},
            "total": {"size_mb": 0}
        }

        for layer in pipeline_model.layers.values():
            layer_memory = utils.MemoryUtils.get_model_memory_usage(layer)
            for key in memory_stats:
                if key == "parameters":
                    memory_stats[key]["count"] += layer_memory[key]["count"]
                memory_stats[key]["size_mb"] += layer_memory[key]["size_mb"]

        return memory_stats


def setup_logging(mode: str, is_async: bool, enable_preload: bool):
    """设置日志配置

    Args:
        mode (str): 运行模式 ('sync' 或 'async')
        is_async (bool): 是否异步模式
        enable_preload (bool): 是否启用预加载
    """
    # 创建日志目录
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)

    # 清理相关的日志文件
    pattern = f"./logs/model_eval_*_{mode}.csv"
    for file in glob.glob(pattern):
        os.remove(file)

    # 如果是异步模式，清理特定的统计文件
    if is_async:
        preload_status = 'preload' if enable_preload else 'nopreload'
        stats_file = f"./logs/pipeline_stats_allmodels_{preload_status}.csv"
        if os.path.exists(stats_file):
            os.remove(stats_file)

    return log_dir


def get_run_config(args):
    """获取运行配置并返回格式化的配置信息"""
    config = {
        'mode': 'async' if args.async_load else 'sync',
        'device': args.device or ('cuda' if torch.cuda.is_available() else 'cpu'),
        'features': []
    }

    # 收集启用的特性
    if args.async_load:
        config['features'].append('异步加载')
    if args.pre_load:
        config['features'].append('预加载')
    if args.mini_loader:
        config['features'].append('Mini Loader')
    if args.record_stats:
        config['features'].append('统计记录')

    return config


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Pipeline Model Loader')

    # 添加命令行参数
    parser.add_argument('--async-load', action='store_true', default=False,
                        help='使用异步加载模式（默认：同步）')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                        help='指定运行设备 (默认：自动选择)')
    parser.add_argument('--record-stats', action='store_true', default=False,
                        help='记录层创建和权重加载的统计信息 (默认：关闭)')
    parser.add_argument('--pre-load', action='store_true', default=False,
                        help='启用权重预加载 (默认：关闭）')
    parser.add_argument('--mini-loader', action='store_true', default=False,
                        help='使用 mini loader 模式（默认：关闭）')

    # 解析命令行参数
    args = parser.parse_args()

    # 设置环境变量
    os.environ['PIPELINE_MODE'] = 'async' if args.async_load else 'sync'
    os.environ['ENABLE_PRELOAD'] = str(args.pre_load)
    os.environ['USE_MINILOADER'] = str(args.mini_loader)
    os.environ['RECORD_LAYER_STATS'] = str(args.record_stats)

    # 设置日志（移动到配置获取之前）
    mode = 'async' if args.async_load else 'sync'
    log_dir = setup_logging(mode, args.async_load, args.pre_load)

    # 获取运行配置
    config = get_run_config(args)

    # 显示运行配置
    logger.info("="*50)
    logger.info("运行配置:")
    logger.info(f"- 运行模式: {config['mode']}")
    logger.info(f"- 运行设备: {config['device']}")
    if config['features']:
        logger.info("- 启用特性:")
        for feature in config['features']:
            logger.info(f"  • {feature}")
    else:
        logger.info("- 启用特性: 无")
    logger.info("="*50)

    # 加载所有模型配置
    all_model_configs = utils.load_model_configs()
    if not all_model_configs:
        logger.error("未找到任何模型配置")
        exit(1)

    azure_trace_path = os.path.expanduser(
        "/data/data1/jairwu/data/Azure/AzureFunctionsInvocationTraceForTwoWeeksJan2021/AzureFunctionsInvocationTraceForTwoWeeksJan2021Day13.csv")
    trace_df = pd.read_csv(azure_trace_path)
    trace_df['invo_ts'] = pd.to_datetime(trace_df['invo_ts'], errors='coerce')
    trace_df = trace_df.dropna(subset=['invo_ts'])

    # 固定映射关系配置
    FIXED_MAPPING = {
        "resnet": {
            "resnet50": "155e47f8e7f751d0c845049456d01832013c61336a8cd85901330ac821a71534",
            "resnet101": "41630cdded05ac1d73e45a72ff07c22e90fe6b1d537c5825377a983998c05ad0",
            "resnet152": "58b5ab07aba3f2312b7c99f7d4561e7195fa81744cad27b6e989fbdbb5c6eac7",
        },
        "vit": {
            "vit_b_16": "155e47f8e7f751d0c845049456d01832013c61336a8cd85901330ac821a71534",
            "vit_b_32": "41630cdded05ac1d73e45a72ff07c22e90fe6b1d537c5825377a983998c05ad0",
            "vit_l_16": "58b5ab07aba3f2312b7c99f7d4561e7195fa81744cad27b6e989fbdbb5c6eac7",
        },
        "vgg": {
            "vgg11": "155e47f8e7f751d0c845049456d01832013c61336a8cd85901330ac821a71534",
            "vgg16": "41630cdded05ac1d73e45a72ff07c22e90fe6b1d537c5825377a983998c05ad0",
            "vgg19": "58b5ab07aba3f2312b7c99f7d4561e7195fa81744cad27b6e989fbdbb5c6eac7",
        }

    }

    def validate_mapping(family_name, model_name, trace_df):
        """验证映射关系有效性"""
        family_data = FIXED_MAPPING.get(family_name)
        if not family_data:
            raise ValueError(f"未找到模型家族 {family_name} 的固定映射配置")

        func = family_data.get(model_name)

        if not func:
            raise ValueError(f"模型 {model_name} 在家族 {family_name} 中未定义映射关系")

        # 验证func存在性
        func_df = trace_df[trace_df['func'] == func]
        if func_df.empty:
            raise ValueError(f"函数 {func} 不存在或没有调用记录")

        return func

    # 修改全局映射构建逻辑
    global_mapping = defaultdict(list)
    for family_name, model_configs in all_model_configs.items():
        try:
            # 验证并获取固定映射
            for config in model_configs:
                model_name = config['model']
                func = validate_mapping(family_name, model_name, trace_df)

                func_trace = trace_df[trace_df['func'] == func]

                # 时间戳处理（添加调用次数限制）
                min_ts = func_trace['invo_ts'].min().timestamp()
                normalized_ts = [ts.timestamp() - min_ts
                                 for ts in func_trace['invo_ts'][:MAX_INVOCATIONS]]  # 添加切片操作

                global_mapping[family_name].append({
                    'model_config': config,
                    'func': func,
                    'time_offsets': normalized_ts
                })

        except ValueError as e:
            logger.error(f"固定映射验证失败 - {family_name}/{model_name}: {str(e)}")
            continue

    for family_name, mapping_data in global_mapping.items():
        for item in mapping_data:
            actual_count = len(item['time_offsets'])
            logger.info(
                f"  ▸ {item['model_config']['model']} => {item['func']}, "
                f"实际调用数: {actual_count} (限制前{MAX_INVOCATIONS}个)")  # 添加实际调用数显示

        logger.info("#" * 60)

    # 生成全局测试计划
    global test_schedule  # 使成为全局变量以便调试
    test_schedule = []

    # 构建带家族信息的测试计划
    for family_name, mapping_data in global_mapping.items():
        base_time = time.time()
        for item in mapping_data:
            for offset in item['time_offsets']:
                test_schedule.append({
                    'execute_time': base_time + offset,
                    'family': family_name,
                    'func_name': item['func'],
                    'model_name': item['model_config']['model'],
                    'weights_key': item['model_config'].get('weights')
                })

    # 按执行时间排序并分组
    test_schedule.sort(key=lambda x: x['execute_time'])

    # 按家族分组执行测试
    for family_name in all_model_configs.keys():
        # 筛选当前家族的测试任务
        family_schedule = [
            t for t in test_schedule if t['family'] == family_name]
        if not family_schedule:
            logger.warning(f"家族 {family_name} 没有可用的测试任务，跳过")
            continue

        start_time = family_schedule[0]['execute_time']
        end_time = family_schedule[-1]['execute_time']
        total_duration = end_time - start_time
        mins, secs = divmod(total_duration, 60)

        # 增强日志输出
        duration_info = f"{secs:.1f}秒" if mins == 0 else f"{int(mins)}分{secs:.1f}秒"
        logger.info(
            f"开始执行 {family_name} 家族测试 | "
            f"任务数: {len(family_schedule)} | "
            f"预计持续时间: {duration_info} "
            f"(从 {time.ctime(start_time)} 到 {time.ctime(end_time)})"
        )
        # for task in family_schedule:
        #     print(task)

        # 调用测试方法
        success = PipelineModel.test_model_family(
            test_schedule=family_schedule,
            family_name=family_name
        )

        if success:
            logger.info(
                f"✅ 家族 {family_name} 全部 {len(family_schedule)} 个任务执行成功")
        else:
            logger.error(f"❌ 家族 {family_name} 有任务执行失败")


if __name__ == "__main__":
    main()