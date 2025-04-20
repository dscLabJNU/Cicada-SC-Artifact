import time
import torch
from logger_config import get_logger
import utils
from mini_model import MiniModel
import os
from enum import Enum, auto

logger = get_logger(__name__)


def get_model_structure(model_name: str, mini_load: bool = False):
    """使用本地方法获取模型结构

    Args:
        model_name: 模型名称
        mini_load: 是否使用 MiniModel 获取模型结构

    Returns:
        torch.nn.Module: 模型实例
    """
    try:
        if mini_load:
            mini_model = MiniModel()
            model_type = mini_model._get_model_type(model_name)
            mini_model.add_model_types([model_type])
            return mini_model.get_model(model_name)
        else:
            from model_loader import load_model_from_source
            return load_model_from_source(model_name, weights_key="")

    except Exception as e:
        logger.error(f"本地获取模型结构失败: {e}")
        return None


class LoadStrategy(Enum):
    MINI = auto()
    TRADITIONAL = auto()


def clear_log_files():
    """清空所有日志 CSV 文件"""
    log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)
    for file in ['model_eval_resnet_async_nopreload_mini_memory.csv', 'model_eval_resnet_async_nopreload_traditional_memory.csv']:
        file_path = os.path.join(log_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)
    logger.info("已清空 ResNet 相关的日志 CSV 文件")


def test_model_family(family_name: str, strategy: LoadStrategy):
    """
    测试特定模型家族的加载性能和内存使用情况
    """
    log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)
    strategy_name = strategy.name.lower()
    log_path = os.path.join(
        log_dir, f'model_eval_{family_name}_async_nopreload_{strategy_name}_memory.csv')

    # 检查文件是否存在，如果不存在则写入表头
    file_exists = os.path.exists(log_path)
    log_file = open(log_path, 'a', newline='')  # 使用追加模式

    if not file_exists:
        # 写入表头
        print('model_name,structure_load_time,'
              'before_param_count,before_param_mb,before_buffer_mb,before_serialized_mb,before_instance_mb,before_total_mb,'
              'after_param_count,after_param_mb,after_buffer_mb,after_serialized_mb,after_instance_mb,after_total_mb',
              file=log_file, flush=True)

    logger.info(f"开始测试 {family_name} 家族模型 (使用 {strategy_name} 方式):")
    logger.info("=" * 50)

    model_configs = utils.load_model_configs(family_name)

    for config in model_configs:
        model_nickname = config['model_nickname']
        weights_key = config['weights_key']
        model_name = config['model_name']

        logger.info(f"测试 {model_nickname}:")
        logger.info("-" * 40)

        # 根据策略选择加载方式
        start = time.time()
        if strategy == LoadStrategy.MINI:
            model = get_model_structure(model_name, mini_load=True)
        else:
            model = get_model_structure(model_nickname, mini_load=False)
        load_time = time.time() - start

        # 记录加载权重前的内存使用
        before_memory = utils.MemoryUtils.get_model_memory_usage(model)

        # 加载权重
        weights = utils.get_weights_instance(model_nickname, weights_key)
        if weights:
            with torch.no_grad():
                for param in model.parameters():
                    param.data = param.data.float()

            state_dict = weights.get_state_dict()
            if family_name == 'densenet':
                utils.load_densenet_state_dict(model, state_dict)
            else:
                model.load_state_dict(state_dict)
            logger.info(f"已加载 {weights_key} 权重")
        else:
            logger.warning(f"无法加载 {weights_key} 权重")

        # 记录加载权重后的内存使用
        after_memory = utils.MemoryUtils.get_model_memory_usage(model)

        # 记录性能指标，包含所有内存位置的数据
        print(f'{model_name},{load_time},'
              f'{before_memory["parameters"]["count"]},'
              f'{before_memory["parameters"]["size_mb"]},'
              f'{before_memory["buffers"]["size_mb"]},'
              f'{before_memory["serialized"]["size_mb"]},'
              f'{before_memory["instance"]["size_mb"]},'
              f'{before_memory["total"]["size_mb"]},'
              f'{after_memory["parameters"]["count"]},'
              f'{after_memory["parameters"]["size_mb"]},'
              f'{after_memory["buffers"]["size_mb"]},'
              f'{after_memory["serialized"]["size_mb"]},'
              f'{after_memory["instance"]["size_mb"]},'
              f'{after_memory["total"]["size_mb"]}',
              file=log_file, flush=True)

        # 日志输出
        logger.info(f"结构加载时间: {load_time:.4f}秒")
        logger.info("加载权重前:")
        logger.info(utils.MemoryUtils.format_size(
            before_memory, detailed=True))
        logger.info("加载权重后:")
        logger.info(utils.MemoryUtils.format_size(after_memory, detailed=True))
        logger.info("=" * 50)


def init_mini_model(model_types: list):
    start = time.time()
    mini_model = MiniModel(model_types=model_types)
    mini_time = time.time() - start

    # 计算 MiniModel 实例的大小
    instance_size = utils.MemoryUtils.get_model_memory_usage(mini_model)
    logger.info(f"初始化 MiniModel 实例时间: {mini_time:.2f} s")
    logger.info(f"MiniModel 实例大小:")
    logger.info(utils.MemoryUtils.format_size(instance_size, detailed=True))


if __name__ == "__main__":
    # 清空日志文件
    clear_log_files()

    model_confis = utils.load_model_configs()
    for i in range(1):
        for family_name, family_configs in model_confis.items():
            test_model_family(family_name=family_name,
                              strategy=LoadStrategy.MINI)
            test_model_family(family_name=family_name,
                              strategy=LoadStrategy.TRADITIONAL)
