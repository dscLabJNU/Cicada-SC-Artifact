import torchvision.models as models
import time
import pandas as pd
import yaml
from pathlib import Path
import sys
import os
sys.path.append('..')
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from utils import MemoryUtils, get_weights_instance  # noqa


def analyze_layer_weight_loading(model, model_name):
    """分析模型各层权重加载和应用的时间"""
    import torch
    from pathlib import Path
    import time

    data = []
    cache_dir = Path(f'persistent_cache/{model_name}')
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 生成有效的层名称映射（跳过空名称）
    layer_names = {}
    for idx, (name, layer) in enumerate(model.named_modules()):
        # 只保留第一级子模块且名称非空
        if name.count('.') < 1 and name:  # 新增名称非空检查
            safe_name = name
            layer_names[layer] = safe_name

    # 首先保存所有层的权重（父模块自动包含子模块参数）
    for layer, safe_name in layer_names.items():
        if len(list(layer.parameters())) > 0:
            # 获取包含所有子模块参数的完整state_dict
            layer_state = layer.state_dict()
            save_path = cache_dir / f"{safe_name}.pt"
            torch.save(layer_state, save_path)

    # 测量加载和应用时间
    for layer, safe_name in layer_names.items():
        if len(list(layer.parameters())) > 0:
            layer_path = cache_dir / f"{safe_name}.pt"

            # 测量加载时间
            load_start = time.time()
            state_dict = torch.load(layer_path)
            load_time = (time.time() - load_start) * 1000

            # 测量应用时间（自动应用到所有子模块）
            apply_start = time.time()
            layer.load_state_dict(state_dict)
            apply_time = (time.time() - apply_start) * 1000

            data.append({
                'model_name': model_name,
                'layer_name': safe_name,
                'load_time_ms': load_time,
                'apply_time_ms': apply_time,
                'total_time_ms': load_time + apply_time
            })

    return data


def load_model_with_timing(model_config):
    """加载单个模型并记录时间和内存使用情况"""
    data = []
    weight_loading_data = []
    total_start = time.time()
    model_name = model_config['name']
    weights_key = model_config['weights']

    try:
        weights_instance = get_weights_instance(model_name, weights_key)
        # 获取模型
        model = getattr(models, model_name)(weights=weights_instance)

        # 分析层权重加载时间
        weight_loading_data = analyze_layer_weight_loading(model, model_name)

        # 如果有数据，保存到CSV
        if weight_loading_data:
            df = pd.DataFrame(weight_loading_data)
            df.to_csv('layer_weight_loading_applying.csv',
                      mode='a',
                      header=not Path(
                          'layer_weight_loading_applying.csv').exists(),
                      index=False)

        # 记录总时间
        total_time = (time.time() - total_start) * 1000

        print(f"分析模型: {model_name}的内存使用情况")
        # 获取模型内存使用情况
        memory_info = MemoryUtils.get_model_memory_usage(model)
        # 格式化内存信息
        memory_str = MemoryUtils.format_size(memory_info, detailed=True)

        print(f"\n{model_name}:")
        print(f"初始化时间: {total_time:.2f}ms")
        print(f"内存使用: {memory_str}")

        data.append({
            'model_name': model_name,
            # 'init_time_ms': total_time,
            'params_count': memory_info['parameters']['count'],
            'params_size_mb': memory_info['parameters']['size_mb'],
            'buffers_size_mb': memory_info['buffers']['size_mb'],
            'instance_size_mb': memory_info['instance']['size_mb'],
            # 'serialized_size_mb': memory_info['serialized']['size_mb'],
            'total_size_mb': memory_info['total']['size_mb']
        })

    except Exception as e:
        print(f"加载模型 {model_name} 时出错: {str(e)}")
        return []

    return data


def load_models_from_config():
    # 读取配置文件
    config_path = Path('../models_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    all_data = []

    # 遍历所有家族的模型
    for family_name, family_models in config['families'].items():
        print(f"\n开始加载 {family_name} 家族的模型:")

        if not family_models:  # 如果是空列表或者被注释掉了
            continue

        for model_config in family_models:
            model_data = load_model_with_timing(model_config)
            for data in model_data:
                data['family'] = family_name
            all_data.extend(model_data)

    # 创建DataFrame并保存
    if all_data:
        df = pd.DataFrame(all_data)
        csv_path = 'layer_memory_usage.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n所有数据已保存到 {csv_path}")

        # 按家族打印统计信息
        print("\n各模型家族统计信息:")
        for family in df['family'].unique():
            family_data = df[df['family'] == family]
            print(f"\n{family} 家族:")
            for _, row in family_data.iterrows():
                print(f"  {row['model_name']}:")
                # print(f"    初始化时间: {row['init_time_ms']:.2f}ms")
                print(f"    参数数量: {row['params_count']:,}")
                print(f"    参数大小: {row['params_size_mb']:.2f}MB")
                print(f"    总内存: {row['total_size_mb']:.2f}MB")
    else:
        print("没有成功加载任何模型")


if __name__ == "__main__":
    os.system(
        'rm -rf layer_memory_usage.csv layer_init_timing.csv layer_weight_loading.csv layer_weight_loading_applying.csv')
    load_models_from_config()
    os.system('python analyze_model_loading.py')
