import requests
import torch
import io
from torchvision import models
import time
import os
from transformers import AutoModel, AutoConfig
import numpy as np
from model_config import ModelConfig
import utils
import pandas as pd
import argparse
from collections import defaultdict
import datetime

# 设置最大调用次数限制
MAX_INVOCATIONS = 100

def get_model_constructor(model_identifier, model_fn_map):
    """
    获取模型构造函数的统一接口。
    逻辑：
    如果 model_identifier 在 model_fn_map 中，直接返回。
    """
    if model_identifier in model_fn_map:
        return model_fn_map[model_identifier]

def load_model_from_local(name, model_id, weights_key="", model_fn_map=None, weights_fn_map=None):
    """
    本地加载模型及其 state_dict, 包括从 Hugging Face 下载的权重。
    支持 TorchVision 官方模型和 Hugging Face 模型。

    参数:
        name (str): 模型的唯一名称（如 'resnet50_default' 或 'bert-base-uncased_default'）。
        model_id (str): 实际的模型标识符（如 'resnet50' 或 'bert-base-uncased'）。
        weights_key (str): 权重键名或外部权重标识符。
                            对于内置权重，如 'ResNet101_Weights.DEFAULT'。
                            对于 Hugging Face 权重，如 'huggingface:bert-base-uncased'。
        model_fn_map (dict): 模型标识符到模型构造函数的映射。
        weights_fn_map (dict): 权重键名到权重枚举的映射。

    返回:
        tuple: (local_model, local_state_dict) 或 (None, None) 如果不支持的模型。
    """
    if model_fn_map is None or weights_fn_map is None:
        print("model_fn_map 和 weights_fn_map 必须被提供。")
        return None, None

    if name not in model_fn_map:
        model_fn = get_model_constructor(model_id, model_fn_map)
        if model_fn is None:
            print(
                f"Model '{model_id}' not supported by model_fn_map nor Hugging Face.")
            return None, None
        model_fn_map[name] = model_fn

    model_fn = model_fn_map[name]

    if weights_key:
        weights = weights_fn_map.get(weights_key, None)
        print(
            f"Verifying weights: {weights}, type: {type(weights)} (weights_key: {weights_key})")
        if weights is None:
            print(
                f"Unsupported weights_key '{weights_key}' for model '{name}'.")
            return None, None
        try:
            model = model_fn(weights=weights)
            print(f"Loaded model '{name}' with weights '{weights_key}'.")
        except Exception as e:
            print(
                f"Failed to load weights '{weights_key}' for model '{name}': {e}")
            return None, None
    else:
        # 标准模型，无需额外权重加载
        try:
            model = model_fn()
            print(f"Loaded model '{name}' without additional weights.")
        except Exception as e:
            print(f"Failed to load model '{name}': {e}")
            return None, None

    # 获取模型的 state_dict
    local_state_dict = model.state_dict()
    return model, local_state_dict


def verify_model(server_state_dict, model_name, model_id, weights_key, model_fn_map, weights_fn_map):
    """
    验证服务器返回的 state_dict 是否与本地加载的模型一致。

    Args:
        server_state_dict (dict): 从服务器获取的模型 state_dict。
        model_name (str): 模型的唯一名称（如 'resnet50_default' 或 'bert-base-uncased_default'）。
        model_id (str): 实际的模型标识符（如 'resnet50' 或 'bert-base-uncased'）。
        weights_key (str): 权重键名或外部权重标识符。
        model_fn_map (dict): 模型标识符到模型构造函数的映射。
        weights_fn_map (dict): 权重键名到权重枚举的映射。
    """
    local_model, local_state_dict = load_model_from_local(
        model_name, model_id, weights_key=weights_key, model_fn_map=model_fn_map, weights_fn_map=weights_fn_map
    )
    if local_model is None:
        print(f"Skipping verification for model '{model_name}'.")
        return

    # 加载服务器的 state_dict 到本地模型
    try:
        local_model.load_state_dict(server_state_dict)
        print(f"Loaded server state_dict into local model '{model_name}'.")
    except Exception as e:
        print(
            f"Error loading server state_dict into local model '{model_name}': {e}")
        return

    # 验证一致性
    utils.verify_model_state_dict(
        server_state_dict, local_state_dict, model_name)


def load_model_from_server(model_name, model_id, weights_key="", server_load_url="http://localhost:8888/model_load", verify=False, model_fn_map=None, weights_fn_map=None):
    """
    从服务器加载模型的 state_dict 并验证其与本地模型的一致性。

    Args:
        model_name (str): 模型的唯一名称（如 'resnet50_default' 或 'bert-base-uncased_default'）。
        model_id (str): 实际的模型标识符（如 'resnet50' 或 'bert-base-uncased'）。
        weights_key (str, optional): 权重键名或外部权重标识符。
        server_load_url (str, optional): 服务器加载模型的URL。
        verify (bool, optional): 是否进行验证。
        model_fn_map (dict): 模型标识符到模型构造函数的映射。
        weights_fn_map (dict): 权重键名到权重枚举的映射。

    Returns:
        dict or None: 服务器的 state_dict，如果加载成功。
    """
    data = {"model_name": model_name, "weights_key": weights_key}

    try:
        response = requests.get(server_load_url, json=data)
        response.raise_for_status()
        print(f"Successfully requested model '{model_name}' from server.")
    except requests.exceptions.RequestException as e:
        print(f"Request failed for model '{model_name}': {e}")
        return None

    # 获取服务器返回的序列化 state_dict
    state_dict_bytes = response.content

    # 反序列化服务器的 state_dict
    try:
        server_buffer = io.BytesIO(state_dict_bytes)
        server_state_dict = torch.load(server_buffer)
        print(
            f"Successfully deserialized state_dict for model '{model_name}'.")
    except Exception as e:
        print(
            f"Failed to deserialize state_dict for model '{model_name}': {e}")
        return None

    # 验证模型
    if verify:
        verify_model(server_state_dict, model_name, model_id,
                     weights_key, model_fn_map, weights_fn_map)

    return server_state_dict

# 新增: 读取Azure trace数据和模型映射功能
def load_azure_trace(trace_path):
    """
    读取Azure函数调用追踪数据
    
    Args:
        trace_path (str): Azure trace数据文件路径
        
    Returns:
        pd.DataFrame: 处理后的追踪数据
    """
    print(f"从 {trace_path} 加载Azure函数调用追踪数据...")
    
    try:
        trace_df = pd.read_csv(trace_path)
        trace_df['invo_ts'] = pd.to_datetime(trace_df['invo_ts'], errors='coerce')
        trace_df = trace_df.dropna(subset=['invo_ts'])
        print(f"成功加载Azure追踪数据，共 {len(trace_df)} 条记录")
        return trace_df
    except Exception as e:
        print(f"加载Azure追踪数据失败: {e}")
        return None

def get_fixed_mapping():
    """
    获取固定的模型到函数的映射关系
    
    Returns:
        dict: 模型家族和模型到函数的映射
    """
    return {
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
    """
    验证映射关系的有效性
    
    Args:
        family_name (str): 模型家族名称
        model_name (str): 模型名称
        trace_df (pd.DataFrame): Azure追踪数据
        
    Returns:
        str: 有效的函数ID，如果无效则抛出异常
    """
    fixed_mapping = get_fixed_mapping()
    family_data = fixed_mapping.get(family_name)
    
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

def build_test_schedule(all_model_configs, trace_df, use_tetris=False):
    """
    构建基于Azure trace的测试计划
    
    Args:
        all_model_configs (dict): 所有模型配置
        trace_df (pd.DataFrame): Azure追踪数据
        use_tetris (bool): 是否使用tetris策略
        
    Returns:
        list: 测试计划列表
        dict: 全局映射关系
    """
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
                                for ts in func_trace['invo_ts'][:MAX_INVOCATIONS]]
                
                global_mapping[family_name].append({
                    'model_config': config,
                    'func': func,
                    'time_offsets': normalized_ts
                })
                
                print(f"模型 {model_name} 映射到函数 {func}, 实际调用数: {len(normalized_ts)} (限制前{MAX_INVOCATIONS}个)")
                
        except ValueError as e:
            print(f"固定映射验证失败 - {family_name}/{model_name}: {str(e)}")
            continue
    
    # 生成测试计划
    test_schedule = []
    base_time = time.time()
    
    for family_name, mapping_data in global_mapping.items():
        for item in mapping_data:
            for offset in item['time_offsets']:
                test_schedule.append({
                    'execute_time': base_time + offset,
                    'family': family_name,
                    'func_name': item['func'],
                    'model_name': item['model_config']['model'],
                    'weights_key': item['model_config'].get('weights')
                })
    
    # 按执行时间排序
    test_schedule.sort(key=lambda x: x['execute_time'])
    
    if use_tetris:
        # 这里可以添加tetris策略的处理逻辑
        print("启用tetris策略进行负载调度优化...")
        # 预留tetris实现部分
    
    return test_schedule, global_mapping

def execute_test_schedule(test_schedule, model_fn_map, weights_fn_map, log_file):
    """
    执行测试计划
    
    Args:
        test_schedule (list): 测试计划列表
        model_fn_map (dict): 模型构造函数映射
        weights_fn_map (dict): 权重映射
        log_file (TextIO): 日志文件
    """
    current_idx = 0
    success_count = 0
    
    while current_idx < len(test_schedule):
        current_task = test_schedule[current_idx]
        now = time.time()
        
        # 等待直到执行时间到达
        if now < current_task['execute_time']:
            wait_time = current_task['execute_time'] - now
            if wait_time > 0.1:  # 仅当等待时间超过0.1秒时打印信息
                print(f"等待 {wait_time:.2f} 秒后执行下一任务...")
            time.sleep(wait_time)
            continue
        
        print("="*40)
        print(f"执行任务 {current_idx+1}/{len(test_schedule)}")
        print(f"模型: {current_task['model_name']}, 权重: {current_task['weights_key']}")
        print(f"函数: {current_task['func_name']}, 家族: {current_task['family']}")
        print(f"执行时间: {datetime.datetime.fromtimestamp(current_task['execute_time']).strftime('%Y-%m-%d %H:%M:%S.%f')}")
        
        # 加载模型
        start = time.time()
        model_name = current_task['model_name']
        weights_key = current_task['weights_key']
        model_id = model_name  # 在此简化实现中，我们直接使用model_name作为model_id
        
        # server_state_dict = load_model_from_local(
        #     model_name, model_id, weights_key, model_fn_map=model_fn_map, weights_fn_map=weights_fn_map
        # )
        server_state_dict = load_model_from_server(
            model_name, model_id, weights_key, verify=False,
            model_fn_map=model_fn_map, weights_fn_map=weights_fn_map
        )
        
        if server_state_dict is not None:
            structure_load_time_ms = time.time() - start
            print(f"模型 '{model_name}' 加载成功，耗时 {structure_load_time_ms:.2f}s")
            print(f"{model_name},{weights_key},{structure_load_time_ms}", file=log_file, flush=True)
            success_count += 1
        else:
            print(f"模型 '{model_name}' 加载失败")
        
        print("="*40)
        print("\n")
        
        current_idx += 1
    
    return success_count

def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='模型加载测试工具')
    
    parser.add_argument('--azure-trace', type=str, 
                        default='/data/data1/jairwu/data/Azure/AzureFunctionsInvocationTraceForTwoWeeksJan2021/AzureFunctionsInvocationTraceForTwoWeeksJan2021Day13.csv',
                        help='Azure函数调用追踪数据文件路径')
    
    parser.add_argument('--use-trace', action='store_true',
                        help='使用Azure追踪数据进行测试')
    
    parser.add_argument('--tetris', action='store_true',
                        help='启用tetris策略进行负载调度优化')
    
    parser.add_argument('--output', type=str,
                        default=None,  # 修改为None，在main函数中生成默认值
                        help='输出文件路径')
    
    parser.add_argument('--family', type=str, choices=['resnet', 'vit', 'vgg', 'all'],
                        default='all', help='指定要测试的模型家族')
    
    return parser.parse_args()

def main():
    """
    主函数，加载并验证多个模型。
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 定义服务器URL
    server_load_url = "http://localhost:8888/model_load"

    # 加载 YAML 配置
    yaml_path = 'models_config.yaml'  # 确保路径正确
    model_config = ModelConfig(yaml_path)
    model_fn_map = model_config.get_model_fn_map()
    weights_fn_map = model_config.get_weights_fn_map()
    model_to_default_weight = model_config.get_default_weight()
    
    # 获取所有模型配置
    all_model_configs = utils.load_model_configs()
    if not all_model_configs:
        print("未找到任何模型配置，使用默认配置...")
        # 如果没有配置文件，使用默认的模型列表
        
    # 生成输出文件名
    if args.output is None:
        # 根据参数生成格式化的输出文件名
        family_str = args.family if args.family != 'all' else 'allmodels'
        tetris_str = 'tetris' if args.tetris else 'notetris'
        output_file = f'model_eval_{family_str}_{tetris_str}_nopreload_normal.csv'
        # 确保输出到logs目录
        os.makedirs('logs', exist_ok=True)
        output_file = os.path.join('logs', output_file)
    else:
        output_file = args.output
    
    # 确保输出文件的目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 直接使用完整路径打开文件
    log_file = open(output_file, "w")
    print(f"model_name,weights_key,structure_load_time", file=log_file, flush=True)
    print(f"输出结果将写入文件: {output_file}")
    
    if args.use_trace:
        # 使用Azure追踪数据进行测试
        print("使用Azure追踪数据进行测试...")
        
        # 加载Azure trace数据
        trace_df = load_azure_trace(args.azure_trace)
        if trace_df is None:
            print("Azure追踪数据加载失败，退出测试")
            log_file.close()
            return
            
        # 构建测试计划
        test_schedule, global_mapping = build_test_schedule(
            all_model_configs, trace_df, use_tetris=args.tetris
        )
        
        if args.family != 'all':
            # 过滤特定家族的测试任务
            test_schedule = [t for t in test_schedule if t['family'] == args.family]
            print(f"过滤后的测试计划包含 {len(test_schedule)} 个任务 (仅{args.family}家族)")
        
        if not test_schedule:
            print("测试计划为空，退出测试")
            log_file.close()
            return
            
        # 输出测试计划摘要
        start_time = test_schedule[0]['execute_time']
        end_time = test_schedule[-1]['execute_time']
        total_duration = end_time - start_time
        mins, secs = divmod(total_duration, 60)
        
        duration_info = f"{secs:.1f}秒" if mins == 0 else f"{int(mins)}分{secs:.1f}秒"
        print(f"开始执行测试计划 | 任务数: {len(test_schedule)} | 预计持续时间: {duration_info}")
        
        # 执行测试计划
        success_count = execute_test_schedule(
            test_schedule, model_fn_map, weights_fn_map, log_file
        )
        
        print(f"测试计划执行完成: {success_count}/{len(test_schedule)} 成功")
        
    else:
        # 使用静态定义的模型列表进行测试
        print("使用静态定义的模型列表进行测试...")
        
        # 定义要加载和验证的模型列表
        models_to_load = [
            # Vision Transformer models
            {"model_name": "vit_b_16", "weights_key": "ViT_B_16_Weights.DEFAULT"},
            {"model_name": "vit_b_32", "weights_key": "ViT_B_32_Weights.DEFAULT"},
            {"model_name": "vit_l_16", "weights_key": "ViT_L_16_Weights.DEFAULT"},

            # ResNet模型示例：
            {"model_name": "resnet50", "weights_key": "ResNet50_Weights.DEFAULT"},
            {"model_name": "resnet101", "weights_key": "ResNet101_Weights.DEFAULT"},
            {"model_name": "resnet152", "weights_key": "ResNet152_Weights.DEFAULT"},

            # VGG 模型示例：
            {"model_name": "vgg11", "weights_key": "VGG11_Weights.DEFAULT"},
            {"model_name": "vgg16", "weights_key": "VGG16_Weights.DEFAULT"},
            {"model_name": "vgg19", "weights_key": "VGG19_Weights.DEFAULT"},
        ]
        
        # 根据指定的家族过滤模型
        if args.family != 'all':
            models_to_load = [m for m in models_to_load if m['model_name'].startswith(args.family)]
            print(f"过滤后的模型列表包含 {len(models_to_load)} 个模型 (仅{args.family}家族)")
            
        # 遍历模型列表进行测试
        for model_info in models_to_load:
            name = model_info["model_name"]
            weights_key = model_info.get("weights_key", "")
            # 通过 name 获取 model_id
            model_id = name.replace('_default', '')

            print("="*40)
            # 如果 weights_key 为空，则从 YAML 配置中获取默认权重
            if not weights_key:
                weights_key = model_to_default_weight.get(name, "")
                if weights_key:
                    print(
                        f"No weights_key specified for model '{name}'. Using default weight '{weights_key}' from YAML.")
                else:
                    print(
                        f"No weights_key specified for model '{name}', and no default weight found in YAML.")
            print(
                f"\nLoading and verifying model '{name}' with weights_key '{weights_key}'...")
            start = time.time()
            server_state_dict = load_model_from_server(
                name, model_id, weights_key, server_load_url, verify=False,
                model_fn_map=model_fn_map, weights_fn_map=weights_fn_map
            )
            # server_state_dict = load_model_from_local(
            #     name, model_id, weights_key, model_fn_map=model_fn_map, weights_fn_map=weights_fn_map
            # )
            if server_state_dict is not None:
                structure_load_time_ms = (time.time() - start) * 1000
                print(
                    f"Model '{name}' loaded and verified (optional) in {structure_load_time_ms:.2f}ms.")
                print(f"{name},{weights_key},{structure_load_time_ms}", file=log_file, flush=True)
            print("="*40)
            print("\n\n")
    
    # 关闭日志文件
    log_file.close()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"测试结束，总耗时: {end_time - start_time:.2f} 秒")
