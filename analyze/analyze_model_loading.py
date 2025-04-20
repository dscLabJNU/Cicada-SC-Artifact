import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from pathlib import Path
import utils
from matplotlib.ticker import MaxNLocator

DEFAULT_FIGSIZE = (2, 1)
DEFAULT_FONTSIZE = 10
MARGIN_SETTINGS = {
    'left': 0.15,
    'right': 0.95,
    'top': 0.85,
    'bottom': 0.4
}
Y_LIMIT = 1000
Y_HIGHT_THRESHOLD = Y_LIMIT + 20
MODEL_DISPLAY_NAMES = {
    'resnet50': 'ResNet50',
    'resnet101': 'ResNet101',
    'resnet152': 'ResNet152',
    'vgg11': 'VGG11',
    'vgg16': 'VGG16',
    'vgg19': 'VGG19',
    'vit_b_16': 'ViT-B-16',
    'vit_l_16': 'ViT-L-16',
    'vit_b_32': 'ViT-B-32'
}


def get_model_family(model_name):
    """获取模型所属的家族"""
    if 'vit' in model_name.lower():
        return 'ViT'
    if 'resnet' in model_name.lower():
        return "ResNet"
    if 'vgg' in model_name.lower():
        return "VGG"
    # 通用处理：提取第一个数字或下划线之前的部分
    family = re.split(r'[\d_]', model_name)[0]
    return family


def generate_x_positions(models: list, gap: float = 0.8) -> list:
    """生成带家族间隔的x轴坐标（与所有分析函数兼容）"""
    x_positions = []
    current_pos = 0
    for i, model in enumerate(models):
        x_positions.append(current_pos)
        if i < len(models)-1:
            current_family = get_model_family(model)
            next_family = get_model_family(models[i+1])
            if current_family != next_family:
                current_pos += gap
        current_pos += 1
    return x_positions


def init_plot_style(figsize=DEFAULT_FIGSIZE, fontsize=DEFAULT_FONTSIZE):
    """初始化统一绘图样式"""
    plt.figure(figsize=figsize)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = fontsize
    plt.rcParams['axes.titlesize'] = fontsize + 2
    plt.rcParams['xtick.labelsize'] = fontsize - 4
    plt.rcParams['ytick.labelsize'] = fontsize - 2
    plt.rcParams['legend.fontsize'] = fontsize - 4
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))


def apply_ylim_with_annotation(ylim=(0, Y_LIMIT)):
    """应用统一y轴限制并在超出时标注数值（修正版）"""
    plt.ylim(ylim)  # 先设置y轴限制
    # ax = plt.gca()

    # # 直接检查每个柱子的高度
    # for bar in ax.patches:
    #     actual_height = bar.get_height()
    #     if actual_height > ylim[1]:
    #         # 计算显示位置（限制在图表范围内）
    #         display_height = min(actual_height, ylim[1] * 1.05)  # 留5%的顶部空间
    #         ax.text(bar.get_x() + bar.get_width()/2.,
    #                 display_height,
    #                 f'{actual_height:.0f}',
    #                 ha='center', va='bottom',
    #                 fontsize=DEFAULT_FONTSIZE-8,
    #                 color='darkred',
    #                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))


def save_and_close_plot(save_path: str, ylim=(0, Y_LIMIT)):
    """统一保存并关闭图表（新增ylim参数）"""
    apply_ylim_with_annotation(ylim)
    plt.subplots_adjust(**MARGIN_SETTINGS)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()


def analyze_init_timing(csv_path: str):
    """分析所有模型的层初始化时间分布"""
    # 创建 images 目录
    Path('./images').mkdir(exist_ok=True)

    # 读取CSV文件
    df = pd.read_csv(csv_path)
    df = utils.filter_df(df)

    # 自定义排序函数
    def sort_model_name(name):
        # 获取模型家族
        family = get_model_family(name)
        # 提取数字部分
        numbers = re.findall(r'\d+', name)
        # 如果有数字，使用第一个数字作为排序依据
        number = int(numbers[0]) if numbers else 0
        return (family, number, name)
    df['model_name'] = df['model_name'].map(MODEL_DISPLAY_NAMES)
    # 获取所有唯一的模型名称并排序
    models = sorted(df['model_name'].unique(), key=sort_model_name)

    # 准备绘图数据
    model_components = {}
    for model in models:
        model_data = df[df['model_name'] == model]
        components_dict = {}

        # 获取该模型的所有组件（排除包含 init 的组件）
        components = model_data[~model_data['component'].str.contains(
            'init', case=False, na=False)]

        # 按时间戳和出现顺序排序组件
        components = components.sort_values(['timestamp', 'component'])

        # 针对ViT模型特殊处理：从encoder_layer中扣除MLPBlock的初始化时间
        if get_model_family(model) == 'vit':
            # 获取所有MLPBlock初始化时间
            mlp_blocks = model_data[model_data['component'].str.contains(
                'init_weights_MLPBlock')]
            mlp_total = mlp_blocks['time_ms'].sum()
            print(f"MLPBlock初始化时间: {mlp_total:.2f}ms")

            # 调整encoder_layer的时间
            encoder_layer = components[components['component']
                                       == 'encoder_layer']
            if not encoder_layer.empty:
                components.loc[encoder_layer.index, 'time_ms'] -= mlp_total

        # 存储组件时间
        for _, row in components.iterrows():
            comp_name = row['component']
            comp_time = row['time_ms']
            # 如果是ViT的encoder_layer，扣除MLPBlock的时间
            if get_model_family(model) == 'vit' and comp_name == 'encoder_layer':
                comp_time -= mlp_total
            components_dict[comp_name] = comp_time

        model_components[model] = components_dict

    # 获取所有唯一的组件名称（保持时序顺序）
    all_components = []
    for model_data in model_components.values():
        for comp in model_data.keys():
            if comp not in all_components:
                all_components.append(comp)

    # 计算每个家族的模型数量和位置
    family_counts = {}
    family_positions = {}
    current_pos = 0

    for model in models:
        family = get_model_family(model)
        if family not in family_counts:
            family_counts[family] = 1
            family_positions[family] = current_pos
        else:
            family_counts[family] += 1
        current_pos += 1

    x_positions = generate_x_positions(models)
    init_plot_style()

    # 准备绘图数据
    x = np.array(x_positions)
    bottom = np.zeros(len(models))

    # 使用离散的颜色方案
    colors = sns.color_palette("Set2", len(all_components))

    # 计算每个模型的总层数
    model_layer_counts = {}
    for model in models:
        model_data = df[df['model_name'] == model]
        layer_count = len(model_data[
            ~model_data['component'].str.contains('init', case=False, na=False)
        ]['component'].unique())
        model_layer_counts[model] = layer_count

    for i, model in enumerate(models):
        total_height = sum(model_components[model].values())
        if model in ['ViT-H-14', 'ViT-L-16']:
            if total_height > Y_LIMIT:
                plt.text(
                    x[i], Y_HIGHT_THRESHOLD, f'{total_height:.0f}', color='darkred',
                    ha='center', va='bottom', fontsize=DEFAULT_FONTSIZE-5)

    # 绘制每个组件的条形
    for i, component in enumerate(all_components):
        component_times = []
        for model in models:
            if component in model_components[model]:
                component_times.append(model_components[model][component])
            else:
                component_times.append(0)

        plt.bar(x, component_times, bottom=bottom, edgecolor='black',
                color=colors[i], label=component)
        bottom += np.array(component_times)

    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    plt.xticks(x_positions, models, rotation=30, ha='right')
    plt.yticks()
    plt.ylabel('Time (ms)', weight='bold')
    plt.xlabel('Model Name', weight='bold')
    # plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    save_and_close_plot('./images/layerConstructTimeOverhead.pdf')

    # 打印统计信息
    print("\n=== 模型初始化时间分析 ===")
    for model in models:
        try:
            # 计算不包含 init 的组件的总时间
            components = model_components[model]
            total_time = sum(time for comp, time in components.items()
                             if 'init' not in comp.lower())

            print(f"\n{model}:")
            print(f"总初始化时间: {total_time:.2f}ms")

            # 打印主要组件时间（按时间降序）
            components = model_components[model]
            if components:
                print("主要耗时组件:")
                for comp_name, comp_time in sorted(
                    components.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]:  # 只显示前5个最耗时的组件
                    print(f"  {comp_name:20s}: {comp_time:8.2f}ms")
            else:
                print("没有找到组件时间数据")

        except Exception as e:
            print(f"\n处理模型 {model} 统计信息时出错: {str(e)}")
            continue


def analyze_init_overhead(csv_path: str):
    """分析所有模型的初始化开销时间"""
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    df = utils.filter_df(df)

    def sort_model_name(name):
        family = get_model_family(name)
        numbers = re.findall(r'\d+', name)
        number = int(numbers[0]) if numbers else 0
        return (family, number, name)
    df['model_name'] = df['model_name'].map(MODEL_DISPLAY_NAMES)
    models = sorted(df['model_name'].unique(), key=sort_model_name)

    print("\n=== 模型初始化开销分析 ===")

    # 为每个模型分析初始化开销
    init_stats = {}
    for model in models:
        model_data = df[df['model_name'] == model]

        # 获取包含 init 的组件
        init_components = model_data[
            model_data['component'].str.contains('init', case=False, na=False)
        ]

        # 按时间戳排序
        init_components = init_components.sort_values(
            ['timestamp', 'component'])

        # 收集统计信息
        total_init_time = init_components['time_ms'].sum()
        init_count = len(init_components)

        # 按时间降序排列的初始化组件
        top_init_components = init_components.nlargest(5, 'time_ms')

        init_stats[model] = {
            'total_time': total_init_time,
            'init_count': init_count,
            'top_components': top_init_components
        }

        # 打印该模型的统计信息
        print(f"\n{model}:")
        print(f"初始化组件数量: {init_count}")
        print(f"总初始化时间: {total_init_time:.2f}ms")

        if not top_init_components.empty:
            print("耗时最多的初始化组件:")
            for _, row in top_init_components.iterrows():
                print(f"  {row['component']:50s}: {row['time_ms']:8.2f}ms")

    # 准备数据 - 使用排序后的模型列表
    init_times = [init_stats[m]['total_time'] for m in models]
    init_counts = [init_stats[m]['init_count'] for m in models]

    x_positions = generate_x_positions(models)
    init_plot_style()

    for i, model in enumerate(models):
        total_time = init_stats[model]['total_time']

        # 特别标注ViT大模型
        if model in ['vit_l_16', 'vit_h_14']:
            if total_time > Y_LIMIT:
                plt.text(
                    x_positions[i], Y_HIGHT_THRESHOLD,
                    f'{total_time:.0f}',
                    color='darkred',
                    ha='center',
                    va='bottom',
                    fontsize=DEFAULT_FONTSIZE-8)
    # 绘制柱状图
    plt.bar(x_positions, init_times, alpha=0.8, edgecolor='black')
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    plt.xticks(x_positions, models, rotation=30, ha='right')
    plt.yticks()
    plt.xlabel('Model Name', weight='bold')
    plt.ylabel('Time (ms)',  weight='bold')
    # plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # 保存图表
    save_and_close_plot('./images/layerWeightInitOverhead.pdf')

    return init_stats


def analyze_weight_loading(csv_path: str):
    """分析权重加载时间"""
    # 创建 images 目录
    Path('./images').mkdir(exist_ok=True)

    # 读取CSV文件
    df = pd.read_csv(csv_path)
    df = utils.filter_df(df)

    # 使用与之前分析相同的排序函数
    def sort_model_name(name):
        family = get_model_family(name)
        numbers = re.findall(r'\d+', name)
        number = int(numbers[0]) if numbers else 0
        return (family, number, name)
    df['model_name'] = df['model_name'].map(MODEL_DISPLAY_NAMES)
    # 按模型分组并计算总时间
    grouped = df.groupby(['model_name', 'stage'])[
        'time_ms'].sum().unstack().fillna(0)
    grouped = grouped.reindex(index=sorted(grouped.index, key=sort_model_name))

    # 新增统计信息输出
    print("\n=== 权重加载时间占比分析 ===")
    for model in grouped.index:
        get_time = grouped.loc[model, 'weights_get']
        load_time = grouped.loc[model, 'weights_load']
        total = get_time + load_time
        if total > 0:
            ratio = get_time / total * 100
            print(f"{model}: {ratio:.2f}% (Get) | {100 - ratio:.2f}% (Apply)")

    # 新增全局总占比计算
    total_get = grouped['weights_get'].sum()
    total_load = grouped['weights_load'].sum()
    grand_total = total_get + total_load
    if grand_total > 0:
        print(f"\n全局总占比:")
        print(f"Get阶段总时间: {total_get:.2f}ms ({total_get/grand_total:.1%})")
        print(f"Apply阶段总时间: {total_load:.2f}ms ({total_load/grand_total:.1%})")
        print(f"合计总时间: {grand_total:.2f}ms")

    models = grouped.index.tolist()
    weights_get = grouped['weights_get'].values
    weights_load = grouped['weights_load'].values

    x_positions = generate_x_positions(models)
    init_plot_style()

    p1 = plt.bar(x_positions, weights_get,
                 label='Weight File Processing', edgecolor='black')
    p2 = plt.bar(x_positions, weights_load,
                 bottom=weights_get, label='Weight Applying', edgecolor='black')
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    plt.legend(frameon=False)

    plt.xticks(x_positions, models, rotation=30, ha='right')
    plt.yticks()
    plt.ylabel('Time (ms)',  weight='bold')
    plt.xlabel('Model Name', weight='bold')
    # plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    save_and_close_plot('./images/weightLoadingAndApplyingOverhead.pdf')


def analyze_memory_usage(csv_path: str):
    """分析所有模型的内存使用分布"""
    # 复用现有基础设施
    Path('./images').mkdir(exist_ok=True)
    df = pd.read_csv(csv_path)
    df = utils.filter_df(df)

    # 使用公共排序函数
    def sort_model_name(name):
        family = get_model_family(name)
        numbers = re.findall(r'\d+', name)
        number = int(numbers[0]) if numbers else 0
        return (family, number, name)
    # 统一数据准备流程
    if 'model' in df.columns:
        df = df.rename(columns={'model': 'model_name'})
    df['model_name'] = df['model_name'].map(MODEL_DISPLAY_NAMES)
    models = sorted(df['model_name'].unique(), key=sort_model_name)

    # 使用公共坐标生成函数
    x_positions = generate_x_positions(models)

    # 统一绘图初始化
    init_plot_style()

    # 准备数据并绘制柱状图
    total_memory = [df[df['model_name'] == m]
                    ['total_size_mb'].iloc[0] for m in models]
    plt.bar(x_positions, total_memory, color='steelblue',
            edgecolor='black', width=0.8)

    for i, model in enumerate(models):
        mem_usage = total_memory[i]
        if model in ['ViT-H-14', 'ViT-L-16']:
            if mem_usage > Y_LIMIT:
                plt.text(
                    x_positions[i], Y_HIGHT_THRESHOLD, f'{mem_usage:.0f}', color='darkred',
                    ha='center', va='bottom', fontsize=DEFAULT_FONTSIZE-5)

    # 统一坐标轴设置
    plt.xticks(x_positions, models, rotation=30, ha='right')
    plt.yticks()
    plt.ylabel('Memory (MB)', weight='bold')
    plt.xlabel('Model Name', weight='bold')
    # plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # 准备数据时记录最大值
    max_memory = max(total_memory)
    custom_ylim = (0, max(1200, max_memory*1.1))  # 动态调整

    save_and_close_plot('./images/modelMemoryOverhead.pdf')

    # 统一统计信息输出格式
    print("\n=== 模型内存使用分析 ===")
    for model in models:
        try:
            data = df[df['model_name'] == model].iloc[0]
            print(f"\n{model}:")
            print(f"参数数量: {data['params_count']:,}")
            print(f"参数大小: {data['params_size_mb']:.2f}MB")
            print(f"缓冲区大小: {data['buffers_size_mb']:.2f}MB")
            print(f"实例大小: {data['instance_size_mb']:.2f}MB")
            print(f"总内存使用: {data['total_size_mb']:.2f}MB")
        except Exception as e:
            print(f"\n处理模型 {model} 统计信息时出错: {str(e)}")


def analyze_layer_loading_overhead(csv_path='layer_weight_loading_applying.csv'):
    """分析各层权重加载和应用时间开销"""
    df = pd.read_csv(csv_path)
    df = utils.filter_df(df)

    # 统一模型排序逻辑
    def sort_model_name(name):
        family = get_model_family(name)
        numbers = re.findall(r'\d+', name)
        number = int(numbers[0]) if numbers else 0
        return (family, number, name)

    df['model_name'] = df['model_name'].map(MODEL_DISPLAY_NAMES)
    models = sorted(df['model_name'].unique(), key=sort_model_name)
    x_positions = generate_x_positions(models)
    init_plot_style()

    # 准备颜色映射
    unique_layers = df['layer_name'].unique()
    color_palette = sns.color_palette("Set2", len(unique_layers))
    layer_colors = dict(zip(unique_layers, color_palette))

    plt.figure(figsize=DEFAULT_FIGSIZE)
    legend_elements = []

    # 主绘图循环
    for model_idx, model_name in enumerate(models):
        model_df = df[df['model_name'] == model_name]
        layers = model_df.sort_values('layer_name')['layer_name'].unique()
        current_x = x_positions[model_idx]

        # 绘制加载时间（底部实心）
        load_bottom = 0
        for layer in layers:
            layer_data = model_df[model_df['layer_name'] == layer]
            total_load = layer_data['load_time_ms'].sum()

            plt.bar(current_x, total_load,
                    bottom=load_bottom,
                    color=layer_colors[layer],
                    edgecolor='black',
                    # linewidth=0.5,
                    width=0.8)

            load_bottom += total_load

        # 绘制应用时间（顶部阴影）
        apply_bottom = load_bottom
        for layer in layers:
            layer_data = model_df[model_df['layer_name'] == layer]
            total_apply = layer_data['apply_time_ms'].sum()

            plt.bar(current_x, total_apply,
                    bottom=apply_bottom,
                    color=layer_colors[layer],
                    edgecolor='black',
                    # linewidth=0.5,
                    hatch='////',
                    alpha=0.8,
                    width=0.8)

            apply_bottom += total_apply

            # 记录图例（每个层只添加一次）
            if model_idx == 0:
                legend_elements.append(plt.Rectangle(
                    (0, 0), 1, 1,
                    facecolor=layer_colors[layer],
                    edgecolor='black',
                    label="Weights Loading"))
                legend_elements.append(plt.Rectangle(
                    (0, 0), 1, 1,
                    facecolor=layer_colors[layer],
                    hatch='////',
                    alpha=0.8,
                    edgecolor='black',
                    label="Weights Applying"))
    for model_idx, model_name in enumerate(models):
        # 计算总时间（加载+应用）
        model_df = df[df['model_name'] == model_name]
        total_load = model_df['load_time_ms'].sum()
        total_apply = model_df['apply_time_ms'].sum()
        total_time = total_load + total_apply

        # 为ViT大模型添加标注
        if model_name in ['ViT-L-16', 'ViT-H-14']:
            print(f"{model_name}: {total_time:.2f}ms")
            if total_time > Y_LIMIT:
                plt.text(
                    x_positions[model_idx],
                    Y_HIGHT_THRESHOLD,
                    f'{total_time:.0f}',
                    color='darkred',
                    ha='center',
                    va='bottom',
                    fontsize=DEFAULT_FONTSIZE-8)

    # 统一坐标轴设置
    plt.xticks(x_positions, models, rotation=30, ha='right')
    plt.yticks()
    plt.ylabel('Time (ms)', weight='bold')
    plt.xlabel('Model Name', weight='bold')
    # plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    shown_legend = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor='gray',
            edgecolor='black',
            linewidth=0.5,
            hatch='////',
            alpha=0.8,
            label='Weights Application'),
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor='gray',
            edgecolor='black',
            linewidth=0.5,
            label='Weights File Retrieval'),
    ]

    plt.legend(handles=shown_legend,
               loc='upper left',
               frameon=False,
               edgecolor='gray')

    save_and_close_plot('./images/layerLoadingAndApplyingOverheadStacked.pdf')


if __name__ == "__main__":
    csv_path = "layer_init_timing.csv"
    analyze_init_timing(csv_path)
    analyze_init_overhead(csv_path)
    analyze_weight_loading("layer_weight_loading.csv")
    analyze_memory_usage("layer_memory_usage.csv")
    analyze_layer_loading_overhead()
