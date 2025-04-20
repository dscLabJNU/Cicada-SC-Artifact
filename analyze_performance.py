import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import glob
import os
from pathlib import Path
import logging
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import datetime

DEFAULT_FIGSIZE = (7, 2)
DEFAULT_FONTSIZE = 14

# 添加策略显示名称映射
STRATEGY_DISPLAY_NAMES = {
    'ASYNC_PRELOAD_MINI': 'Cicada',
    'ASYNC_PRELOAD': 'Preload',
    'ASYNC_MINI': 'Mini',
    'ASYNC': 'PISeL',
    # 'SYNC': 'Sync'
    'TETRIS': 'Tetris'
}
STRATEGY_ORDER = ['Tetris', 'PISeL', 'Preload', 'Mini', 'Cicada']

# 添加模型显示名称映射
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

logger = logging.getLogger(__name__)


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


def grouping_model_request_id(df):
    df = df.copy()
    # 1) 为不同模型定义其层序
    layer_sequences = {
        'resnet': ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'],
        'vgg':    ['features', 'avgpool', 'classifier'],
        'vit':    ['conv_proj', 'class_token', 'encoder', 'heads']
    }

    # 用来根据 model_name 判断其属于哪一类
    def get_model_type(model_name: str):
        if model_name.startswith('resnet'):
            return 'resnet'
        elif model_name.startswith('vgg'):
            return 'vgg'
        elif model_name.startswith('vit'):
            return 'vit'
        else:
            return 'other'  # 如果不属于上述三类，可以自行处理

    # 2) 根据 (model_name, 时间戳) 排序，让同一模型的行顺序一致
    df = df.sort_values(
        by=['model_name', 'submit_time_ts']).reset_index(drop=True)

    request_id = 0
    last_model_type = None   # 用来跟踪当前遍历到的模型类型
    model_request_ids = []

    for idx, row in df.iterrows():
        # 当前行的模型类型
        model_type = get_model_type(row['model_name'])
        # 拿到对应模型的层序，如果是 'other' 或不在字典中的模型，则给个空列表
        seq = layer_sequences.get(model_type, [])

        # 3) 如果模型类型切换了，重新开始计数
        if model_type != last_model_type:
            request_id = 0
            last_model_type = model_type

        # 4) 如果该行的 layer_name 是该模型类型序列的「第一个层」，视为一个新的请求
        if seq and (row['layer_name'] == seq[0]):
            request_id += 1

        model_request_ids.append(request_id)

    df['model_request_id'] = model_request_ids
    return df


def merge_intervals(intervals):
    """合并重叠的时间区间"""
    if not intervals:
        return []
    # 按开始时间排序
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for current in intervals[1:]:
        previous = merged[-1]
        # 如果当前区间与前一个区间重叠
        if current[0] <= previous[1]:
            # 更新前一个区间的结束时间
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:
            merged.append(current)
    return merged


def calculate_utilization_group(df_group):
    """计算单个请求（由 model_request_id 分组）的流水线利用率"""
    intervals = []
    # 遍历该 Model 请求内的所有 layer，收集所有阶段的时间区间
    for _, row in df_group.iterrows():
        # 结构构建阶段
        if row['structure_start_ts'] < row['structure_end_ts']:
            intervals.append(
                (row['structure_start_ts'], row['structure_end_ts']))
        else:
            raise ValueError(
                f"structure_start_ts is greater than structure_end_ts: {row['structure_start_ts']} - {row['structure_end_ts']}")

        # 权重预加载阶段
        if pd.notnull(row['weight_preload_start_ts']) and pd.notnull(row['weight_preload_end_ts']):
            if row['weight_preload_start_ts'] < row['weight_preload_end_ts']:
                intervals.append(
                    (row['weight_preload_start_ts'], row['weight_preload_end_ts']))
            else:
                raise ValueError(
                    f"weight_preload_start_ts is greater than weight_preload_end_ts: {row['weight_preload_start_ts']} - {row['weight_preload_end_ts']}")

        # 权重加载阶段
        if row['weight_start_ts'] < row['weight_end_ts']:
            intervals.append((row['weight_start_ts'], row['weight_end_ts']))
        else:
            raise ValueError(
                f"weight_start_ts is greater than weight_end_ts: {row['weight_start_ts']} - {row['weight_end_ts']}")

        # 计算阶段
        if row['compute_start_ts'] < row['compute_end_ts']:
            intervals.append((row['compute_start_ts'], row['compute_end_ts']))
        else:
            raise ValueError(
                f"compute_start_ts is greater than compute_end_ts: {row['compute_start_ts']} - {row['compute_end_ts']}")

    # 合并重叠的区间
    merged_intervals = merge_intervals(intervals)
    # 计算总忙碌时间（秒）再转换为毫秒
    busy_time_ms = sum(end - start for start, end in merged_intervals) * 1000

    model_pipeline_time_ms_large = (
        df_group['end_time_ts'].max() - df_group['submit_time_ts'].min()) * 1000
    model_pipeline_time_ms_small = (
        df_group['compute_end_ts'].max() - df_group['structure_start_ts'].min()) * 1000
    # print(
    #     f'model_pipeline_time_ms_large-model_pipeline_time_ms_small: {(model_pipeline_time_ms_large - model_pipeline_time_ms_small)}')
    model_pipeline_time_ms = model_pipeline_time_ms_small
    # print(f"waste_time_ms: {model_pipeline_time_ms - busy_time_ms}ms")

    utilization = (busy_time_ms / model_pipeline_time_ms) * 100

    return pd.Series({'busy_time_ms': busy_time_ms, 'model_pipeline_time_ms': model_pipeline_time_ms, 'utilization': utilization})


class PerformanceAnalyzer:
    def __init__(self, log_dir='./logs'):
        self.combined_df = None
        self.log_dir = log_dir
        self.plot_dir = f'{self.log_dir}/images'
        Path(self.plot_dir).mkdir(exist_ok=True)

    def init_plot_style(self, fontsize=DEFAULT_FONTSIZE, figsize=DEFAULT_FIGSIZE):
        """初始化统一绘图样式"""
        plt.figure(figsize=figsize)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        # 确保字体设置
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'

        plt.rcParams['axes.labelsize'] = fontsize
        plt.rcParams['axes.titlesize'] = fontsize + 2
        plt.rcParams['xtick.labelsize'] = fontsize - 2
        plt.rcParams['ytick.labelsize'] = fontsize - 2
        plt.rcParams['legend.fontsize'] = fontsize - 5
        plt.rcParams['font.size'] = fontsize - 5
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    def load_data(self, suffix="model_eval", end_suffix=''):
        """加载并合并所有策略的日志文件"""
        csv_files = glob.glob(f'{self.log_dir}/{suffix}_*_*{end_suffix}.csv')
        if len(csv_files) == 0:
            raise ValueError(
                f"No CSV files found: {suffix}_*{end_suffix}.csv, make sure you have run the `run_memory_comparison.sh`")
        dfs = []
        for file in csv_files:
            filename = os.path.basename(file)
            # 解析文件名，例如: model_eval_resnet_async_nopreload_mini.csv
            parts = filename.replace(f'{suffix}_', '').replace(
                '.csv', '').split('_')

            if len(parts) >= 3:  # 确保文件名格式正确
                family = parts[0]  # resnet
                mode = parts[1]   # async/sync

                # 构建策略标识
                strategy_parts = []

                # 添加基础模式
                if mode.lower() == 'sync':
                    strategy_parts.append('SYNC')
                elif mode.lower() == 'async':
                    strategy_parts.append('ASYNC')
                elif mode.lower() == 'tetris':
                    strategy_parts.append('TETRIS')

                # 检查预加载状态
                if 'preload' in filename and 'nopreload' not in filename:
                    strategy_parts.append('PRELOAD')

                # 检查mini状态
                if 'mini' in filename:
                    strategy_parts.append('MINI')

                strategy = '_'.join(strategy_parts)

                # 添加调试信息
                # print(f"Loading file: {file}")
                print(f"Strategy identified: {strategy}({mode})")

                df = pd.read_csv(file)
                df['model_family'] = family
                df['strategy'] = strategy
                if mode.lower() == 'tetris':
                    df['structure_load_time'] = df['structure_load_time'] * 1000
                dfs.append(df)

        if not dfs:
            logger.error("没有找到任何CSV文件")
            return None

        self.combined_df = pd.concat(
            dfs, ignore_index=True).dropna(axis=1, how='all')

        # 打印所有可用的策略名称，用于调试
        print("\nAvailable strategies:", self.combined_df['strategy'].unique())
        print("Available models:", self.combined_df['model_name'].unique())

    def _calculate_timing_metrics(self, df):
        """统一的时间指标计算函数"""

        df = grouping_model_request_id(df.copy())
        return df

    def analyze_unified_load_time(self):
        """在同一张图中分析所有模型的加载时间"""
        self.init_plot_style()
        plt.figure(figsize=DEFAULT_FIGSIZE)
        plt.rcParams['font.family'] = 'Times New Roman'

        # 准备数据
        plot_data = self.combined_df.copy()
        plot_data['strategy'] = plot_data['strategy'].map(
            STRATEGY_DISPLAY_NAMES)
        plot_data['model_name'] = plot_data['model_name'].map(
            lambda x: MODEL_DISPLAY_NAMES.get(x, x)
        )

        # 计算每个模型在不同策略下的平均加载时间
        avg_times = plot_data.groupby(['model_name', 'strategy'])[
            'structure_load_time'].mean().reset_index()

        # 计算优化比例
        improvements = []
        for model in avg_times['model_name'].unique():
            model_data = avg_times[avg_times['model_name'] == model]
            pisel_time = model_data[model_data['strategy']
                                    == 'PISeL']['structure_load_time'].values[0]

            for strategy in ['Preload', 'Mini', 'Cicada', 'Tetris']:
                if strategy in model_data['strategy'].values:
                    strategy_time = model_data[model_data['strategy']
                                               == strategy]['structure_load_time'].values[0]
                    improvement = (pisel_time - strategy_time) / \
                        pisel_time * 100
                    improvements.append({
                        'Model': model,
                        'Strategy': strategy,
                        'Improvement': improvement
                    })

        # 打印优化比例
        print("\n优化比例分析（相对于PISeL）：")
        improvements_df = pd.DataFrame(improvements)
        for strategy in ['Preload', 'Mini', 'Cicada', 'Tetris']:
            strategy_improvements = improvements_df[improvements_df['Strategy'] == strategy]
            avg_improvement = strategy_improvements['Improvement'].mean()
            print(f"\n{strategy}策略：")
            print(f"平均优化比例: {avg_improvement:.2f}%")
            print("各模型优化比例:")
            for _, row in strategy_improvements.iterrows():
                print(f"{row['Model']}: {row['Improvement']:.2f}%")

        # 计算Cicada相对于Preload和Mini的优化比例
        cicada_comparisons = []
        mini_comparisons = []
        for model in avg_times['model_name'].unique():
            model_data = avg_times[avg_times['model_name'] == model]

            # 获取各策略的时间
            cicada_time = model_data[
                model_data['strategy']
                == 'Cicada']['structure_load_time'].mean()
            preload_time = model_data[
                model_data['strategy']
                == 'Preload']['structure_load_time'].mean()
            mini_time = model_data[
                model_data['strategy']
                == 'Mini']['structure_load_time'].mean()
            tetris_time = model_data[
                model_data['strategy']
                == 'Tetris']['structure_load_time'].mean()

            # 计算相对优化比例
            vs_preload = (preload_time - cicada_time) / preload_time * 100
            vs_mini = (mini_time - cicada_time) / mini_time * 100
            vs_tetris = (tetris_time - cicada_time) / tetris_time * 100
            cicada_comparisons.append({
                'Model': model,
                'vs_Preload': vs_preload,  # Mini的作用
                'vs_Mini': vs_mini,         # Preload的作用
                'vs_Tetris': vs_tetris
            })

            mini_vs_tetris = (tetris_time - mini_time) / tetris_time * 100

            mini_comparisons.append({
                'Model': model,
                'vs_Tetris': mini_vs_tetris
            })

        # 创建输出文件
        output_path = os.path.join(
            self.log_dir, 'analyze_unified_load_time.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            # 第一部分：相对于PISeL的优化比例
            f.write("一、相对于PISeL的优化比例分析：\n")
            f.write("-" * 50 + "\n")

            for strategy in ['Preload', 'Mini', 'Cicada', 'Tetris']:
                strategy_improvements = improvements_df[improvements_df['Strategy'] == strategy]
                avg_improvement = strategy_improvements['Improvement'].mean()

                f.write(f"\n{strategy}策略：\n")
                f.write(f"平均优化比例: {avg_improvement:.2f}%\n")
                f.write("各模型优化比例:\n")

                sorted_improvements = strategy_improvements.sort_values(
                    'Model')
                for _, row in sorted_improvements.iterrows():
                    f.write(f"{row['Model']}: {row['Improvement']:.2f}%\n")

            # 第二部分：Cicada的组件分析
            f.write("\n\n二、Cicada策略组件分析：\n")
            f.write("-" * 50 + "\n")

            # 计算平均值
            cicada_comp_df = pd.DataFrame(cicada_comparisons)
            avg_vs_preload = cicada_comp_df['vs_Preload'].mean()
            avg_vs_mini = cicada_comp_df['vs_Mini'].mean()
            avg_vs_tetris = cicada_comp_df['vs_Tetris'].mean()

            f.write(f"Cicada相对于Tetris的优化：\n")
            f.write(f"平均优化比例: {avg_vs_tetris:.2f}%\n")
            f.write("各模型优化比例:\n")
            for _, row in cicada_comp_df.sort_values('Model').iterrows():
                f.write(f"{row['Model']}: {row['vs_Tetris']:.2f}%\n")

            f.write("\nCicada相对于Preload的优化（Mini的作用）：\n")
            f.write(f"平均优化比例: {avg_vs_preload:.2f}%\n")
            f.write("各模型优化比例:\n")
            for _, row in cicada_comp_df.sort_values('Model').iterrows():
                f.write(f"{row['Model']}: {row['vs_Preload']:.2f}%\n")

            f.write("\nCicada相对于Mini的优化（Preload的作用）：\n")
            f.write(f"平均优化比例: {avg_vs_mini:.2f}%\n")
            f.write("各模型优化比例:\n")
            for _, row in cicada_comp_df.sort_values('Model').iterrows():
                f.write(f"{row['Model']}: {row['vs_Mini']:.2f}%\n")

            f.write("\n\n三、Mini策略组件分析：\n")
            f.write("-" * 50 + "\n")

            # 计算平均值
            mini_comp_df = pd.DataFrame(mini_comparisons)
            avg_mini_vs_tetris = mini_comp_df['vs_Tetris'].mean()

            f.write(f"Mini相对于Tetris的优化：\n")
            f.write(f"平均优化比例: {avg_mini_vs_tetris:.2f}%\n")
            f.write("各模型优化比例:\n")
            for _, row in mini_comp_df.sort_values('Model').iterrows():
                f.write(f"{row['Model']}: {row['vs_Tetris']:.2f}%\n")
            print(f"优化比例分析已保存到: {output_path}")

        # 原有的绘图代码
        sns.barplot(
            data=plot_data,
            x='model_name',
            y='structure_load_time',
            hue='strategy',
            palette='Set3',
            edgecolor='black',
            linewidth=1,
            hue_order=STRATEGY_ORDER,
            order=sorted(
                plot_data['model_name'].unique(),
                key=lambda x: (
                    get_model_family(x),
                    int(re.search(r'\d+', x).group()
                        ) if re.search(r'\d', x) else 0,
                    x
                )
            )
        )

        # 设置图表样式
        plt.ylabel('Inference Time (ms)', weight='bold')
        plt.xlabel('Model Name', weight='bold')

        # 调整刻度标签
        plt.xticks(rotation=15, weight='bold')
        plt.yticks()

        # 调整图例
        plt.legend(
            loc="upper center",
            ncol=len(plot_data['strategy'].unique()),
            frameon=False,
            fontsize=plt.rcParams['legend.fontsize']+2
        )

        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.ylim(top=plot_data['structure_load_time'].max())
        plt.ylim(top=5000)
        plt.savefig(f'{self.log_dir}/images/inferenceOverhead.pdf',
                    format='pdf', bbox_inches='tight')
        plt.close()

    def analyze_unified_memory(self):
        """在同一张图中分析所有模型的内存使用情况"""
        self.init_plot_style()
        plt.rcParams['font.family'] = 'Times New Roman'

        # 准备数据
        plot_data = self.combined_df.copy()
        plot_data['strategy'] = plot_data['strategy'].map(
            STRATEGY_DISPLAY_NAMES)
        plot_data['model_name'] = plot_data['model_name'].map(
            lambda x: MODEL_DISPLAY_NAMES.get(x, x)
        )

        # 获取唯一模型和策略列表
        models = sorted(plot_data['model_name'].unique(),
                        key=lambda x: (
            get_model_family(x),
            int(re.search(r'\d+', x).group()
                ) if re.search(r'\d', x) else 0,
            x
        ))
        strategies = ['PISeL', 'Mini']

        # 按模型和策略分组取平均值
        memory_grouped_data = plot_data.groupby(['model_name', 'strategy']).agg({
            'before_param_mb': 'mean',
        }).reset_index()

        self.load_data(suffix="pipeline_stats")
        plot_data = self._calculate_timing_metrics(self.combined_df.copy())
        plot_data['strategy'] = plot_data['strategy'].map(
            STRATEGY_DISPLAY_NAMES)
        plot_data['model_name'] = plot_data['model_name'].map(
            lambda x: MODEL_DISPLAY_NAMES.get(x, x)
        )

        def calculate_duration(df_group):
            """
            针对单个请求（由 model_request_id 分组）
            """
            df = df_group.copy()
            df['memory_usage_time_ms'] = df['weight_start_ts'] - \
                df['init_weight_end_ts']
            return pd.Series({
                # 同一个model请求内的不同layer 计算总时间
                'memory_usage_time_ms': df['memory_usage_time_ms'].sum() * 1000,
            })

        result_df = plot_data.groupby(['strategy', 'model_name', 'model_request_id']).apply(
            calculate_duration).reset_index()

        grouped_data = result_df.groupby(['model_name', 'strategy']).agg({
            'memory_usage_time_ms': 'mean',  # 相同model的不同请求，计算平均值
        }).reset_index()

        stats_data = pd.merge(
            grouped_data[['model_name', 'memory_usage_time_ms', 'strategy']],
            memory_grouped_data[['model_name', 'before_param_mb', 'strategy']],
            on=['strategy', 'model_name'],
        )
        stats_data['memory_overhead_GB*ms'] = stats_data['memory_usage_time_ms'] * \
            (stats_data['before_param_mb']/1024)

        stats_data['family'] = stats_data['model_name'].apply(
            lambda x: get_model_family(x))

        grouped_data = stats_data.copy()
        # 计算 Mini 策略相对于 PISeL 策略的提升倍数
        mini_data = grouped_data[grouped_data['strategy'] == 'Mini']
        pisel_data = grouped_data[grouped_data['strategy'] == 'PISeL']

        # 合并数据以便计算提升倍数
        stats_data = pd.merge(
            pisel_data[['model_name', 'before_param_mb',
                        'memory_usage_time_ms']],
            mini_data[['model_name', 'before_param_mb', 'memory_usage_time_ms']],
            on='model_name',
            suffixes=('_PISeL', '_Mini')
        )

        stats_data['memory_usage_time_increased_by_mini(%)'] = 100*(
            stats_data['memory_usage_time_ms_Mini'] - stats_data['memory_usage_time_ms_PISeL']) / stats_data['memory_usage_time_ms_Mini']
        stats_data['memory_usage_reduction_by_mini_MB'] = (stats_data['before_param_mb_PISeL'] -
                                                           stats_data['before_param_mb_Mini'])
        print(f"stats_data: \n{stats_data}")
        # 设置柱状图参数
        x = np.arange(len(models))
        width = 0.8 / len(strategies)

        fig, ax1 = plt.subplots(figsize=DEFAULT_FIGSIZE)

        # 为每个策略绘制柱状图
        for i, strategy in enumerate(strategies):
            strategy_data = grouped_data[grouped_data['strategy'] == strategy]

            # 确保数据顺序与models列表一致
            merged_data = pd.merge(
                pd.DataFrame({'model_name': models}),
                strategy_data,
                on='model_name',
                how='left').fillna(0)

            # 绘制参数内存
            ax1.bar(x + i * width,
                    merged_data['before_param_mb'],
                    width,
                    alpha=0.8,
                    edgecolor='black',
                    label=strategy,
                    linewidth=1)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Memory Time (ms)', weight='bold')
        for strategy in strategies:
            strategy_data = result_df[result_df['strategy'] == strategy]
            avg_memory_usage = strategy_data.groupby(
                'model_name')['memory_usage_time_ms'].mean()
            # print(
            #     f"strategy: {strategy}, avg_memory_usage: \n{avg_memory_usage}")
            avg_memory_usage.index = pd.Categorical(
                avg_memory_usage.index, categories=models, ordered=True)
            sns.lineplot(
                x=avg_memory_usage.index, y=avg_memory_usage.values,
                linewidth=2,
                markersize=10,
                marker='D', label=strategy, ax=ax2)

        vit_l_16_data = grouped_data[(grouped_data['model_name'] == 'ViT-L-16') & (
            grouped_data['strategy'] == 'PISeL')]
        y_limit = 600
        if not vit_l_16_data.empty:
            vit_l_16_value = vit_l_16_data['before_param_mb'].values[0]
            plt.text(
                x=models.index('ViT-L-16') - 1.2 * width,
                y=790,
                s=1162,  # f'{vit_l_16_value:.1f}',
                ha='center',
                color='darkred',
                fontsize=10,
                weight='bold'
            )
        ax1.set_ylim([0, y_limit])
        ax1.set_ylabel('Memory Usage (MB)', weight='bold')
        ax1.set_xlabel('Model Name', weight='bold')
        ax1.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax1.set_xticklabels(models, rotation=15, weight='bold')

        ax1.legend(loc="upper left", ncol=2, frameon=False,
                   fontsize=plt.rcParams['legend.fontsize']+2)
        ax2.legend(loc="upper right", frameon=False, markerscale=0.5,
                   fontsize=plt.rcParams['legend.fontsize']+2)

        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f'{self.log_dir}/images/memoryOverhead.pdf',
                    format='pdf', bbox_inches='tight')
        plt.close()

    def analyze_layer_timing_comparison(self):
        """分析每个模型的层创建时间和权重加载时间对比"""
        plt.figure(figsize=(24, 4))
        plt.rcParams['font.family'] = 'Times New Roman'

        # 读取CSV文件
        df = pd.read_csv(f'{self.log_dir}/layer_stats_resnet_pipeline.csv')

        # 获取所有唯一的模型名称并排序
        models = sorted(df['model_name'].unique())

        # 计算x轴位置（考虑间隔）
        gap = 0.8  # 模型家族之间的间隔
        x_positions = []
        current_pos = 0

        for i, model in enumerate(models):
            family = get_model_family(model)
            x_positions.append(current_pos)
            if i < len(models) - 1:
                next_family = get_model_family(models[i + 1])
                if family != next_family:
                    current_pos += gap
            current_pos += 1

        x = np.array(x_positions)
        width = 0.35  # 柱状图宽度

        # 为每个模型创建堆叠柱状图
        for i, model in enumerate(models):
            model_data = df[df['model_name'] == model]

            # 准备创建时间数据
            create_times = model_data['create_time_ms'].values
            weight_times = model_data['weights_time_ms'].values
            layers = model_data['layer_name'].values

            # 创建两个柱状图的位置
            x_create = x[i] - width/2
            x_weight = x[i] + width/2

            # 绘制堆叠柱状图
            bottom_create = 0
            bottom_weight = 0

            # 使用 Set2 颜色方案
            colors = sns.color_palette("Set2", len(layers))

            for j, (layer, create_time, weight_time) in enumerate(zip(layers, create_times, weight_times)):
                # 创建时间柱状图
                plt.bar(x_create, create_time, width, bottom=bottom_create,
                        color=colors[j], label=layer if i == 0 else "")
                bottom_create += create_time

                # 权重加载时间柱状图
                plt.bar(x_weight, weight_time, width, bottom=bottom_weight,
                        color=colors[j], hatch='/')
                bottom_weight += weight_time

            # 添加总时间标签
            plt.text(x_create, bottom_create, f'{bottom_create:.0f}',
                     ha='center', va='bottom')
            plt.text(x_weight, bottom_weight, f'{bottom_weight:.0f}',
                     ha='center', va='bottom')

        # 设置图表样式
        plt.title('Layer Creation vs Weights Loading Time Comparison', pad=20)
        plt.ylabel('Time (ms)', weight='bold')

        # 设置x轴刻度和标签
        plt.xticks(x, models, rotation=45, ha='right')
        plt.yticks()

        # 创建自定义图例
        # 1. 为创建时间和权重时间创建图例
        timing_legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='gray',
                          label='Layer Creation'),
            plt.Rectangle((0, 0), 1, 1, facecolor='gray',
                          hatch='/', label='Weights Loading')
        ]
        l1 = plt.legend(handles=timing_legend_elements,
                        bbox_to_anchor=(0, 0.85, 1, 0.2),
                        loc="lower center",
                        ncol=2,
                        frameon=False)
        plt.gca().add_artist(l1)

        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f'{self.log_dir}/images/layer_timing_comparison.pdf',
                    format='pdf', bbox_inches='tight')
        plt.close()

    def analyze_pipeline_stages(self):
        """分析流水线中各阶段的执行和等待时间"""
        plt.rcParams['font.family'] = 'Times New Roman'

        # 定义颜色方案和标记
        colors = ['#66c2a5', '#fc8d62', '#8da0cb']
        markers = ['o', 's', '^']

        # 定义要分析的指标
        metrics = [
            ('structure_time_ms', 'structure_wait_ms', 'Structure'),
            ('weight_time_ms', 'weight_wait_ms', 'Weight'),
            ('compute_time_ms', 'compute_wait_ms', 'Compute')
        ]

        # 获取所有唯一的模型名称和策略
        all_models = sorted(self.combined_df['model_name'].unique())
        strategies = ['ASYNC', 'ASYNC_MINI',
                      'ASYNC_PRELOAD', 'ASYNC_PRELOAD_MINI']

        for model_name in all_models:
            # 创建2x2的子图布局
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()

            # 首先计算所有策略中的最大y值
            global_max_time = 0
            for strategy in strategies:
                model_data = self.combined_df[
                    (self.combined_df['model_name'] == model_name) &
                    (self.combined_df['strategy'] == strategy)
                ]

                if len(model_data) > 0:
                    for work_metric, wait_metric, _ in metrics:
                        total_times = model_data[work_metric] + \
                            model_data[wait_metric]
                        global_max_time = max(
                            global_max_time, total_times.max())

            # 设置统一的y轴限制为最大值+100
            y_limit = global_max_time + 100

            for idx, strategy in enumerate(strategies):
                model_data = self.combined_df[
                    (self.combined_df['model_name'] == model_name) &
                    (self.combined_df['strategy'] == strategy)
                ]

                if len(model_data) == 0:
                    continue

                ax = axes[idx]
                layers = model_data['layer_name'].values
                x_positions = np.arange(len(layers))

                # 计算每个策略的总时间
                total_times = {}  # 存储每个指标的总时间
                total_work = 0
                total_wait = 0

                for work_metric, wait_metric, label in metrics:
                    work_total = model_data[work_metric].sum()
                    wait_total = model_data[wait_metric].sum()
                    total_times[label] = (work_total, wait_total)
                    total_work += work_total
                    total_wait += wait_total

                # 为每个指标绘制时间线
                for (work_metric, wait_metric, label), color, marker in zip(metrics, colors, markers):
                    work_times = model_data[work_metric]
                    total_times_line = work_times + model_data[wait_metric]

                    # 绘制总时间线
                    ax.plot(x_positions, total_times_line,
                            color=color,
                            label=f'{label}',
                            marker=marker, linewidth=2, markersize=8)

                    # 绘制工作时间线
                    ax.plot(x_positions, work_times,
                            color=color,
                            linestyle='--', linewidth=2, markersize=8)

                    # 填充等待时间区域
                    ax.fill_between(x_positions, work_times, total_times_line,
                                    color=color, alpha=0.2)

                # 在图中添加横向时间统计
                text_x = 0  # 起始x位置
                text_y = y_limit * 0.65  # y位置在图的下方
                spacing = len(layers) * 0.15  # 文本间距

                # 添加各阶段时间
                for i, (label, (work_total, wait_total)) in enumerate(total_times.items()):
                    stage_text = f'{label}:\nWork:{work_total:.2f}\nWait:{wait_total:.2f}\nTotal:{work_total + wait_total:.2f}'
                    ax.text(text_x + i * spacing, text_y,
                            stage_text,
                            color=colors[i],
                            ha='left', va='bottom',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

                # 添加总计时间（黑色）
                total_text = f'TotalWork:{total_work:.2f}\nTotalWait:{total_wait:.2f}\nTotal:{total_work + total_wait:.2f}'
                ax.text(text_x + 3 * spacing, text_y,
                        total_text,
                        color='black',
                        ha='left', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

                # 设置子图样式
                strategy_display = STRATEGY_DISPLAY_NAMES.get(
                    strategy, strategy)
                ax.set_title(f'{strategy_display}', pad=20)
                ax.set_xlabel('Layer Name')
                ax.set_ylabel('Time (ms)')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_ylim(0, y_limit)

                # 设置x轴刻度和标签
                ax.set_xticks(x_positions)
                ax.set_xticklabels(layers, rotation=45, ha='right')
                ax.tick_params(axis='y')

                # 添加图例
                ax.legend(ncol=3, loc="upper left", frameon=False)

                # 隐藏非最后子图的x轴刻度线
                if idx != len(strategies)-1:
                    ax.tick_params(
                        axis='x',
                        which='both',
                        bottom=False,      # 隐藏底部刻度线
                        labelbottom=False   # 隐藏底部标签
                    )
                else:
                    ax.tick_params(axis='x', which='both',
                                   length=3)  # 显示最后子图的刻度

            # 设置总标题
            fig.suptitle(f'Pipeline Stage Analysis - {model_name}', y=0.95)

            # 保存图表
            output_filename = f'{self.plot_dir}/pipeline_stages_{model_name}.pdf'
            plt.savefig(output_filename, format='pdf', bbox_inches='tight')
            plt.close()

    def analyze_wait_time_comparison(self):
        """可视化各策略的等待时间对比（完整实现版）"""
        self.init_plot_style()
        plot_data = self._calculate_timing_metrics(self.combined_df.copy())
        plot_data['model_name'] = plot_data['model_name'].map(
            lambda x: MODEL_DISPLAY_NAMES.get(x, x)
        )

        WAIT_METRICS = [
            ('structure_wait', 'Structure Wait', '#1f77b4'),
            ('weight_wait', 'Weight Load Wait', '#ff7f0e'),
            ('compute_wait', 'Compute Wait', '#2ca02c')
        ]

        # 数据预处理
        strategies_order = ['ASYNC', 'ASYNC_MINI',
                            'ASYNC_PRELOAD', 'ASYNC_PRELOAD_MINI']
        grouped_data = plot_data.groupby(['model_name', 'strategy'])[
            [m[0] for m in WAIT_METRICS]].sum()
        grouped_data = grouped_data.reindex(pd.MultiIndex.from_product(
            [grouped_data.index.levels[0], strategies_order],
            names=['model_name', 'strategy']
        ), fill_value=0).reset_index()

        for model in grouped_data['model_name'].unique():
            self.init_plot_style()
            model_data = grouped_data[grouped_data['model_name'] == model]

            # 准备堆叠柱状图数据
            bottom = np.zeros(len(strategies_order))
            for idx, (col, label, color) in enumerate(WAIT_METRICS):
                values = model_data.set_index('strategy').reindex(
                    strategies_order)[col].fillna(0)
                sns.barplot(
                    x=strategies_order,
                    y=values,
                    color=color,
                    bottom=bottom,
                    edgecolor='black',
                )
                bottom += values

            # 设置图表样式
            plt.ylabel('Wait Time (ms)', weight='bold')
            plt.xticks(
                ticks=range(len(strategies_order)),
                labels=[STRATEGY_DISPLAY_NAMES.get(
                    s, s) for s in strategies_order],
            )

            # 添加数值标签
            max_time = bottom.max()
            for i, total in enumerate(bottom):
                plt.text(i, total + max_time*0.02, f'{total:.1f}',
                         ha='center', va='bottom')

            # 创建图例
            legend_handles = [
                plt.Rectangle((0, 0), 1, 1, facecolor=color,
                              edgecolor='black', label=label)
                for _, label, color in WAIT_METRICS
            ]
            plt.legend(
                handles=legend_handles,
                frameon=False,
                loc='upper right',
                bbox_to_anchor=(1, 1)
            )

            plt.savefig(f'{self.plot_dir}/wait_time_comparison_{model}.pdf',
                        format='pdf', bbox_inches='tight')
            plt.close()

    def analyze_pipeline_time_breakdown(self):
        """展示所有模型的各阶段时间指标对比，使用3x3网格布局"""
        # 准备数据
        self.init_plot_style()
        plot_data = self._calculate_timing_metrics(self.combined_df.copy())
        plot_data['model_name'] = plot_data['model_name'].map(
            lambda x: MODEL_DISPLAY_NAMES.get(x, x)
        )
        plot_data['strategy'] = plot_data['strategy'].map(
            lambda x: STRATEGY_DISPLAY_NAMES.get(x, x)
        )
        plot_data = plot_data[plot_data['strategy'] != 'SYNC']

        def calculate_duration(df_group):
            """
            针对单个请求（由 model_request_id 分组），计算各阶段耗时的平均值，
            """
            df = df_group.copy()

            df['structure_duration'] = df['structure_end_ts'] - \
                df['structure_start_ts']
            df['init_weight_duration'] = df['init_weight_end_ts'] - \
                df['init_weight_start_ts']

            df['layer_duration'] = df['structure_duration'] + \
                df['init_weight_duration']
            df['layer_wait'] = df['structure_start_ts'] - df['submit_time_ts']

            df['weight_duration'] = df['weight_end_ts'] - df['weight_start_ts']
            # df['weight_wait'] = df['weight_start_ts'] - max(df['init_weight_end_ts'], df['weight_preload_end_ts'])
            df['weight_wait'] = df['weight_start_ts'] - df['init_weight_end_ts']

            df['compute_duration'] = df['compute_end_ts'] - df['compute_start_ts']
            df['compute_wait'] = df['compute_start_ts'] - df['weight_end_ts']
            df['preload_duration'] = df['weight_preload_end_ts'] - \
                df['weight_preload_start_ts']

            return pd.Series({
                'layer_duration_ms': df['layer_duration'].mean() * 1000,
                'layer_wait_ms': df['layer_wait'].mean() * 1000,
                'weight_duration_ms': np.percentile(df['weight_duration'].fillna(0), 80) * 1000,
                'weight_wait_ms': df['weight_wait'].mean() * 1000,
                'compute_duration_ms': df['compute_duration'].mean() * 1000,
                'compute_wait_ms': df['compute_wait'].mean() * 1000,
                'preload_duration_ms': np.percentile(df['preload_duration'].fillna(0), 80) * 1000,
            })

        result_df = plot_data.groupby(['strategy', 'model_name', 'model_request_id']).apply(
            calculate_duration).reset_index()

        grouped_data = result_df.groupby(['model_name', 'strategy']).agg({
            'layer_wait_ms': 'mean',
            'layer_duration_ms': 'mean',
            'weight_duration_ms': 'mean',
            'weight_wait_ms': 'mean',
            'compute_duration_ms': 'mean',
            'compute_wait_ms': 'mean',
            'preload_duration_ms': 'mean'
        }).reset_index()
        grouped_data['total_wait_ms'] = grouped_data['layer_wait_ms'] + \
            grouped_data['weight_wait_ms'] + grouped_data['compute_wait_ms']
        # print(f"breakdown grouped_data: \n{grouped_data}")

        STRATEGY_ORDER = ['PISeL', 'Preload', 'Mini', 'Cicada']

        # 为每个 model_name 添加 model_family 列
        grouped_data['model_family'] = grouped_data['model_name'].apply(
            get_model_family)

        # 定义需要展示的指标及其对应的标签
        time_metrics = [
            ('layer_duration_ms', 'Layer Work'),
            # ('layer_wait_ms', 'Layer Wait'),
            ('weight_duration_ms', 'Weight Work'),
            ('weight_wait_ms', 'Weight Wait'),
            ('compute_duration_ms', 'Compute Work'),
            ('compute_wait_ms', 'Compute Wait'),
            ('preload_duration_ms', 'Preload Time'),
        ]

        subplot_size = 4
        n_rows = 3
        n_cols = 3
        fig, axes = plt.subplots(3, 3, figsize=(
            subplot_size * n_cols, (subplot_size * n_rows)*0.3), sharey=False)
        fig.subplots_adjust(hspace=0.4, wspace=0.15)
        # 定义家族显示的顺序
        families = ['ResNet', 'VGG', 'ViT']

        def natural_keys(text):
            # Split the text into digit and non-digit parts
            return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

        for i, fam in enumerate(families):
            # 筛选出当前家族的所有模型，并按 model_name 排序（确保每家族有3个模型）
            fam_df = grouped_data[grouped_data['model_family'] == fam]
            models = sorted(fam_df['model_name'].unique(), key=natural_keys)
            # models = sorted(fam_df['model_name'].unique())
            for j, model in enumerate(models):
                ax = axes[i, j]
                # 筛选当前模型数据（各策略对应的指标值）
                model_df = fam_df[fam_df['model_name'] == model]
                # 将指标转换成长格式，方便绘制
                melt_df = model_df.melt(id_vars=['strategy'],
                                        value_vars=[tm[0]
                                                    for tm in time_metrics],
                                        var_name='metric',
                                        value_name='value')
                # 使用一个字典将指标列转换为更易读的标签
                metric_label_map = {tm[0]: tm[1] for tm in time_metrics}
                melt_df['metric'] = melt_df['metric'].map(metric_label_map)

                # 用 Seaborn 绘制条形图，不同策略用不同颜色
                sns.barplot(data=melt_df, x='strategy', y='value', edgecolor='black',
                            hue='metric', ax=ax, order=STRATEGY_ORDER)
                ax.set_title(f"{model}", weight='bold',
                             fontsize=plt.rcParams['axes.titlesize']-4,
                             loc='center', y=0.7)
                # if "VGG" in model:
                #     y_limit = 200
                #     ax.set_ylim(0, y_limit)

                #     layer_work_data = melt_df[melt_df['metric']
                #                               == 'Layer Work']
                #     print(f"layer_work_data: \n{layer_work_data}")

                #     # 获取柱状图中每个柱子的位置
                #     bar_positions = ax.get_xticks()

                #     # 为每个策略的Layer Work值添加标签
                #     for idx, (_, row) in enumerate(layer_work_data.iterrows()):
                #         # 找到对应策略在x轴上的位置
                #         strategy_idx = STRATEGY_ORDER.index(row['strategy'])
                #         x_pos = bar_positions[strategy_idx]

                #         # 获取y值并添加标签
                #         y_val = row['value']
                #         if y_val > y_limit:
                #             ax.text(x_pos-0.05, y_limit-50, f"{y_val:.1f}",
                #                     ha='center', va='bottom',
                #                     color='red',
                #                     fontweight='bold',
                #                     fontsize=plt.rcParams['font.size'])

                ax.set_ylabel(
                    'Time (ms)' if j == 0 else '',
                    weight='bold',
                    fontsize=plt.rcParams['ytick.labelsize'])
                ax.set_xlabel("")
                ax.tick_params(
                    axis='x', labelsize=plt.rcParams['xtick.labelsize'])
                ax.tick_params(
                    axis='y', labelsize=plt.rcParams['ytick.labelsize'])
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                ax.get_legend().remove()

                # 保存第一个子图的图例信息，用于创建总图例
                if i == 0:
                    handles, _ = ax.get_legend_handles_labels()

        fig.legend(
            handles=handles,
            labels=[m[1] for m in time_metrics],
            loc='upper center',
            bbox_to_anchor=(0.5, 0.98),
            ncol=len(time_metrics),
            frameon=False,
            fontsize=plt.rcParams['legend.fontsize'] + 2
        )
        plt.savefig(f'{self.plot_dir}/metrics_breakdown_grid.pdf',
                    format='pdf', bbox_inches='tight')
        plt.close()

        # # 按模型和策略分组计算平均值
        # avg_times = plot_data.groupby(['model_name', 'strategy'])[
        #     [m[0] for m in time_metrics]].mean().reset_index()
        avg_times = grouped_data.copy()

        # 计算各指标的优化比例
        improvements = []
        for model in avg_times['model_name'].unique():
            model_data = avg_times[avg_times['model_name'] == model]
            pisel_data = model_data[model_data['strategy'] == 'PISeL']

            if not pisel_data.empty:
                for strategy in ['Preload', 'Mini', 'Cicada']:
                    strategy_data = model_data[model_data['strategy']
                                               == strategy]
                    if not strategy_data.empty:
                        # 计算每个指标的优化比例
                        for metric, metric_name in time_metrics:
                            pisel_value = pisel_data[metric].values[0]
                            strategy_value = strategy_data[metric].values[0]
                            if pisel_value > 0:  # 避免除以零
                                improvement = (
                                    pisel_value - strategy_value) / pisel_value * 100
                                improvement_reverse = None
                                if improvement < 0:
                                    improvement_reverse = (
                                        strategy_value - pisel_value) / strategy_value * 100
                                improvements.append({
                                    'Model': model,
                                    'Strategy': strategy,
                                    'Metric': metric_name,
                                    'Improvement': improvement,
                                    'Improvement_note': improvement_reverse,
                                    "info": f"PISeL_value:{pisel_value:.2f}ms, {strategy}_value:{strategy_value:.2f}ms"
                                })
        # 创建输出文件
        output_path = os.path.join(
            self.log_dir, 'analyze_pipeline_time_breakdown.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("各策略相对于PISeL的优化比例分析：\n")
            f.write("-" * 50 + "\n")

            # 将improvements转换为DataFrame便于分析
            improvements_df = pd.DataFrame(improvements)

            # 对每个策略进行分析
            for strategy in ['Preload', 'Mini', 'Cicada']:
                f.write(f"\n{strategy}策略：\n")
                strategy_data = improvements_df[improvements_df['Strategy'] == strategy]

                # 对每个指标进行分析
                for _, metric_name in time_metrics:
                    # if not 'Wait' in metric_name:
                    #     continue

                    metric_data = strategy_data[strategy_data['Metric']
                                                == metric_name]
                    if not metric_data.empty:
                        avg_improvement = metric_data['Improvement'].mean()
                        avg_improvement_reverse = metric_data['Improvement_note'].mean(
                        )
                        f.write(f"\n{metric_name}:\n")
                        f.write(f"平均优化比例: {avg_improvement:.2f}%\n")
                        f.write(
                            f"平均优化比例(反向): {avg_improvement_reverse:.2f}%\n")
                        f.write("各模型优化比例:\n")

                        # 按模型名称排序输出具体优化比例
                        sorted_improvements = metric_data.sort_values('Model')
                        for _, row in sorted_improvements.iterrows():
                            f.write(
                                f"{row['Model']}: {row['Improvement']:.2f}%\n")
                            f.write(row['info'] + "\n")
                            if row['Improvement_note'] is not None:
                                f.write(
                                    f"{row['Model']}-reverse: {row['Improvement_note']:.2f}%\n")

        print(f"流水线时间分解分析已保存到: {output_path}")

    def analyze_pipeline_overlap_ratio(self):
        """分析流水线各阶段执行时间与总时间的重叠比例"""
        self.init_plot_style()
        plot_data = self._calculate_timing_metrics(self.combined_df)
        plot_data['model_name'] = plot_data['model_name'].map(
            lambda x: MODEL_DISPLAY_NAMES.get(x, x)
        )

        def merge_intervals(intervals):
            """合并重叠的时间区间"""
            if not intervals:
                return []
            # 按开始时间排序
            intervals.sort(key=lambda x: x[0])
            merged = [intervals[0]]

            for current in intervals[1:]:
                previous = merged[-1]
                # 如果当前区间与前一个区间重叠
                if current[0] <= previous[1]:
                    # 更新前一个区间的结束时间
                    merged[-1] = (previous[0], max(previous[1], current[1]))
                else:
                    merged.append(current)
            return merged

        def calculate_overlap_ratio(row):
            """计算单个任务的流水线重叠比例"""
            # 收集工作时间区间
            work_intervals = []

            # 结构构建阶段
            if row['structure_start_ts'] < row['structure_end_ts']:
                work_intervals.append(
                    (row['structure_start_ts'], row['structure_end_ts']))

            # 权重预加载阶段
            if pd.notnull(row['weight_preload_start_ts']) and pd.notnull(row['weight_preload_end_ts']):
                if row['weight_preload_start_ts'] < row['weight_preload_end_ts']:
                    work_intervals.append(
                        (row['weight_preload_start_ts'], row['weight_preload_end_ts']))

            # 权重加载阶段
            if row['weight_start_ts'] < row['weight_end_ts']:
                work_intervals.append(
                    (row['weight_start_ts'], row['weight_end_ts']))

            # 计算阶段
            if row['compute_start_ts'] < row['compute_end_ts']:
                work_intervals.append(
                    (row['compute_start_ts'], row['compute_end_ts']))

            # 收集等待时间区间
            wait_intervals = []

            # 结构等待
            if row['submit_time_ts'] < row['structure_start_ts']:
                wait_intervals.append(
                    (row['submit_time_ts'], row['structure_start_ts']))

            # 权重等待
            if row['structure_end_ts'] < row['weight_start_ts']:
                wait_intervals.append(
                    (row['structure_end_ts'], row['weight_start_ts']))

            # 计算等待
            if row['weight_end_ts'] < row['compute_start_ts']:
                wait_intervals.append(
                    (row['weight_end_ts'], row['compute_start_ts']))

            # 合并工作和等待区间
            merged_work = merge_intervals(work_intervals)
            merged_wait = merge_intervals(wait_intervals)

            # 计算总的工作和等待时间
            total_work_time = sum(end - start for start, end in merged_work)
            total_wait_time = sum(end - start for start, end in merged_wait)

            # 计算总时间
            total_time = row['end_time_ts'] - row['submit_time_ts']

            structure_duration = row['structure_end_ts'] - \
                row['structure_start_ts']
            init_weight_duration = row['init_weight_end_ts'] - \
                row['init_weight_start_ts']
            preload_duration = row['weight_preload_end_ts'] - \
                row['weight_preload_start_ts']
            if not pd.isnull(preload_duration):
                preload_duration = preload_duration
            else:
                preload_duration = 0
            weight_duration = row['weight_end_ts'] - row['weight_start_ts']

            compute_duration = row['compute_end_ts'] - row['compute_start_ts']

            total_duration = structure_duration + init_weight_duration + \
                preload_duration + weight_duration + compute_duration
            total_pipeline_time = row['end_time_ts'] - row['submit_time_ts']

            # 计算重叠比例
            if total_time > 0:
                overlap_ratio = (total_duration /
                                 (total_pipeline_time - total_wait_time)) * 100
                return overlap_ratio
            return 0

        # 计算每个任务的重叠比例
        plot_data['overlap_ratio'] = plot_data.apply(
            calculate_overlap_ratio, axis=1)

        # 按模型和策略分组计算平均值
        grouped_data = plot_data.groupby(['model_name', 'strategy'])[
            'overlap_ratio'].mean().reset_index()
        grouped_data['strategy'] = grouped_data['strategy'].map(
            STRATEGY_DISPLAY_NAMES)

        # 使用seaborn绘制分组柱状图
        sns.barplot(
            data=grouped_data,
            x='model_name',
            y='overlap_ratio',
            hue='strategy',
            edgecolor='black',
            order=sorted(
                grouped_data['model_name'].unique(),
                key=lambda x: (
                    get_model_family(x),
                    int(re.search(r'\d+', x).group()
                        ) if re.search(r'\d', x) else 0,
                    x
                )
            ),
            hue_order=[s for s in STRATEGY_ORDER if s != 'Sync']
        )

        plt.ylabel('Pipeline Overlap Ratio (%)', weight='bold')
        # plt.ylim(0, 100)
        plt.xlabel('')
        plt.legend(
            loc="upper center",
            ncol=len(grouped_data['strategy'].unique()),
            frameon=False
        )

        plt.savefig(f'{self.plot_dir}/pipeline_overlap_ratio.pdf',
                    format='pdf', bbox_inches='tight')
        plt.close()

    def visualize_pipeline_overlap(self):
        """可视化不同策略下的流水线阶段时间线"""
        self.init_plot_style()

        PIPELINE_UNITS = [
            ('structure', ('', '//')),  # (structure样式, init_weight样式)
            ('weight_preload', ''),
            ('weight', ''),
            ('compute', '')
        ]
        PIPELINE_UNITS_DISPLAY = {
            'structure': 'L',
            'weight_preload': 'R',
            'weight': 'A',
            'compute': 'E'
        }

        # 策略排序
        STRATEGY_ORDER = [
            'ASYNC_PRELOAD_MINI',
            'ASYNC_PRELOAD', 'ASYNC_MINI', 'ASYNC']

        for model in self.combined_df['model_name'].unique():
            model_data = self.combined_df[self.combined_df['model_name'] == model]
            layers = sorted(
                model_data['layer_name'].unique(),
                key=lambda x: (x.startswith('layer'),
                               int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0))

            # 为每个层创建唯一颜色映射
            layer_colors = sns.color_palette("tab20", n_colors=len(layers))
            color_map = {
                layer: color for layer,
                color in zip(layers, layer_colors)}

            # 创建图表布局
            fig, axs = plt.subplots(
                len(STRATEGY_ORDER), 1,
                figsize=(
                    DEFAULT_FIGSIZE[0], 5),
                sharex=True
            )
            plt.subplots_adjust(hspace=0.05)

            # 为每个策略创建子图
            for strategy_idx, strategy in enumerate(STRATEGY_ORDER):
                ax = axs[strategy_idx] if len(STRATEGY_ORDER) > 1 else axs
                strategy_data = model_data[model_data['strategy'] == strategy]
                if strategy_data.empty:
                    continue

                # 时间戳归一化（从0开始）并转换为毫秒
                min_time = strategy_data[[
                    'structure_start_ts', 'weight_start_ts', 'compute_start_ts'
                ]].min().min()

                # 转换为毫秒并保留3位小数精度
                time_scale = 1000  # 秒到毫秒转换系数

                # 绘制每个pipeline unit
                for unit_idx, (unit, hatches) in enumerate(PIPELINE_UNITS):
                    height = 0.4
                    y_pos = len(PIPELINE_UNITS) - unit_idx - 1

                    for layer_idx, layer in enumerate(layers):
                        layer_data = strategy_data[strategy_data['layer_name'] == layer]
                        if not layer_data.empty:
                            if unit == 'weight_preload':
                                preload_start = layer_data['weight_preload_start_ts'].values[0]
                                preload_end = layer_data['weight_preload_end_ts'].values[0]
                                # 当预加载时间无效时跳过绘制
                                if preload_end - preload_start <= 0:
                                    continue

                            if unit == 'structure':
                                # 绘制structure部分
                                structure_start = layer_data['structure_start_ts'].values[0]
                                structure_end = layer_data['structure_end_ts'].values[0]
                                ax.barh(y=y_pos,
                                        width=(structure_end -
                                               structure_start) * time_scale,
                                        left=(structure_start -
                                              min_time) * time_scale,
                                        height=height,
                                        color=color_map[layer],
                                        edgecolor='black',
                                        hatch=hatches[0],
                                        alpha=0.8)

                                # 在其上绘制init_weight部分（//填充）
                                init_weight_start = layer_data['init_weight_start_ts'].values[0]
                                init_weight_end = layer_data['init_weight_end_ts'].values[0]
                                ax.barh(y=y_pos,
                                        width=(init_weight_end -
                                               init_weight_start) * time_scale,
                                        left=(init_weight_start -
                                              min_time) * time_scale,
                                        height=height,
                                        color=color_map[layer],
                                        edgecolor='black',
                                        hatch=hatches[1],
                                        alpha=0.8)
                            else:
                                # 处理预加载和其他阶段
                                start = (
                                    layer_data[f'{unit}_start_ts'].values[0] - min_time) * time_scale
                                end = (
                                    layer_data[f'{unit}_end_ts'].values[0] - min_time) * time_scale
                                if end - start > 0:  # 仅绘制有效时间段
                                    ax.barh(y=y_pos,
                                            width=end - start,
                                            left=start,
                                            height=height,
                                            color=color_map[layer],
                                            edgecolor='black',
                                            hatch=hatches,
                                            alpha=0.8)

                # 设置子图样式
                strategy_name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
                if strategy_name != 'Cicada':
                    y_position = 0.5
                else:
                    y_position = 1
                ax.set_title(f"{strategy_name}",
                             fontsize=plt.rcParams['axes.titlesize']+2,
                             loc='center', y=y_position)
                ax.set_yticks(range(len(PIPELINE_UNITS)))
                ax.set_yticklabels([PIPELINE_UNITS_DISPLAY.get(u[0], u[0].replace('_', ' ').title())
                                    for u in reversed(PIPELINE_UNITS)], fontweight='bold',
                                   fontsize=plt.rcParams['ytick.labelsize']+4)  # 反转标签顺序
                ax.tick_params(
                    axis='x', labelsize=plt.rcParams['xtick.labelsize'])
                ax.grid(True, axis='x', linestyle='--', alpha=0.7)

                # 隐藏非最后子图的x轴刻度线
                if strategy_idx != len(STRATEGY_ORDER)-1:
                    ax.tick_params(
                        axis='x',
                        which='both',
                        bottom=False,      # 隐藏底部刻度线
                        labelbottom=False   # 隐藏底部标签
                    )
                else:
                    ax.tick_params(axis='x', which='both',
                                   labelsize=plt.rcParams['xtick.labelsize']+4,
                                   length=3)  # 显示最后子图的刻度

            # 设置全局标签
            fig.supxlabel('Time (ms)', weight='bold',
                          fontsize=plt.rcParams['axes.labelsize']+6)

            # 层颜色图例
            layer_handles = [Patch(
                facecolor=color, edgecolor='black',
                label=layer) for layer, color in color_map.items()]
            # Pipeline unit样式图例
            unit_handles = [
                # Patch(
                #     facecolor='white', edgecolor='black',
                #     hatch='', label='Structure Build'),
                Patch(
                    facecolor='white', edgecolor='black',
                    hatch='//', label='Params'),
                # Patch(
                #     facecolor='white', edgecolor='black',
                #     hatch='xx', label='Weight Preload'),
                # Patch(
                #     facecolor='white', edgecolor='black',
                #     hatch='..', label='Weight Load'),
                # Patch(
                #     facecolor='white', edgecolor='black',
                #     hatch='--', label='Compute')
            ]
            columnspacing = 0.4
            fontsize = plt.rcParams['legend.fontsize'] + 4
            ncol = 2
            if model.startswith('resnet'):
                fontsize = plt.rcParams['legend.fontsize'] + 2
                ncol = 3
                columnspacing = 0.1
            # 合并图例并分两列显示
            fig.legend(
                handles=unit_handles + layer_handles,
                bbox_to_anchor=(0.9, 0.89),
                fontsize=fontsize,
                frameon=False,
                handletextpad=0.1,
                columnspacing=columnspacing,
                ncol=ncol)

            plt.savefig(f'{self.plot_dir}/pipeline_timeline_{model}.pdf',
                        bbox_inches='tight')
            plt.close()

    def analyze_pipeline_utilization(self):
        """分析流水线利用率 - 考虑重叠时间"""
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'  # 使用 STIX 字体渲染数学公式

        self.init_plot_style()
        plot_data = self._calculate_timing_metrics(self.combined_df.copy())
        plot_data = plot_data[plot_data['strategy'] != 'SYNC']

        plot_data['model_name'] = plot_data['model_name'].map(
            lambda x: MODEL_DISPLAY_NAMES.get(x, x)
        )
        plot_data['strategy'] = plot_data['strategy'].map(
            lambda x: STRATEGY_DISPLAY_NAMES.get(x, x)
        )

        busy_time_df = (
            plot_data.groupby(['strategy', 'model_name', 'model_request_id'])
            .apply(calculate_utilization_group)
            .reset_index()
        )
        merged_df = busy_time_df.copy()

        model_data = merged_df.copy()
        model_grouped = model_data.groupby(['model_name', 'strategy']).agg({
            'utilization': 'mean',
            'busy_time_ms': 'mean',
            'model_pipeline_time_ms': 'mean'
        }).reset_index()

        # 同样为模型家族计算
        family_data = merged_df.copy()
        family_data['model_family'] = family_data['model_name'].apply(
            get_model_family)
        family_grouped = family_data.groupby(['model_family', 'strategy']).agg({
            'utilization': 'mean',
            'busy_time_ms': 'mean',
            'model_pipeline_time_ms': 'mean'
        }).reset_index()

        # 绘制分组柱状图
        sns.barplot(
            data=merged_df,
            x='model_name',
            y='utilization',
            hue='strategy',
            edgecolor='black',
            order=sorted(
                merged_df['model_name'].unique(),
                key=lambda x: (
                    get_model_family(x),
                    int(re.search(r'\d+', x).group()
                        ) if re.search(r'\d', x) else 0,
                    x
                )
            ),
            hue_order=[s for s in STRATEGY_ORDER if s != 'Tetris']
        )

        plt.ylabel('Utilization (%)', weight='bold',
                   fontsize=plt.rcParams['ytick.labelsize']+2)
        plt.xticks(rotation=15, weight='bold')
        plt.ylim(0, 120)
        plt.xlabel('Model Name', weight='bold')
        plt.legend(
            loc="upper center",
            ncol=len(merged_df['strategy'].unique()),
            bbox_to_anchor=(0.5, 1.05),
            frameon=False,
            fontsize=plt.rcParams['legend.fontsize']+2
        )

        plt.savefig(f'{self.plot_dir}/pipeline_utilization.pdf',
                    format='pdf', bbox_inches='tight')
        plt.close()
        # 计算优化比例

        def calculate_improvements(data, group_col='model_name'):
            improvements = []
            for group in data[group_col].unique():
                group_data = data[data[group_col] == group]

                # 获取基准策略的数据
                pisel_data = group_data[group_data['strategy'] == 'PISeL']
                preload_data = group_data[group_data['strategy'] == 'Preload']

                # 计算基准策略的平均值
                baseline_util = (
                    pisel_data['utilization'].values[0] + preload_data['utilization'].values[0]) / 2
                baseline_busy = (
                    pisel_data['busy_time_ms'].values[0] + preload_data['busy_time_ms'].values[0]) / 2
                baseline_total = (
                    pisel_data['model_pipeline_time_ms'].values[0] + preload_data['model_pipeline_time_ms'].values[0]) / 2

                # 获取优化策略的数据
                mini_data = group_data[group_data['strategy'] == 'Mini']
                cicada_data = group_data[group_data['strategy'] == 'Cicada']

                # 计算优化策略的平均值
                optimized_util = (
                    mini_data['utilization'].values[0] + cicada_data['utilization'].values[0]) / 2
                optimized_busy = (
                    mini_data['busy_time_ms'].values[0] + cicada_data['busy_time_ms'].values[0]) / 2
                optimized_total = (
                    mini_data['model_pipeline_time_ms'].values[0] + cicada_data['model_pipeline_time_ms'].values[0]) / 2

                # 计算提升比例
                improvement = (
                    (optimized_util - baseline_util) / baseline_util) * 100
                busy_time_reduction = (
                    (baseline_busy - optimized_busy) / baseline_busy) * 100
                total_time_reduction = (
                    (baseline_total - optimized_total) / baseline_total) * 100

                improvements.append({
                    group_col: group,
                    'baseline_util': baseline_util,
                    'optimized_util': optimized_util,
                    'improvement': improvement,
                    'baseline_busy_time': baseline_busy,
                    'optimized_busy_time': optimized_busy,
                    'busy_time_reduction': busy_time_reduction,
                    'baseline_total_time': baseline_total,
                    'optimized_total_time': optimized_total,
                    'total_time_reduction': total_time_reduction
                })
            return pd.DataFrame(improvements)

        # 计算模型级别的改进
        model_improvements = calculate_improvements(model_grouped)
        # 计算模型家族级别的改进
        family_improvements = calculate_improvements(
            family_grouped, 'model_family')

        # 保存分析结果
        output_path = os.path.join(
            self.log_dir, 'analyze_pipeline_utilization.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            # 1. 输出模型家族级别的分析
            f.write("一、模型家族级别的流水线分析：\n")
            f.write("-" * 50 + "\n\n")

            for _, row in family_improvements.iterrows():
                f.write(f"模型家族: {row['model_family']}\n")
                f.write("基准策略 (PISeL & Preload 平均):\n")
                f.write(f"  - 利用率: {row['baseline_util']:.2f}%\n")
                f.write(f"  - 忙碌时间: {row['baseline_busy_time']:.2f}ms\n")
                f.write(f"  - 总时间: {row['baseline_total_time']:.2f}ms\n")
                f.write(
                    f"  - 总时间是忙碌时间的: {row['baseline_total_time']/row['baseline_busy_time']:.2f}x\n")

                f.write("\n优化策略 (Mini & Cicada 平均):\n")
                f.write(f"  - 利用率: {row['optimized_util']:.2f}%\n")
                f.write(f"  - 忙碌时间: {row['optimized_busy_time']:.2f}ms\n")
                f.write(f"  - 总时间: {row['optimized_total_time']:.2f}ms\n")
                f.write(
                    f"  - 总时间是忙碌时间的: {row['optimized_total_time']/row['optimized_busy_time']:.2f}x\n")
                f.write("\n性能改进:\n")
                f.write(f"  - 利用率提升: {row['improvement']:.2f}%\n")
                f.write(f"  - 忙碌时间减少: {row['busy_time_reduction']:.2f}%\n")
                f.write(f"  - 总时间减少: {row['total_time_reduction']:.2f}%\n")
                f.write("-" * 30 + "\n\n")

            f.write(f"模型家族平均改进:\n")
            f.write(
                f"  - 利用率提升: {family_improvements['improvement'].mean():.2f}%\n")
            f.write(
                f"  - 忙碌时间减少: {family_improvements['busy_time_reduction'].mean():.2f}%\n")
            f.write(
                f"  - 总时间减少: {family_improvements['total_time_reduction'].mean():.2f}%\n\n")

            # 2. 输出具体模型级别的分析
            f.write("\n二、具体模型级别的流水线分析:\n")
            f.write("-" * 50 + "\n\n")

            for _, row in model_improvements.iterrows():
                f.write(f"模型: {row['model_name']}\n")
                f.write("基准策略 (PISeL & Preload 平均):\n")
                f.write(f"  - 利用率: {row['baseline_util']:.2f}%\n")
                f.write(f"  - 忙碌时间: {row['baseline_busy_time']:.2f}ms\n")
                f.write(f"  - 总时间: {row['baseline_total_time']:.2f}ms\n")

                f.write("\n优化策略 (Mini & Cicada 平均):\n")
                f.write(f"  - 利用率: {row['optimized_util']:.2f}%\n")
                f.write(f"  - 忙碌时间: {row['optimized_busy_time']:.2f}ms\n")
                f.write(f"  - 总时间: {row['optimized_total_time']:.2f}ms\n")

                f.write("\n性能改进:\n")
                f.write(f"  - 利用率提升: {row['improvement']:.2f}%\n")
                f.write(f"  - 忙碌时间减少: {row['busy_time_reduction']:.2f}%\n")
                f.write(f"  - 总时间减少: {row['total_time_reduction']:.2f}%\n")
                f.write("-" * 30 + "\n\n")

            f.write(f"所有模型平均改进:\n")
            f.write(
                f"  - 利用率提升: {model_improvements['improvement'].mean():.2f}%\n")
            f.write(
                f"  - 忙碌时间减少: {model_improvements['busy_time_reduction'].mean():.2f}%\n")
            f.write(
                f"  - 总时间减少: {model_improvements['total_time_reduction'].mean():.2f}%\n\n")

        print(f"流水线利用率分析已保存到: {output_path}")

        self.init_plot_style(figsize=(7, 2.5))
        # 准备绘图数据

        strategies = []
        for strategy in STRATEGY_ORDER:
            if strategy in ['Preload', 'Mini']:
                continue
            strategies.append(strategy)
        aux_df = merged_df[merged_df['strategy'].isin(
            strategies)]
        aux_df.rename(columns={'strategy': 'Strategy'}, inplace=True)

        df_melted = aux_df.melt(
            id_vars=['model_name', 'Strategy'],          # 不变的列
            value_vars=['busy_time_ms', 'model_pipeline_time_ms'],
            var_name='Time Type',       # 新列，用来区分两种指标
            value_name='time_ms'        # 新列，存放数值
        )
        # 为了在图例中显示更易读的标签，可以映射一下
        df_melted['Time Type'] = df_melted['Time Type'].map({
            'busy_time_ms': 'Total Active Time',
            'model_pipeline_time_ms': 'Total Pipeline Time'
        })

        group_data = df_melted.groupby(['Strategy', 'model_name', 'Time Type'])[
            'time_ms'].mean().reset_index()
        order_list = sorted(
            plot_data['model_name'].unique(),
            key=lambda x: (
                get_model_family(x),
                int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0,
                x
            )
        )
        group_data['model_name'] = pd.Categorical(
            group_data['model_name'], categories=order_list, ordered=True)

        ax = sns.lineplot(
            data=group_data,
            x='model_name',
            y='time_ms',
            hue='Strategy',       # 不同策略不同颜色
            style='Time Type',    # 不同时间指标不同线型/标记
            markers=True,
            markersize=15,
            linewidth=2
        )
        plt.xticks(rotation=15, weight='bold')
        plt.xlabel("Model Name", weight='bold')
        plt.ylabel("Time (ms)", weight='bold',
                   fontsize=plt.rcParams['ytick.labelsize']+2)
        plt.tight_layout()

        # 获取当前图例的句柄和标签，但忽略分组标题
        handles, labels = ax.get_legend_handles_labels()
        # 移除'Strategy'和'Time Type'标题（它们通常是在索引0和第一个分组后的位置）
        legend_items_to_keep = [i for i, label in enumerate(labels) if label not in [
            'Strategy', 'Time Type']]
        handles = [handles[i] for i in legend_items_to_keep]
        labels = [labels[i] for i in legend_items_to_keep]

        # 删除现有图例并创建新图例
        ax.get_legend().remove()
        ax.legend(handles=handles, labels=labels,
                  markerscale=0.5, frameon=False,
                  fontsize=plt.rcParams['legend.fontsize']+2)

        plt.savefig(f'{self.plot_dir}/pipeline_times_comparison.pdf',
                    format='pdf', bbox_inches='tight')
        plt.close()


def main():
    analyzer = PerformanceAnalyzer(log_dir='./expected_results')
    analyzer.load_data(suffix="model_eval")
    analyzer.analyze_unified_load_time()
    analyzer.load_data(suffix="model_eval", end_suffix="_memory")
    analyzer.analyze_unified_memory()

    analyzer.load_data(suffix="pipeline_stats")
    analyzer.analyze_pipeline_time_breakdown()
    analyzer.visualize_pipeline_overlap()
    analyzer.analyze_pipeline_utilization()

    print("分析完成！请查看 logs 目录下生成的图表。")


if __name__ == "__main__":
    main()
