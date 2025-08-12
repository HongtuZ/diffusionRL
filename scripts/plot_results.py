import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_tensorboard_data(event_file_path, scalar_tag):
    """从TensorBoard事件文件中加载指定标量数据"""
    event_acc = EventAccumulator(event_file_path)
    event_acc.Reload()
    
    if scalar_tag not in event_acc.Tags()['scalars']:
        raise ValueError(f"Scalar tag '{scalar_tag}' not found in event file")
    
    scalar_events = event_acc.Scalars(scalar_tag)
    steps = [event.step for event in scalar_events]
    values = [event.value for event in scalar_events]
    return np.array(steps), np.array(values)

def align_and_process_run_data(run_dir, scalar_tag, max_steps=1000, smooth_factor=0.6):
    """
    处理单个run目录下的所有种子数据
    返回对齐后的时间步和所有种子的值数组
    """
    # 查找所有种子的事件文件
    event_files = []
    for root, _, files in os.walk(run_dir):
        for f in files:
            if f.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, f))
    
    if len(event_files) != 4:
        print(f"Warning: Expected 4 seed files in {run_dir}, found {len(event_files)}")
    
    seed_data = []
    for event_file in event_files:
        try:
            steps, values = load_tensorboard_data(event_file, scalar_tag)
            
            # 应用指数移动平均平滑
            if smooth_factor > 0:
                smoothed_values = [values[0]]
                for j in range(1, len(values)):
                    smoothed_values.append(smoothed_values[-1] * smooth_factor + 
                                          values[j] * (1 - smooth_factor))
                values = smoothed_values
            
            seed_data.append((steps, np.array(values)))
            
        except Exception as e:
            print(f"Error processing {event_file}: {str(e)}")
            continue
    
    if not seed_data:
        return None, None
    
    # 对齐数据到相同的时间步
    all_steps = [data[0] for data in seed_data]
    min_step = max([steps.min() for steps in all_steps])
    max_step = min([steps.max() for steps in all_steps])
    
    interp_steps = np.linspace(min_step, max_step, max_steps)
    aligned_values = []
    
    for steps, values in seed_data:
        interp_values = np.interp(interp_steps, steps, values)
        aligned_values.append(interp_values)
    
    return interp_steps, np.array(aligned_values)

def plot_all_runs(env_dir, scalar_tag, title="Training Curves", 
                 xlabel="Steps", ylabel="Value", max_steps=1000,
                 style="whitegrid", palette="tab10", smooth_factor=0.6):
    """
    绘制指定env目录下所有run的训练曲线
    
    参数:
        env_dir: 环境目录路径(如'results/env_name')
        scalar_tag: 要绘制的标量标签
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        max_steps: 最大插值步数
        style: Seaborn样式
        palette: 调色板名称
        smooth_factor: 平滑因子(0-1)
    """
    # 设置Seaborn样式
    sns.set_style(style)
    plt.figure(figsize=(12, 7))
    
    # 查找所有run目录
    run_dirs = []
    for entry in os.listdir(env_dir):
        full_path = os.path.join(env_dir, entry)
        if os.path.isdir(full_path):
            run_dirs.append((entry, full_path))  # (run_name, run_path)
    
    if not run_dirs:
        print(f"No run directories found in {env_dir}")
        return
    
    # 准备绘图数据
    all_data = []
    color_palette = sns.color_palette(palette, len(run_dirs))
    
    for i, (run_name, run_dir) in enumerate(run_dirs):
        steps, aligned_values = align_and_process_run_data(
            run_dir, scalar_tag, max_steps, smooth_factor)
        
        if steps is None or aligned_values is None:
            continue
        
        # 计算均值和标准差
        mean_values = np.mean(aligned_values, axis=0)
        std_values = np.std(aligned_values, axis=0)
        
        # 为当前run创建DataFrame
        df = pd.DataFrame({
            'Step': steps,
            'Value': mean_values,
            'Std': std_values,
            'Run': run_name
        })
        all_data.append(df)
        
        # 绘制当前run的曲线
        plt.plot(steps, mean_values, 
                 color=color_palette[i], 
                 linewidth=2,
                 label=run_name)
        
        # 绘制标准差带
        plt.fill_between(steps, 
                        mean_values - std_values, 
                        mean_values + std_values,
                        color=color_palette[i], 
                        alpha=0.2)
    
    if not all_data:
        print("No valid data to plot.")
        return
    
    # 美化图表
    plt.title(f"{os.path.basename(env_dir)}", fontsize=14, pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 调整图例
    plt.legend(
        title="Runs",
        loc='lower right',
        fontsize=10
    )
    
    plt.tight_layout()
    plt.savefig(f'{env_dir}/comparison_curves.png', dpi=300, bbox_inches='tight')
    # plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot TensorBoard training curves for multiple runs')
    parser.add_argument('env_dir', type=str, 
                       help='Path to environment directory (e.g. "results/env_name")')
    parser.add_argument('--tag', type=str, default='evaluation/average_means',
                       help='TensorBoard scalar tag to plot (default: train/loss)')
    parser.add_argument('--title', type=str, default='Return',
                       help='Plot title (default: Training Curves)')
    parser.add_argument('--ylabel', type=str, default='Return',
                       help='Y-axis label (default: Value)')
    parser.add_argument('--palette', type=str, default='tab10',
                       help='Seaborn color palette (default: tab10)')
    parser.add_argument('--smooth', type=float, default=0.6,
                       help='Smoothing factor (0-1, default: 0.6)')
    
    args = parser.parse_args()
    
    # 设置Seaborn上下文
    sns.set_context("notebook", font_scale=1.1)
    
    # 绘制曲线
    plot_all_runs(
        env_dir=args.env_dir,
        scalar_tag=args.tag,
        title=args.title,
        ylabel=args.ylabel,
        palette=args.palette,
        smooth_factor=args.smooth
    )

if __name__ == "__main__":
    main()