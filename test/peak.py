import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, peak_prominences
import argparse
import sys

def load_csv_data(file_path):
    """加载CSV文件数据"""
    try:
        df = pd.read_csv(file_path)
        print(f"CSV文件列名: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
        sys.exit(1)
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        sys.exit(1)

def find_peaks_analysis(x_data, y_data, **kwargs):
    """
    寻峰分析主函数
    
    参数:
    x_data: x轴数据
    y_data: y轴数据
    **kwargs: 传递给find_peaks的参数
    
    返回:
    peaks: 峰的位置索引
    properties: 峰的属性字典
    """
    # 默认参数
    default_kwargs = {
        'height': None,      # 最小峰高
        'threshold': None,   # 阈值
        'distance': 10,      # 峰之间的最小距离
        'prominence': None,  # 最小突出度
        'width': None,       # 最小宽度
        'wlen': None         # 计算突出度和宽度的窗口长度
    }
    
    # 更新默认参数
    for key, value in kwargs.items():
        if key in default_kwargs:
            default_kwargs[key] = value
    
    # 寻峰
    peaks, properties = find_peaks(y_data, **default_kwargs)
    
    return peaks, properties

def calculate_fwhm(x_data, y_data, peaks, properties, rel_height=0.5):
    """
    计算半高全宽(FWHM)
    
    参数:
    x_data: x轴数据
    y_data: y轴数据
    peaks: 峰的位置索引
    properties: 峰的属性
    rel_height: 相对高度，默认为0.5（半高）
    
    返回:
    fwhm_values: 每个峰的FWHM值
    left_ips: 左交点索引
    right_ips: 右交点索引
    """
    # 计算峰宽（在相对高度处）
    width_results = peak_widths(y_data, peaks, rel_height=rel_height)
    
    # 获取宽度信息
    widths = width_results[0]  # 宽度（以索引为单位）
    width_heights = width_results[1]  # 宽度处的高度
    left_ips = width_results[2]  # 左交点插值位置
    right_ips = width_results[3]  # 右交点插值位置
    
    # 将索引转换为x坐标
    fwhm_values = []
    left_x = []
    right_x = []
    
    for i, peak_idx in enumerate(peaks):
        # 线性插值将索引位置转换为x坐标
        if peak_idx < len(x_data) - 1:
            # 计算FWHM对应的x坐标
            left_idx = left_ips[i]
            right_idx = right_ips[i]
            
            # 插值得到x坐标
            if left_idx.is_integer():
                left_x_val = x_data[int(left_idx)]
            else:
                idx_low = int(np.floor(left_idx))
                idx_high = int(np.ceil(left_idx))
                if idx_high < len(x_data):
                    frac = left_idx - idx_low
                    left_x_val = x_data[idx_low] * (1 - frac) + x_data[idx_high] * frac
                else:
                    left_x_val = x_data[idx_low]
            
            if right_idx.is_integer():
                right_x_val = x_data[int(right_idx)]
            else:
                idx_low = int(np.floor(right_idx))
                idx_high = int(np.ceil(right_idx))
                if idx_high < len(x_data):
                    frac = right_idx - idx_low
                    right_x_val = x_data[idx_low] * (1 - frac) + x_data[idx_high] * frac
                else:
                    right_x_val = x_data[idx_low]
            
            fwhm = right_x_val - left_x_val
        else:
            fwhm = 0
            left_x_val = x_data[peak_idx]
            right_x_val = x_data[peak_idx]
        
        fwhm_values.append(fwhm)
        left_x.append(left_x_val)
        right_x.append(right_x_val)
    
    return fwhm_values, left_x, right_x

def analyze_peaks(x_data, y_data, peaks, properties):
    """
    分析峰并返回详细信息
    
    返回:
    DataFrame包含峰的详细信息
    """
    # 计算突出度
    prominences = peak_prominences(y_data, peaks)[0]
    
    # 计算FWHM
    fwhm_values, left_x, right_x = calculate_fwhm(x_data, y_data, peaks, properties)
    
    # 收集结果
    results = []
    for i, peak_idx in enumerate(peaks):
        peak_info = {
            '峰序号': i + 1,
            'x位置': x_data[peak_idx],
            'y位置': y_data[peak_idx],
            '索引位置': peak_idx,
            '峰高': y_data[peak_idx],
            '突出度': prominences[i],
            '半高全宽(FWHM)': fwhm_values[i],
            '左边界(x)': left_x[i],
            '右边界(x)': right_x[i]
        }
        results.append(peak_info)
    
    return pd.DataFrame(results)

def plot_results(x_data, y_data, peaks, df_results, output_file=None):
    """绘制结果图形"""
    plt.figure(figsize=(12, 8))
    
    # 绘制原始数据
    plt.plot(x_data, y_data, 'b-', label='原始数据', linewidth=1, alpha=0.7)
    
    # 标记峰的位置
    plt.plot(x_data[peaks], y_data[peaks], 'rx', markersize=10, label='检测到的峰')
    
    # 标记FWHM
    for _, row in df_results.iterrows():
        # 绘制半高线
        half_height = row['y位置'] - row['突出度'] / 2
        plt.hlines(half_height, row['左边界(x)'], row['右边界(x)'], 
                  colors='g', linestyles='dashed', linewidth=1)
        
        # 标记FWHM宽度
        plt.plot([row['左边界(x)'], row['右边界(x)']], 
                [half_height, half_height], 'go', markersize=5)
        
        # 添加文本标签
        plt.text(row['x位置'], row['y位置'] * 1.05, 
                f"峰{int(row['峰序号'])}", ha='center', fontsize=9)
        plt.text((row['左边界(x)'] + row['右边界(x)']) / 2, 
                half_height * 0.95, 
                f"FWHM={row['半高全宽(FWHM)']:.3f}", 
                ha='center', fontsize=8, color='green')
    
    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('寻峰分析结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图形已保存到: {output_file}")
    
    plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CSV文件寻峰分析工具')
    parser.add_argument('input_file', help='输入CSV文件路径')
    parser.add_argument('--x-col', default=0, help='X轴数据列名或索引（默认: 0）')
    parser.add_argument('--y-col', default=1, help='Y轴数据列名或索引（默认: 1）')
    parser.add_argument('--height', type=float, help='最小峰高阈值')
    parser.add_argument('--distance', type=int, default=10, help='峰之间的最小距离（默认: 10）')
    parser.add_argument('--prominence', type=float, help='最小突出度阈值')
    parser.add_argument('--width', type=float, help='最小宽度阈值')
    parser.add_argument('--output', help='输出CSV文件路径（默认: peaks_output.csv）')
    parser.add_argument('--plot', action='store_true', help='显示结果图形')
    parser.add_argument('--plot-output', help='图形输出文件路径')
    
    args = parser.parse_args()
    
    # 加载数据
    df = load_csv_data(args.input_file)
    
    # 确定x和y数据列
    try:
        if isinstance(args.x_col, str) and args.x_col in df.columns:
            x_data = df[args.x_col].values
        else:
            x_col_idx = int(args.x_col)
            x_data = df.iloc[:, x_col_idx].values
    except:
        print(f"警告: 无法获取X轴数据，使用索引作为X轴")
        x_data = np.arange(len(df))
    
    try:
        if isinstance(args.y_col, str) and args.y_col in df.columns:
            y_data = df[args.y_col].values
        else:
            y_col_idx = int(args.y_col)
            y_data = df.iloc[:, y_col_idx].values
    except:
        print(f"错误: 无法获取Y轴数据")
        sys.exit(1)
    
    print(f"数据加载完成: {len(x_data)} 个数据点")
    
    # 准备寻峰参数
    peak_kwargs = {}
    if args.height is not None:
        peak_kwargs['height'] = args.height
    if args.distance is not None:
        peak_kwargs['distance'] = args.distance
    if args.prominence is not None:
        peak_kwargs['prominence'] = args.prominence
    if args.width is not None:
        peak_kwargs['width'] = args.width
    
    # 寻峰分析
    print("正在执行寻峰分析...")
    peaks, properties = find_peaks_analysis(x_data, y_data, **peak_kwargs)
    
    if len(peaks) == 0:
        print("未检测到任何峰！")
        print("尝试调整参数：--height, --distance, --prominence")
        sys.exit(0)
    
    print(f"检测到 {len(peaks)} 个峰")
    
    # 分析峰的详细信息
    df_results = analyze_peaks(x_data, y_data, peaks, properties)
    
    # 输出结果
    output_file = args.output or 'peaks_output.csv'
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"结果已保存到: {output_file}")
    
    # 显示结果表格
    print("\n" + "="*80)
    print("寻峰分析结果:")
    print("="*80)
    print(df_results.to_string(index=False))
    
    # 计算统计信息
    print("\n" + "="*80)
    print("统计信息:")
    print("="*80)
    print(f"峰数量: {len(peaks)}")
    print(f"平均峰高: {df_results['峰高'].mean():.4f}")
    print(f"平均FWHM: {df_results['半高全宽(FWHM)'].mean():.4f}")
    print(f"最大峰高: {df_results['峰高'].max():.4f} (峰{df_results['峰高'].idxmax()+1})")
    print(f"最小FWHM: {df_results['半高全宽(FWHM)'].min():.4f} (峰{df_results['半高全宽(FWHM)'].idxmin()+1})")
    
    # 绘制图形
    if args.plot or args.plot_output:
        plot_results(x_data, y_data, peaks, df_results, args.plot_output)

if __name__ == "__main__":
    # 如果直接运行，检查是否安装了必要的库
    try:
        import scipy
        import pandas
        import matplotlib
    except ImportError as e:
        print(f"错误: 缺少必要的库。请安装: pip install numpy pandas scipy matplotlib")
        sys.exit(1)
    
    main()