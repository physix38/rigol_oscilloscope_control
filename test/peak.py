import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d







def find_peaks_in_csv(csv_file, x_col=0, y_col=1, 
                      prominence=1, width=1, distance=1,
                      height=None, wlen=None, rel_height=0.5,
                      plot=False, title=None, x_label=None, y_label=None):
    """
    从CSV文件中读取数据并进行寻峰分析
    
    参数:
    ----------
    csv_file : str
        CSV文件路径
    x_col : int or str, 默认 0
        X轴数据列索引或列名
    y_col : int or str, 默认 1
        Y轴数据列索引或列名
    prominence : float, 默认 1
        峰的最小突出度
    width : float, 默认 1
        峰的最小宽度（样本数）
    distance : float, 默认 1
        峰之间的最小距离（样本数）
    height : float, 可选
        峰的最小高度
    wlen : int, 可选
        计算突出度时的窗口长度
    rel_height : float, 默认 0.5
        计算半高全宽时的相对高度（0-1之间）
    plot : bool, 默认 False
        是否绘制寻峰结果图
    title : str, 可选
        图表标题
    x_label : str, 可选
        X轴标签
    y_label : str, 可选
        Y轴标签
    
    返回:
    -------
    peaks_df : pandas.DataFrame
        包含峰位置、高度和半高全宽的数据框
    """
    


    # 1. 读取CSV文件
    # skiprows忽略前11行
    try:
        data = pd.read_csv(csv_file,skiprows=11, names=['x_col', 'y_col'])
    except Exception as e:
        raise ValueError(f"无法读取CSV文件: {e}")
    print(data.columns.tolist())
    # 2. 提取X和Y数据
    if isinstance(x_col, str):
        x = data['x_col'].values
    else:
        x = data.iloc[:, x_col].values
    
    if isinstance(y_col, str):
        y = data['y_col'].values
    else:
        y = data.iloc[:, y_col].values
    y=y-y[0]
    x=x*10000
    # 3. 寻峰
    peaks, properties = find_peaks(
        y, 
        prominence=prominence,
        #width=width,
        distance=distance,
        #height=height,
        #wlen=wlen
    )
    
    if len(peaks) == 0:
        print("未找到符合条件的峰")
        return pd.DataFrame()
    
    # 4. 计算峰的宽度（半高全宽）
    widths, width_heights, left_ips, right_ips = peak_widths(
        y, peaks, rel_height=rel_height
    )
    
    # 5. 准备结果数据
    results = []
    for i, peak_idx in enumerate(peaks):
        peak_info = {
            'peak_index': peak_idx,  # 在数组中的索引
            'x_position*10000': x[peak_idx],  # X轴位置
            'y_height': y[peak_idx],  # 峰高度
            'fwhm': widths[i],  # 半高全宽（样本数）
            'fwhm_x*10000': x[int(np.round(right_ips[i]))] - x[int(np.round(left_ips[i]))],  # X轴上的FWHM
            'left_base': x[int(np.round(left_ips[i]))],  # 左半高点X坐标
            'right_base': x[int(np.round(right_ips[i]))],  # 右半高点X坐标
            'prominence': properties['prominences'][i] if 'prominences' in properties else np.nan,
            'width': properties['widths'][i] if 'widths' in properties else np.nan
        }
        results.append(peak_info)
    
    peaks_df = pd.DataFrame(results)
    
    # 6. 可选：绘制结果
    if plot:
        plt.figure(figsize=(12, 6))
        
        # 绘制原始数据
        plt.plot(x, y, 'b-', label='原始数据', alpha=0.7, linewidth=1)
        
        # 标记峰的位置
        plt.plot(x[peaks], y[peaks], 'rx', markersize=10, label='峰位置')
        
        # 绘制半高全宽
        for i, peak_idx in enumerate(peaks):
            # 绘制水平线表示半高
            half_max = y[peak_idx] * rel_height
            plt.hlines(half_max, 
                      x[int(np.round(left_ips[i]))], 
                      x[int(np.round(right_ips[i]))],
                      colors='g', linestyles='--', linewidth=2)
            
            # 标记FWHM
            plt.text(x[peak_idx], half_max, 
                    f'FWHM={peaks_df["fwhm_x"].iloc[i]:.3f}',
                    verticalalignment='bottom',
                    horizontalalignment='center')
        
        # 图表装饰
        plt.xlabel(x_label if x_label else 'X轴')
        plt.ylabel(y_label if y_label else 'Y轴')
        plt.title(title if title else '寻峰结果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return peaks_df


def find_peaks_with_baseline(csv_file, x_col=0, y_col=1, 
                            baseline_method='poly', poly_order=3,
                            **kwargs):
    """
    带基线扣除的寻峰函数
    
    参数:
    ----------
    csv_file : str
        CSV文件路径
    x_col, y_col : 同find_peaks_in_csv
    baseline_method : str
        基线扣除方法: 'poly' (多项式拟合) 或 'rolling' (滚动中值)
    poly_order : int
        多项式拟合的阶数
    **kwargs : 传递给find_peaks_in_csv的其他参数
    
    返回:
    -------
    peaks_df : pandas.DataFrame
        寻峰结果
    """
    
    # 读取数据
    data = pd.read_csv(csv_file,skiprows=11, names=['x_col', 'y_col'])
    print(data.columns.tolist())
    if isinstance(x_col, str):
        x = data['x_col'].values
    else:
        x = data.iloc[:, x_col].values
    
    if isinstance(y_col, str):
        y = data['y_col'].values
    else:
        y = data.iloc[:, y_col].values
    
    # 基线扣除
    if baseline_method == 'poly':
        # 多项式拟合基线
        coeffs = np.polyfit(x, y, poly_order)
        baseline = np.polyval(coeffs, x)
    elif baseline_method == 'rolling':
        # 滚动中值作为基线
        window = kwargs.get('window_size', len(x)//10)
        baseline = pd.Series(y).rolling(window, center=True, min_periods=1).median().values
    else:
        baseline = np.zeros_like(y)
    
    # 扣除基线
    y_corrected = y - baseline
    
    # 使用校正后的数据进行寻峰
    temp_df = pd.DataFrame({'x': x, 'y_corrected': y_corrected})
    temp_file = 'temp_peak_data.csv'
    temp_df.to_csv(temp_file, index=False)
    
    try:
        peaks_df = find_peaks_in_csv(
            temp_file, 
            x_col=0, 
            y_col=1,
            **kwargs
        )
    finally:
        # 清理临时文件
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    return peaks_df


# 使用示例
if __name__ == "__main__":
    
    
    peaks_baseline = find_peaks_in_csv('./processedtestdata1/Scope_Data_Ch1_20260208_173228.csv',
        x_col='x',
        y_col='y',
        distance=10000,
        prominence=0.01,
        
        
        plot=False,
       
        title='寻峰')
        
        
    
    print("寻峰结果:")
    print(peaks_baseline.round(3))
    
    

