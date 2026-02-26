#!/usr/bin/env python3
"""
边带调制测量脚本
调用 dataaquisition 和 peak2 模块，自动从 Rigol 示波器读取数据，
进行寻峰分析找到峰与调制边带，计算位置差的平均值和标准差。
结果输出到CSV文件。

"""

import sys
import os
import time
import glob
from datetime import datetime

# 添加当前目录到路径，以便导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import dataaquisition
import peak2
import csvprocess

def ask_modulation_value():
    """
    向用户询问边带调制的值（单位：MHz）
    返回用户输入的浮点数
    """
    while True:
        try:
            value_str = input("请输入边带调制值 (MHz，或直接按Enter使用默认值0): ").strip()
            if value_str == "":
                return 0.0
            value = float(value_str)
            return value
        except ValueError:
            print("输入无效，请输入一个数字。")

def get_files_in_folder(folder, pattern="*.csv"):
    """返回文件夹中匹配模式的文件列表（完整路径）"""
    if not os.path.exists(folder):
        return []
    return glob.glob(os.path.join(folder, pattern))

def acquire_data_and_get_files(scope_acq, save_dir='./testdata'):
    """
    使用示波器采集数据并保存为CSV文件
    返回采集后新生成的CSV文件列表（完整路径）
    """
    print("开始采集数据...")
    # 获取采集前已有的文件列表
    before_files = set(get_files_in_folder(save_dir))
    
    success = scope_acq.acquire_and_save_both_channels()
    if not success:
        print("数据采集失败")
        return None
    
    # 等待文件写入完成
    time.sleep(1)
    
    # 获取采集后的文件列表
    after_files = set(get_files_in_folder(save_dir))
    
    # 找出新增的文件
    new_files = list(after_files - before_files)
    if not new_files:
        print("警告：未发现新生成的CSV文件")
        return None
    
    print(f"采集生成的文件: {[os.path.basename(f) for f in new_files]}")
    return new_files

def process_specific_files(input_files, output_folder='./processedtestdata3'):
    """
    处理指定的CSV文件，删除空行，保存到输出文件夹
    返回处理后的文件路径列表
    """
    if not input_files:
        return []
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    processed_files = []
    for input_file in input_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_folder, filename)
        # 使用csvprocess模块的单个文件处理功能
        # csvprocess没有直接提供单个文件处理函数，我们可以调用其内部函数
        # 为简化，我们直接复制并删除空行（使用csvprocess中的process_single_csv_file）
        try:
            # 导入csvprocess中的函数
            from csvprocess import process_single_csv_file
            empty_rows = process_single_csv_file(input_file, output_file)
            print(f"处理 {filename}，删除了 {empty_rows} 个空行")
            processed_files.append(output_file)
            # 删除原始文件
            os.remove(input_file)
            #print(f"已删除原始文件: {filename}")
            
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            # 如果出错，仍尝试将原文件复制到输出文件夹
            import shutil
            shutil.copy2(input_file, output_file)
            processed_files.append(output_file)
    
    return processed_files

def find_peaks_and_calculate(csv_file, small_prominence_range=(0.01, 0.08), large_prominence_range=(0.1, None)):
    """
    对单个CSV文件进行寻峰分析，计算峰与调制边带的位置差
    返回平均值和标准差
    """
    print(f"分析文件: {csv_file}")
    # 使用peak2模块的寻峰函数
    peaks_df = peak2.find_peaks_in_csv(
        csv_file,
        x_col=0,
        y_col=1,
        distance=10000,
        prominence=0.01,
        plot=False,
        title='寻峰'
    )
    if peaks_df.empty:
        print("未找到峰")
        return None, None
    
    # 计算峰与调制边带的位置差
    enhanced_results = peak2.calculate_peak_differences_enhanced(
        peaks_df, 
        prominence_col=7,  # prominence列索引
        position_col=1,    # x_position*10000列索引
        small_prominence_range=small_prominence_range,
        large_prominence_range=large_prominence_range
    )
    
    avg_diff = enhanced_results.get('average_diff')
    std_diff = enhanced_results.get('std')
    summary  = enhanced_results.get('summary')
    if avg_diff is not None:
        avg_diff = avg_diff.iloc[0] if hasattr(avg_diff, 'iloc') else avg_diff
    if std_diff is not None:
        std_diff = std_diff.iloc[0] if hasattr(std_diff, 'iloc') else std_diff
    
    return avg_diff, std_diff,summary

def save_results_to_csv(results, output_csv):
    """
    将测量结果保存到CSV文件
    results: 列表，每个元素为字典，包含以下键：
        'timestamp', 'modulation_value', 'filename', 'average_diff', 'std_diff'
    """
    import pandas as pd
    df = pd.DataFrame(results)
    # 如果文件已存在，则追加
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(output_csv, index=False)
    print(f"结果已保存到CSV文件: {output_csv}")

def main():
    """主函数"""
    print("边带调制测量程序")
    print("=" * 50)
    
    # 1. 询问边带调制值
    modulation_value = ask_modulation_value()
    print(f"使用的边带调制值: {modulation_value} MHz")
    
    # 2. 连接示波器
    scope_acq = dataaquisition.RigolMHO984_DataAcquisition(
        visa_address='USB0::0x1AB1::0x0452::MHO9B280400571::INSTR'
    )
    
    try:
        if not scope_acq.connect():
            print("请检查以下可能的问题：")
            print("1. 示波器是否已通过USB连接")
            print("2. 是否安装了正确的VISA驱动")
            print("3. VISA地址是否正确")
            return
        
        scope_acq.setup_acquisition()
        
        # 3. 采集数据并获取生成的文件
        new_files = acquire_data_and_get_files(scope_acq)
        if not new_files:
            print("无法获取采集的文件，退出")
            return
        
        # 4. 处理这些文件（删除空行）
        processed_files = process_specific_files(new_files)
        if not processed_files:
            print("没有处理后的文件可供分析")
            return
        
        # 5. 寻峰分析（通常我们只分析通道1，但也可以分析所有处理后的文件）
        # 筛选通道1的文件（文件名包含 Ch1）
        ch1_files = [f for f in processed_files if 'Ch1' in os.path.basename(f)]
        if not ch1_files:
            # 如果没有通道1文件，则使用第一个文件
            ch1_files = [processed_files[0]]
        
        results = []
        for csv_file in ch1_files:
            avg_diff, std_diff,summary = find_peaks_and_calculate(csv_file)
            if avg_diff is None or std_diff is None:
                print(f"文件 {csv_file} 无法计算位置差，跳过")
                continue
            
            # 6. 输出结果到控制台
            print("\n" + "=" * 50)
            print("测量结果:")
            print(f"峰与调制边带位置差的平均值: {avg_diff}")
            print(f"峰与调制边带位置差的标准差: {std_diff}")
            print(f"峰与调制边带位置差的统计信息:{summary}")
            # 记录结果
            result_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'modulation_value_mhz': modulation_value,
                'filename': os.path.basename(csv_file),
                'average_difference': avg_diff,
                'standard_deviation': std_diff
            }
            results.append(result_entry)
        
        if not results:
            print("没有成功分析的文件")
            return
        
        # 7. 将结果输出到新的CSV文件
        output_csv = f"sideband_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        save_results_to_csv(results, output_csv)
        
        print(f"\n所有结果已保存到 {output_csv}")
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scope_acq.disconnect()

if __name__ == "__main__":
    # 检查必要的库
    try:
        import numpy as np
        import pandas as pd
        import scipy
        import matplotlib
    except ImportError as e:
        print(f"错误: 缺少必要的库。请安装: pip install numpy pandas scipy matplotlib")
        sys.exit(1)
    
    main()