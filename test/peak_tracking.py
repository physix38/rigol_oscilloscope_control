
"""
自动化峰值跟踪程序
实现：
1. 设定采集次数和间隔时间
2. 循环采集数据（使用 Rigol 示波器）
3. 对每次采集的数据进行预处理（删除 CSV 空行）
4. 寻峰并记录突出度高于阈值的峰
5. 与上一次采集的峰进行匹配，计算水平位置移动
6. 输出峰位移数据到 CSV 文件并绘图

依赖：
- dataaquisition.py (示波器采集)
- csvprocess.py (CSV 处理)

- peak2.py (寻峰)

运行实例：uv run test\peak_tracking.py --interval 20 --max_displacement 5 --iteration 2 --plot
"""
import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from pylab import mpl
from matplotlib import font_manager



try:
    import dataaquisition
    import csvprocess
    import peak2
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保 dataaquisition.py, csvprocess.py, peak2.py 位于 test/ 目录下")
    sys.exit(1)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='自动化峰值跟踪程序')
    parser.add_argument('--iterations', type=int, default=5,
                        help='采集次数 (默认: 5)')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='每次采集之间的间隔时间（秒）(默认: 2.0)')
    parser.add_argument('--prominence_threshold', type=float, default=0.1,
                        help='峰突出度阈值，高于此值的峰才被记录 (默认: 0.1)')
    parser.add_argument('--max_displacement', type=float, default=5,
                        help='匹配峰时允许的最大水平位移（单位同 x 轴）(默认: 0.5)')
    parser.add_argument('--output_dir', type=str, default='./peak_tracking_results',
                        help='输出结果目录 (默认: ./peak_tracking_results)')
    parser.add_argument('--visa_address', type=str, 
                        default='USB0::0x1AB1::0x0452::MHO9B280400571::INSTR',
                        help='示波器 VISA 地址 (默认: USB0::0x1AB1::0x0452::MHO9B280400571::INSTR)')
    parser.add_argument('--channel', type=int, default=1,
                        help='采集通道 (默认: 1)')
    parser.add_argument('--skip_processing', action='store_true',
                        help='跳过 CSV 处理步骤（直接使用原始文件）')
    parser.add_argument('--plot', action='store_true',
                        help='生成位移图表')
    return parser.parse_args()

def match_peaks(prev_peaks, curr_peaks, max_distance):
    """
    匹配两次采集之间的峰
    参数:
        prev_peaks: DataFrame，包含前一次采集的峰，必须有 'x_position*10000' 列
        curr_peaks: DataFrame，当前采集的峰
        max_distance: 最大匹配距离
    返回:
        matches: 列表，元素为 (prev_index, curr_index, distance)
        new_peaks: 当前采集中新出现的峰的索引列表
        lost_peaks: 前一次采集中消失的峰的索引列表
    """
    if prev_peaks.empty:
        return [], list(curr_peaks.index), []
    if curr_peaks.empty:
        return [], [], list(prev_peaks.index)
    
    prev_positions = prev_peaks['x_position*10000'].values
    curr_positions = curr_peaks['x_position*10000'].values
    
    # 简单最近邻匹配
    matches = []
    used_prev = set()
    used_curr = set()
    
    for i, curr_pos in enumerate(curr_positions):
        distances = np.abs(prev_positions - curr_pos)
        if len(distances) == 0:
            continue
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        if min_dist <= max_distance and min_idx not in used_prev:
            matches.append((min_idx, i, min_dist))
            used_prev.add(min_idx)
            used_curr.add(i)
    
    new_peaks = [i for i in range(len(curr_positions)) if i not in used_curr]
    lost_peaks = [i for i in range(len(prev_positions)) if i not in used_prev]
    
    return matches, new_peaks, lost_peaks

def acquire_and_process(scope_acq, channel, output_dir, iteration, skip_processing=False):
    """
    执行单次采集、保存、处理，返回 CSV 文件路径、数据、原始文件路径和处理后文件路径
    返回: (csv_file, data, raw_path, processed_path)
    """
    # 生成文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    raw_filename = f'Scope_Data_Ch{channel}_{timestamp}_iter{iteration}.csv'
    raw_path = os.path.join(output_dir, 'raw', raw_filename)
    
    # 确保目录存在
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'processed'), exist_ok=True)
    
    # 执行单次采集
    print(f"  第 {iteration} 次采集...")
    scope_acq.scope.write('TRIGger:MODE EDGE')
    time.sleep(0.1)
    scope_acq.scope.write('SINGLE')
    time.sleep(2)  # 等待采集完成
    
    # 采集指定通道数据
    data = scope_acq.acquire_channel_data(channel)
    if data is None:
        print(f"  采集失败")
        return None, None, None, None
    
    # 保存原始数据
    success = scope_acq.save_to_csv(data, save_dir=os.path.join(output_dir, 'raw'), filename=raw_filename)
    if not success:
        print(f"  保存原始数据失败")
        return None, None, None, None
    
    # 处理 CSV（删除空行）
    processed_filename = f'processed_{raw_filename}'
    processed_path = os.path.join(output_dir, 'processed', processed_filename)
    if not skip_processing:
        try:
            empty_rows = csvprocess.process_single_csv_file(raw_path, processed_path)
            print(f"  处理完成，删除了 {empty_rows} 个空行")
            # 使用处理后的文件进行寻峰
            csv_file = processed_path
        except Exception as e:
            print(f"  处理 CSV 文件时出错: {e}，将使用原始文件")
            csv_file = raw_path
    else:
        csv_file = raw_path
        processed_path = None  # 未生成处理后文件
    
    return csv_file, data, raw_path, processed_path

def find_peaks_above_threshold(csv_file, prominence_threshold, distance=1000):
    """
    寻峰并返回突出度高于阈值的峰 DataFrame
    """
    peaks_df = peak2.find_peaks_in_csv(
        csv_file,
        x_col=0,
        y_col=1,
        distance=distance,
        prominence=prominence_threshold ,  # 寻峰时使用较低阈值以确保检测到
        plot=False
    )
    if peaks_df.empty:
        return peaks_df
    # 筛选突出度高于阈值的峰
    peaks_df = peaks_df[peaks_df['prominence'] >= prominence_threshold]
    return peaks_df

def main():
    args = parse_args()
    
    print("=" * 60)
    print("自动化峰值跟踪程序启动")
    print(f"采集次数: {args.iterations}")
    print(f"间隔时间: {args.interval} 秒")
    print(f"突出度阈值: {args.prominence_threshold}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化示波器采集器
    print("初始化示波器连接...")
    scope_acq = dataaquisition.RigolMHO984_DataAcquisition(visa_address=args.visa_address)
    if not scope_acq.connect():
        print("连接示波器失败，请检查 VISA 地址和连接")
        sys.exit(1)
    scope_acq.setup_acquisition()
    
    # 存储每次采集的结果
    all_peaks = []          # 每次采集的峰 DataFrame 列表
    all_matches = []        # 每次匹配结果
    csv_files = []          # CSV 文件路径
    acquisition_times = []  # 采集时间戳
    
    prev_peaks = pd.DataFrame()
    
    try:
        for i in range(1, args.iterations + 1):
            print(f"\n--- 第 {i}/{args.iterations} 次采集 ---")
            start_time = time.time()
            
            # 采集与处理
            csv_file, data, raw_path, processed_path = acquire_and_process(
                scope_acq, args.channel, args.output_dir, i, args.skip_processing
            )
            if csv_file is None:
                print("  采集失败，跳过此次迭代")
                continue
            
            csv_files.append(csv_file)
            if data:
                acquisition_times.append(data.get('acquisition_time', datetime.now().isoformat()))
            
            # 寻峰
            peaks_df = find_peaks_above_threshold(csv_file, args.prominence_threshold)
            print(f"  检测到 {len(peaks_df)} 个峰（突出度 >= {args.prominence_threshold}）")
            if not peaks_df.empty:
                print(f"  峰位置: {peaks_df['x_position*10000'].values}")
            
            # 删除原始和处理后的 CSV 文件以节省空间
            try:
                if raw_path and os.path.exists(raw_path):
                    os.remove(raw_path)
                    print(f"  已删除原始文件: {raw_path}")
                if processed_path and os.path.exists(processed_path):
                    os.remove(processed_path)
                    print(f"  已删除处理后文件: {processed_path}")
            except Exception as e:
                print(f"  删除文件时出错: {e}")
            
            all_peaks.append(peaks_df)
            
            # 匹配峰
            matches, new_peaks, lost_peaks = match_peaks(
                prev_peaks, peaks_df, args.max_displacement
            )
            print(f"  匹配到 {len(matches)} 个峰，新增 {len(new_peaks)} 个，消失 {len(lost_peaks)} 个")
            
            # 计算位移
            displacements = []
            for prev_idx, curr_idx, dist in matches:
                prev_pos = prev_peaks.iloc[prev_idx]['x_position*10000']
                curr_pos = peaks_df.iloc[curr_idx]['x_position*10000']
                if (curr_idx%2==0):
                    displacement = (curr_pos - prev_pos)
                else:
                    displacement = (-curr_pos + prev_pos)
                
                displacements.append({
                    'iteration': i,
                    'prev_idx': prev_idx,
                    'curr_idx': curr_idx,
                    'prev_position': prev_pos,
                    'curr_position': curr_pos,
                    'displacement': displacement,
                    'distance': dist,
                    'prominence': peaks_df.iloc[curr_idx]['prominence']
                })
            
            # 计算平均位移
            if displacements:
                avg_displacement = np.mean([d['displacement'] for d in displacements])
            else:
                avg_displacement = 0.0  # 无匹配峰时位移为零
            
            all_matches.append({
                'iteration': i,
                'matches': matches,
                'new_peaks': new_peaks,
                'lost_peaks': lost_peaks,
                'displacements': displacements,
                'average_displacement': avg_displacement
            })
            
            prev_peaks = peaks_df
            
            # 实时更新累积位移图（如果启用绘图）
            if args.plot:
                plot_cumulative_live(args.output_dir, all_matches, acquisition_times)
            
            # 等待间隔时间
            elapsed = time.time() - start_time
            if elapsed < args.interval:
                time.sleep(args.interval - elapsed)
    
    except KeyboardInterrupt:
        print("\n用户中断采集")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 断开示波器连接
        
        scope_acq.disconnect()
        
        # 保存结果
        save_results(args.output_dir, all_peaks, all_matches, acquisition_times, csv_files)
        
        # 绘图
        if args.plot:
            plot_results(args.output_dir, all_matches, all_peaks, acquisition_times)
    
    print("\n程序完成")

def save_results(output_dir, all_peaks, all_matches, acquisition_times, csv_files):
    """保存结果到 CSV 文件"""
    import pandas as pd
    
    # 1. 峰位移数据
    rows = []
    for match_info in all_matches:
        for disp in match_info['displacements']:
            rows.append(disp)
    if rows:
        disp_df = pd.DataFrame(rows)
        disp_path = os.path.join(output_dir, 'peak_displacements.csv')
        disp_df.to_csv(disp_path, index=False, encoding='utf-8-sig')
        print(f"峰位移数据已保存到: {disp_path}")
    
    # 2. 峰出现/消失记录
    peak_events = []
    for match_info in all_matches:
        i = match_info['iteration']
        for idx in match_info['new_peaks']:
            peak_events.append({
                'iteration': i,
                'event': 'new',
                'peak_idx': idx,
                'position': all_peaks[i-1].iloc[idx]['x_position*10000'] if i-1 < len(all_peaks) else None
            })
        for idx in match_info['lost_peaks']:
            peak_events.append({
                'iteration': i,
                'event': 'lost',
                'peak_idx': idx,
                'position': all_peaks[i-2].iloc[idx]['x_position*10000'] if i-2 >= 0 else None
            })
    if peak_events:
        event_df = pd.DataFrame(peak_events)
        event_path = os.path.join(output_dir, 'peak_events.csv')
        event_df.to_csv(event_path, index=False, encoding='utf-8-sig')
        print(f"峰事件记录已保存到: {event_path}")
    
    # 3. 每次采集的峰列表
    peak_summary = []
    for i, peaks_df in enumerate(all_peaks):
        for idx, row in peaks_df.iterrows():
            peak_summary.append({
                'iteration': i+1,
                'peak_idx': idx,
                'x_position': row['x_position*10000'],
                'y_height': row['y_height'],
                'prominence': row['prominence'],
                'fwhm': row['fwhm']
            })
    if peak_summary:
        summary_df = pd.DataFrame(peak_summary)
        summary_path = os.path.join(output_dir, 'peak_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"峰摘要已保存到: {summary_path}")
    
    # 4. 保存采集日志
    log_df = pd.DataFrame({
        'iteration': range(1, len(csv_files)+1),
        'csv_file': csv_files,
        'acquisition_time': acquisition_times[:len(csv_files)]
    })
    log_path = os.path.join(output_dir, 'acquisition_log.csv')
    log_df.to_csv(log_path, index=False, encoding='utf-8-sig')
    print(f"采集日志已保存到: {log_path}")
    
    # 5. 平均位移与累积位移数据
    import numpy as np
    avg_data = []
    for match_info in all_matches:
        iteration = match_info['iteration']
        avg_disp = match_info.get('average_displacement', 0.0)
        avg_data.append({
            'iteration': iteration,
            'average_displacement': avg_disp
        })
    
    if avg_data:
        avg_df = pd.DataFrame(avg_data)
        # 按迭代次数排序
        avg_df = avg_df.sort_values('iteration')
        # 计算累积位移
        avg_df['cumulative_displacement'] = np.cumsum(avg_df['average_displacement'])
        # 添加采集时间戳（如果可用）
        if acquisition_times is not None:
            # 创建迭代次数到时间的映射
            time_map = {}
            for i, t in enumerate(acquisition_times):
                time_map[i+1] = t  # 迭代次数从1开始
            avg_df['acquisition_time'] = avg_df['iteration'].map(time_map)
        # 保存
        avg_path = os.path.join(output_dir, 'average_displacement.csv')
        avg_df.to_csv(avg_path, index=False, encoding='utf-8-sig')
        print(f"平均位移数据已保存到: {avg_path}")
    else:
        print("无平均位移数据，跳过保存")

def plot_cumulative_live(output_dir, all_matches, acquisition_times=None):
    """
    实时更新累积位移图（每次采集后调用）
    参数:
        output_dir: 输出目录
        all_matches: 匹配信息列表（包含已进行的所有迭代）
        acquisition_times: 采集时间戳列表（可选）
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    from pylab import mpl
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams['axes.unicode_minus']=False #用来正常显示符号
    # 提取每次采集的平均位移
    avg_displacements = []
    iterations = []
    for match_info in all_matches:
        avg_displacements.append(match_info.get('average_displacement', 0.0))
        iterations.append(match_info['iteration'])
    
    if len(avg_displacements) == 0:
        return  # 无数据时不绘图
    
    # 计算累积位移（累加和）
    cumulative_displacements = np.cumsum(avg_displacements)
    
    # 确定X轴：时间或迭代次数
    if acquisition_times is not None and len(acquisition_times) >= len(iterations):
        from datetime import datetime
        try:
            times = [datetime.fromisoformat(t) for t in acquisition_times[:len(iterations)]]
            base_time = times[0]
            x_values = [(t - base_time).total_seconds() for t in times]
            x_label = '时间 (秒)'
        except Exception as e:
            print(f"  时间戳解析失败，改用迭代次数: {e}")
            x_values = iterations
            x_label = '采集次数'
    else:
        x_values = iterations
        x_label = '采集次数'
    
    # 创建双Y轴图：平均位移（柱状图）和累积位移（折线图）
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 平均位移柱状图
    bars = ax1.bar(x_values, avg_displacements, width=0.6, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('平均位移量', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('平均位移与累积总位移随时间变化（实时）')
    
    # 在每个柱子上标注数值
    for bar in bars:
        height = bar.get_height()
        if height != 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 累积位移折线图（次Y轴）
    ax2 = ax1.twinx()
    ax2.plot(x_values, cumulative_displacements, 'ro-', linewidth=2, markersize=6, label='累积总位移')
    ax2.set_ylabel('累积总位移量', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper left')
    
    ax1.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    cumulative_path = os.path.join(output_dir, 'cumulative_displacement_vs_time.png')
    plt.savefig(cumulative_path, dpi=150)
    plt.close()
    print(f"  实时累积位移图已更新: {cumulative_path}")
    
    # 单独绘制累积位移折线图（可选）
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, cumulative_displacements, 'ro-', linewidth=2, markersize=6)
    plt.xlabel(x_label)
    plt.ylabel('累积总位移量')
    plt.title('累积总位移随时间变化（实时）')
    plt.grid(True, alpha=0.3)
    cum_only_path = os.path.join(output_dir, 'cumulative_displacement_only.png')
    plt.savefig(cum_only_path, dpi=150)
    plt.close()
    print(f"  实时累积位移单独图已更新: {cum_only_path}")


def plot_results(output_dir, all_matches, all_peaks, acquisition_times=None):
    """绘制位移图表
    参数:
        output_dir: 输出目录
        all_matches: 匹配信息列表
        all_peaks: 峰数据列表
        acquisition_times: 采集时间戳列表（可选）
    """
    import matplotlib.pyplot as plt
    from pylab import mpl
    # 设置显示中文字体
    mpl.rcParams["font.sans-serif"] = ["SimHei"]

    plt.rcParams['axes.unicode_minus']=False #用来正常显示符号
    # 1. 位移随时间变化折线图
    plt.figure(figsize=(12, 6))
    
    # 收集每个峰的位移序列（按峰 ID 跟踪）
    # 由于峰 ID 可能变化，我们使用匹配关系构建轨迹
    # 简化：仅绘制匹配峰的位移
    trajectories = {}  # peak_track_id -> {iteration: displacement}
    for match_info in all_matches:
        i = match_info['iteration']
        for disp in match_info['displacements']:
            prev_idx = disp['prev_idx']
            curr_idx = disp['curr_idx']
            # 这里简化处理：使用 prev_idx 作为轨迹 ID（假设峰 ID 稳定）
            track_id = prev_idx
            if track_id not in trajectories:
                trajectories[track_id] = {}
            trajectories[track_id][i-1] = disp['prev_position']  # 前一次位置
            trajectories[track_id][i] = disp['curr_position']   # 当前位置
    
    for track_id, pos_dict in trajectories.items():
        iterations = sorted(pos_dict.keys())
        positions = [pos_dict[it] for it in iterations]
        plt.plot(iterations, positions, 'o-', label=f'峰 {track_id}', markersize=4)
    
    plt.xlabel('采集次数')
    plt.ylabel('峰位置 (x * 10000)')
    plt.title('峰位置随时间变化')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'peak_positions_vs_iteration.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"峰位置变化图已保存到: {plot_path}")
    
    # 2. 位移量分布直方图
    displacements = []
    for match_info in all_matches:
        for disp in match_info['displacements']:
            displacements.append(disp['displacement'])
    
    if displacements:
        plt.figure(figsize=(10, 5))
        plt.hist(displacements, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('位移量')
        plt.ylabel('频次')
        plt.title('峰位移分布直方图')
        plt.grid(True, alpha=0.3)
        hist_path = os.path.join(output_dir, 'displacement_histogram.png')
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"位移分布直方图已保存到: {hist_path}")
    
    # 3. 峰数量随时间变化
    peak_counts = [len(peaks_df) for peaks_df in all_peaks]
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(peak_counts)+1), peak_counts, 'bo-', markersize=6)
    plt.xlabel('采集次数')
    plt.ylabel('峰数量')
    plt.title('每次采集检测到的峰数量')
    plt.grid(True, alpha=0.3)
    count_path = os.path.join(output_dir, 'peak_count_vs_iteration.png')
    plt.savefig(count_path, dpi=150)
    plt.close()
    print(f"峰数量变化图已保存到: {count_path}")
    
    # 4. 平均位移与累积总位移随时间变化
    import numpy as np
    
    # 提取每次采集的平均位移
    avg_displacements = []
    iterations = []
    for match_info in all_matches:
        # 注意：all_matches 中可能包含第一次采集（无前次匹配），此时平均位移为0
        avg_displacements.append(match_info.get('average_displacement', 0.0))
        iterations.append(match_info['iteration'])
    
    # 确保顺序正确（按迭代次数排序）
    if len(avg_displacements) == 0:
        print("  无位移数据，跳过平均位移绘图")
    else:
        # 计算累积位移（累加和）
        cumulative_displacements = np.cumsum(avg_displacements)
        
        # 确定X轴：时间或迭代次数
        if acquisition_times is not None and len(acquisition_times) >= len(iterations):
            # 使用采集时间戳作为X轴
            # 将时间字符串转换为 datetime 对象，然后计算相对于第一次采集的秒数
            from datetime import datetime
            try:
                times = [datetime.fromisoformat(t) for t in acquisition_times[:len(iterations)]]
                # 计算相对于第一次采集的秒数
                base_time = times[0]
                x_values = [(t - base_time).total_seconds() for t in times]
                x_label = '时间 (秒)'
            except Exception as e:
                print(f"  时间戳解析失败，改用迭代次数: {e}")
                x_values = iterations
                x_label = '采集次数'
        else:
            x_values = iterations
            x_label = '采集次数'
        
        # 创建双Y轴图：平均位移（柱状图）和累积位移（折线图）
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 平均位移柱状图
        bars = ax1.bar(x_values, avg_displacements, width=0.6, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('平均位移量', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title('平均位移与累积总位移随时间变化')
        
        # 在每个柱子上标注数值
        for bar in bars:
            height = bar.get_height()
            if height != 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 累积位移折线图（次Y轴）
        ax2 = ax1.twinx()
        ax2.plot(x_values, cumulative_displacements, 'ro-', linewidth=2, markersize=6, label='累积总位移')
        ax2.set_ylabel('累积总位移量', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 添加图例
        ax2.legend(loc='upper left')
        
        # 网格
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        cumulative_path = os.path.join(output_dir, 'cumulative_displacement_vs_time.png')
        plt.savefig(cumulative_path, dpi=150)
        plt.close()
        print(f"累积总位移图已保存到: {cumulative_path}")
        
        # 单独绘制累积位移折线图（可选）
        plt.figure(figsize=(10, 5))
        plt.plot(x_values, cumulative_displacements, 'ro-', linewidth=2, markersize=6)
        plt.xlabel(x_label)
        plt.ylabel('累积总位移量')
        plt.title('累积总位移随时间变化')
        plt.grid(True, alpha=0.3)
        cum_only_path = os.path.join(output_dir, 'cumulative_displacement_only.png')
        plt.savefig(cum_only_path, dpi=150)
        plt.close()
        print(f"累积总位移单独图已保存到: {cum_only_path}")

if __name__ == '__main__':
    main()