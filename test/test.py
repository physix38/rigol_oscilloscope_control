""" import pyvisa """
import numpy as np
import pandas as pd
from datetime import datetime

from pathlib import Path
import dataaquisition

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, peak_prominences

import sys
import csvprocess
import time
import os
import csv
import shutil
import argparse


def main():
    """主函数"""
    print("Rigol MHO984 示波器数据采集程序")
    print("=" * 50)
    
    # 创建采集器实例
    # 注意：需要根据实际情况修改VISA地址
    # 可以通过以下代码查找设备：
    # rm = pyvisa.ResourceManager()
    # print(rm.list_resources())
    scope_acq = dataaquisition.RigolMHO984_DataAcquisition(
        visa_address='USB0::0x1AB1::0x0452::MHO9B280400571::INSTR'  # 修改为你的设备地址
    )
    
    try:
        # 连接示波器
        if not scope_acq.connect():
            print("请检查以下可能的问题：")
            print("1. 示波器是否已通过USB连接")
            print("2. 是否安装了正确的VISA驱动")
            print("3. VISA地址是否正确")
            return
        
        # 设置采集参数
        scope_acq.setup_acquisition()
        
        # 采集并保存数据
        for i in range (50):
            success = scope_acq.acquire_and_save_both_channels()
        
            if success:
                print("\n" + "=" * 50)
                print("数据采集完成！")
                print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                scope_acq.scope.write('RUN')

            else:
                print("\n数据采集失败")

            time.sleep(120)

        
    except KeyboardInterrupt:
        print("\n用户中断采集")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
    finally:
        # 断开连接
        scope_acq.disconnect()
    csvprocess.process_csv_folder_remove_empty_rows("testdata","processedtestdata1")

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