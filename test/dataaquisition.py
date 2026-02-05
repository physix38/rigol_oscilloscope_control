import pyvisa
import numpy as np
import pandas as pd
from datetime import datetime
import time
import os

def bytes_to_hex_array(data: bytes) -> list:
    """
    将相邻两个字节组合为一个16位整数（小端序），返回整数列表。
    如果字节数为奇数，最后一个字节将被忽略。
    """
    result = []
    # 每次取两个字节，步长为2
    for i in range(0, len(data) - 1, 2):
        # 将两个字节组合为16位整数（低字节在前）
        value = data[i] | (data[i + 1] << 8)
        result.append(value)
    return result

class RigolMHO984_DataAcquisition:
    def __init__(self, visa_address='USB0::0x1AB1::0x0452::MHO9B280400571::INSTR'):
        """
        初始化示波器连接
        参数:
            visa_address: 示波器的VISA地址
                        可以通过 pyvisa.ResourceManager().list_resources() 查看可用设备
        """
        self.rm = pyvisa.ResourceManager()
        self.scope = None
        self.visa_address = visa_address
        
    def connect(self):
        """连接示波器"""
        try:
            self.scope = self.rm.open_resource(self.visa_address)
            # 设置超时时间为10秒
            self.scope.timeout = 10000
            # 清除错误队列
            self.scope.write('*CLS')
            
            # 查询示波器ID以验证连接
            idn = self.scope.query('*IDN?')
            print(f"已连接到示波器: {idn}")
            
            # 设置数据格式为ASCII（更易处理）
            """ self.scope.write('COMM_HEADER OFF')
            self.scope.write('COMM_FORMAT ASCII') """
            
            return True
        except Exception as e:
            print(f"连接示波器失败: {e}")
            return False
    
    def setup_acquisition(self):
        """设置采集参数（使用当前触发和采集方式）"""
        try:
            # 获取当前设置（不改变示波器当前配置）
            """ print("使用示波器当前触发和采集设置...") """
            
            # 获取当前时间基准设置
            timebase = self.scope.query('TIMebase:MAIN:SCALe?')
            print(f"当前时基: {timebase} s/div")
            
            # 获取通道设置
            for ch in [1, 2]:
                scale = self.scope.query(f'CHANnel{ch}:SCALe?')
                offset = self.scope.query(f'CHANnel{ch}:OFFSet?')
                coupling = self.scope.query(f'CHANnel{ch}:COUPling?')
                print(f"通道{ch}: 垂直刻度={scale} V/div, 偏移={offset} V, 耦合={coupling}") 
            
            # 获取触发设置
            trigger_source = self.scope.query('TRIGger:EDGE:SOURce?')
            trigger_level = self.scope.query('TRIGger:EDGE:LEVel?')
            print(f"触发源: {trigger_source}, 触发电平: {trigger_level} V")
             
            return True
        except Exception as e:
            print(f"设置采集参数失败: {e}")
            return False
    
    def acquire_channel_data(self, channel):
        """采集指定通道的数据"""
        try:
            # 选择通道
            """ self.scope.write('CHDR OFF')  # 关闭头部信息 """ 
            self.scope.write(f'WAV:SOUR CHAN{channel}')   
            self.scope.write(f'WAV:MODE RAW')
            # 设置波形参数为ASCII格式
            """ self.scope.write('WFSU SP,1,NP,0,FP,0')  # 设置波形参数 """
            """ resolution_type=self.scope.query('ACQ:TYPE? ') """
            # 获取波形数据
            self.scope.write('WAVeform:FORMat WORD')
            self.scope.write('WAV:DATA?')  # 请求波形数据
            
            
            # 读取波形数据
            raw_data = self.scope.read_raw()
            
          
            yor=self.scope.query('WAV:YOR?')
            yor=np.int64(yor)
            yinc=self.scope.query('WAV:YINC?')  
            print(type(yinc))
            yinc=np.float64(yinc)
            # 解析波形数据
            # 跳过头部信息（找到#字符）
            start_idx = raw_data.find(b'#')
            if start_idx == -1:
                raise ValueError("未找到波形数据头部")
            
            # 解析数据长度信息
            header_length = int(chr(raw_data[start_idx + 1]))
            data_length = int(raw_data[start_idx + 2:start_idx + 2 + header_length])
            
            # 提取实际数据
            data_start = start_idx + 2 + header_length
            data_bytes = raw_data[data_start:data_start + data_length]
            data_bytes = bytes_to_hex_array(data_bytes)
            data_bytes =np.array(data_bytes)
            
            voltage_data = (data_bytes-data_bytes[0]+yor)*yinc
           
            
            # 获取波形参数以进行缩放
            vdiv = float(self.scope.query(f'CHAN{channel}:SCALe?'))
            offset = float(self.scope.query(f'CHAN{channel}:OFFSET?'))
            probe = float(self.scope.query(f'CHAN{channel}:PROB? '))
       
            # 获取时间参数 
            tdiv = float(self.scope.query('TIMebase:MAIN:SCAL? '))
            delay = float(self.scope.query('TIMebase:DEL:OFFSET?'))
            sample_rate = float(self.scope.query('ACQ:SRAT?'))
            """  width= self.scope.query('WAV:WIDT?')
            print(width) """
            
            
            

          
            
            
            # 计算时间轴
            num_points = len(voltage_data)
            time_data = np.linspace(delay - 5 * tdiv, delay + 5 * tdiv, num_points)
            
            # 获取当前时间戳
            acquisition_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            return {
                'time': time_data,
                'voltage': voltage_data,
                'acquisition_time': acquisition_time,
                'channel': channel,
                'vdiv': vdiv,
                'offset': offset,
                'tdiv': tdiv,
                'sample_rate': sample_rate,
                'num_points': num_points
            }
            
        except Exception as e:
            print(f"采集通道{channel}数据失败: {e}")
            return None
    
    def save_to_csv(self, data, save_dir='./testdata',filename=None):
        """保存数据到CSV文件"""


        if data is None:
            return False
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        
        if filename is None:
            # 生成默认文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'Scope_Data_Ch{data["channel"]}_{timestamp}.csv'
            filepath = os.path.join(save_dir, filename)
        
        try:
            # 创建DataFrame
            df = pd.DataFrame({
                'Time (s)': data['time'],
                'Voltage (V)': data['voltage']
            })
            
            # 添加元数据作为注释
            metadata = f"""# Siglent SDS3104X HD Oscilloscope Data
# Acquisition Time: {data['acquisition_time']}
# Channel: {data['channel']}
# Vertical Scale: {data['vdiv']} V/div
# Vertical Offset: {data['offset']} V
# Timebase Scale: {data['tdiv']} s/div
# Sample Rate: {data['sample_rate']:.2f} Sa/s
# Number of Points: {data['num_points']}
# Trigger Mode: Current Setting (Not Modified)
# Acquisition Mode: Current Setting (Not Modified)
"""
            
            # 写入文件
            with open(filepath, 'w') as f:
                f.write(metadata)
                df.to_csv(f, index=False)
            
            """ print(f"数据已保存到: {filepath}")
            print(f"文件大小: {os.path.getsize(filepath)} 字节") """
            return True
            
        except Exception as e:
            print(f"保存CSV文件失败: {e}")
            return False
    
    def single_acquisition(self):
        """执行单次采集"""
        try:
            # 确保示波器处于运行状态
            self.scope.write('TRIGger:MODE EDGE')  # 设置为自动触发模式
            time.sleep(0.1)  # 等待示波器稳定
            
            # 执行单次触发
            self.scope.write('SINGLE')
            """ print("等待触发...") """
            
            # 等待采集完成
            time.sleep(2)
            
            # 检查触发状态
            trigger_status = self.scope.query('TRIG:STAT?')
            """ print(f"触发状态: {trigger_status}") """
            
            return True
            
        except Exception as e:
            print(f"单次采集失败: {e}")
            return False
    
    def acquire_and_save_both_channels(self):
        """采集并保存两个通道的数据"""
        if not self.scope:
            print("示波器未连接")
            return False
        
        """ print("\n开始采集数据...") """
        
        # 执行单次采集
        if not self.single_acquisition():
            return False
        
        # 采集通道1数据
        """ print("\n采集通道1数据...") """
        ch1_data = self.acquire_channel_data(1)
        if ch1_data:
            self.save_to_csv(ch1_data,'./testdata')
        
        # 采集通道2数据
        """ print("\n采集通道2数据...") """
        ch2_data = self.acquire_channel_data(2)
        if ch2_data:
            self.save_to_csv(ch2_data,'./testdata')
        
        return ch1_data is not None and ch2_data is not None
    
    def disconnect(self):
        """断开连接"""
        if self.scope:
            self.scope.close()
            print("已断开示波器连接")

def main():
    """主函数"""
    print("Rigol MHO984 示波器数据采集程序")
    print("=" * 50)
    
    # 创建采集器实例
    # 注意：需要根据实际情况修改VISA地址
    # 可以通过以下代码查找设备：
    # rm = pyvisa.ResourceManager()
    # print(rm.list_resources())
    scope_acq = RigolMHO984_DataAcquisition(
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
        success = scope_acq.acquire_and_save_both_channels()
        
        if success:
            print("\n" + "=" * 50)
            print("数据采集完成！")
            print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("\n数据采集失败")
            
    except KeyboardInterrupt:
        print("\n用户中断采集")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
    finally:
        # 断开连接
        scope_acq.disconnect()

if __name__ == "__main__":
    # 安装必要的库：
    # pip install pyvisa numpy pandas pyvisa-py
    
    # 注意：需要安装VISA库
    # 1. 安装NI-VISA (National Instruments) 或
    # 2. 使用 pyvisa-py (纯Python实现): pip install pyvisa-py
    
    main()