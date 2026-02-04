import os
import csv
import shutil
from pathlib import Path

def process_csv_folder_remove_empty_rows(input_folder, output_folder=None, delete_original=True):
    """
    处理文件夹中所有CSV文件，删除空行后保存到新文件夹，并可选择删除原文件
    
    参数:
        input_folder: 输入CSV文件夹路径
        output_folder: 输出文件夹路径（默认在输入文件夹同级创建_cleaned文件夹）
        delete_original: 是否删除原文件（默认True）
    
    返回:
        dict: 处理结果的统计信息
    """
    # 确保输入文件夹存在
    input_path = Path(input_folder)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {input_folder}")
    
    # 设置输出文件夹路径
    if output_folder is None:
        output_folder = input_path.parent / f"{input_path.name}_cleaned"
    
    output_path = Path(output_folder)
    
    # 创建输出文件夹
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    stats = {
        'total_files': 0,
        'processed_files': 0,
        'empty_rows_removed': 0,
        'failed_files': [],
        'processed_file_details': []
    }
    
    # 获取所有CSV文件
    csv_files = list(input_path.glob("*.csv"))
    stats['total_files'] = len(csv_files)
    
    if not csv_files:
        print(f"在文件夹 {input_folder} 中没有找到CSV文件")
        return stats
    
    """ print(f"找到 {len(csv_files)} 个CSV文件，开始处理...") """
    
    # 处理每个CSV文件
    for csv_file in csv_files:
        try:
            """ print(f"处理文件: {csv_file.name}") """
            
            # 输出文件路径
            output_file = output_path / csv_file.name
            
            # 处理CSV文件（删除空行）
            empty_rows_removed = process_single_csv_file(csv_file, output_file)
            
            # 记录处理详情
            file_stats = {
                'filename': csv_file.name,
                'original_path': str(csv_file),
                'processed_path': str(output_file),
                'empty_rows_removed': empty_rows_removed
            }
            stats['processed_file_details'].append(file_stats)
            stats['empty_rows_removed'] += empty_rows_removed
            stats['processed_files'] += 1
            
            """ print(f"  成功处理，删除了 {empty_rows_removed} 个空行") """
            
        except Exception as e:
            error_msg = f"处理文件 {csv_file.name} 时出错: {str(e)}"
            print(f"  {error_msg}")
            stats['failed_files'].append({
                'filename': csv_file.name,
                'error': str(e)
            })
    
    # 如果要求删除原文件且所有文件都处理成功
    if delete_original and not stats['failed_files']:
        """ print(f"\n正在删除原文件夹中的CSV文件...") """
        for csv_file in csv_files:
            try:
                csv_file.unlink()
                """ print(f"  已删除: {csv_file.name}") """
            except Exception as e:
                print(f"  删除文件 {csv_file.name} 失败: {e}")
    
    # 打印汇总信息
    """ print(f"\n处理完成！")
    print(f"总文件数: {stats['total_files']}")
    print(f"成功处理: {stats['processed_files']}")
    print(f"失败文件: {len(stats['failed_files'])}")
    print(f"总共删除空行: {stats['empty_rows_removed']}")
    print(f"处理后的文件保存在: {output_path}") """
    
    if delete_original and not stats['failed_files']:
        print(f"原文件夹中的CSV文件已删除")
    
    return stats

def process_single_csv_file(input_file, output_file):
    """
    处理单个CSV文件，删除空行
    
    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
    
    返回:
        int: 删除的空行数
    """
    empty_rows_count = 0
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            # 检查行是否为空（所有字段都为空或只包含空格）
            if any(field.strip() for field in row):
                writer.writerow(row)
            else:
                empty_rows_count += 1
    
    return empty_rows_count

# 使用示例
if __name__ == "__main__":
    # 测试
    process_csv_folder_remove_empty_rows("testdata","processedtestdata")