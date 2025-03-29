import pickle
import argparse
import os
import numpy as np
from collections import Counter

def load_pickle_file(file_path):
    """加载 PKL 文件并返回其内容"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None

def analyze_data_structure(data):
    """分析数据结构并提供基本信息"""
    if isinstance(data, list):
        print(f"数据是一个列表，包含 {len(data)} 个元素")
        if data:
            print(f"第一个元素的类型: {type(data[0])}")
            if isinstance(data[0], dict):
                print("字典键: ", list(data[0].keys()))
    elif isinstance(data, dict):
        print(f"数据是一个字典，包含 {len(data)} 个键")
        print("字典键: ", list(data.keys()))
    else:
        print(f"数据类型: {type(data)}")

def main():
    parser = argparse.ArgumentParser(description="读取并分析 PKL 文件")
    parser.add_argument("file_path", help="PKL 文件的路径")
    parser.add_argument("--sample", type=int, default=1, help="显示的样本数量")
    parser.add_argument("--key", type=str, help="要检查的特定键（如果数据是字典列表）")
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"文件不存在: {args.file_path}")
        return
    
    print(f"加载文件: {args.file_path}")
    data = load_pickle_file(args.file_path)
    
    if data is None:
        return
    
    print("\n===== 数据结构分析 =====")
    analyze_data_structure(data)
    
    print("\n===== 样本数据 =====")
    if isinstance(data, list):
        samples = min(args.sample, len(data))
        for i in range(samples):
            print(f"\n样本 {i+1}:")
            if args.key and isinstance(data[i], dict) and args.key in data[i]:
                sample_data = data[i][args.key]
                print(f"{args.key}: {sample_data}")
                if isinstance(sample_data, np.ndarray):
                    print(f"形状: {sample_data.shape}, 类型: {sample_data.dtype}")
                    print(f"最小值: {np.min(sample_data)}, 最大值: {np.max(sample_data)}")
            else:
                print(data[i])
    else:
        print(data)
    
    # 如果是字典列表，统计每个样本的键
    if isinstance(data, list) and data and isinstance(data[0], dict):
        all_keys = Counter()
        for item in data:
            all_keys.update(item.keys())
        
        print("\n===== 键分布统计 =====")
        for key, count in all_keys.most_common():
            print(f"{key}: 出现 {count} 次 ({count/len(data)*100:.1f}%)")

if __name__ == "__main__":
    main()
    # python temp.py ./data/cad_data/train_deduplicate_se.pkl