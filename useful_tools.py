import os
import json
import random

def split_dataset(file_path, train_ratio=0.8):
    """
    根据给定比例切分数据集，并保存为训练集和测试集文件。

    Args:
        file_path (str): 输入的 JSON 数据文件路径。
        train_ratio (float): 训练集占比，默认 0.8（80%）。
    
    Returns:
        None
    """
    try:
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("输入文件内容必须是一个 JSON 列表！")
        
        # 随机打乱数据
        random.shuffle(data)
        
        # 切分数据
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # 从文件路径中提取文件名（去掉扩展名）
        base_name = os.path.basename(file_path)
        file_name_without_extension = os.path.splitext(base_name)[0]
        
        # 构造保存目录
        output_dir = 'data/mquake'
        os.makedirs(output_dir, exist_ok=True)
        
        # 构造训练集和测试集的文件路径
        train_file_path = os.path.join(output_dir, f"{file_name_without_extension}_train.json")
        test_file_path = os.path.join(output_dir, f"{file_name_without_extension}_test.json")
        
        # 保存训练集和测试集
        with open(train_file_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)
        
        with open(test_file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
        
        print(f"训练集已保存到: {train_file_path}")
        print(f"测试集已保存到: {test_file_path}")
    
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

def extract_different_ans_case(list):

    case_list=[]
    for data in list:
        if data['requested_rewrite']['target_new'] in data['requested_rewrite']['portability']['neighborhood']['ground_truth']:
            pass                       
        else:
            case_list.append(data)
    print('case:',len(case_list))
    return case_list


# 主程序入口
if __name__ == "__main__":
    # 示例输入
    input_file_path = "data/mquake/MQuAKE-CF-3k-v2.json"  # 输入文件路径
    train_ratio = 0.8  # 训练集比例
    split_dataset(input_file_path, train_ratio)
