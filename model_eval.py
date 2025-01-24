import os
import json
from eval_utils import calculate_metrics
from useful_tools import extract_different_ans_case

def process_and_save_metrics(input_file_path, suffix=""):
    # 读取 JSON 格式的输入文件
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"read error: {e}")
        return
    # data = extract_different_ans_case(data)
    # 调用 calculate_metrics 函数处理数据
    average_pre_metrics, average_post_metrics = calculate_metrics(data)
    
    # 从文件路径中提取文件名（去掉扩展名）
    base_name = os.path.basename(input_file_path)
    file_name_without_extension = os.path.splitext(base_name)[0]
    
    # 构造输出文件路径
    output_dir = './results/InstructEdit/eval_result'
    os.makedirs(output_dir, exist_ok=True)
    
    # 构造最终的文件路径
    output_file_path = os.path.join(output_dir, f"{file_name_without_extension}{suffix}.json")
    
    # 准备要保存的结果数据
    result = {
        "average_pre_metrics": average_pre_metrics,
        "average_post_metrics": average_post_metrics
    }
    
    # 写入结果到新的 JSON 文件
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"save result to: {output_file_path}")
    except Exception as e:
        print(f"save error: {e}")




if __name__ == "__main__":
    input_file_path = "/data0/liuyuhuan/liuyuhuan/repo/EasyEdit/logs/llama3.1-instruct_MEND_cf_whole_loc.json"  # 替换成你的输入文件路径
    process_and_save_metrics(input_file_path,suffix='')