import os
import json
import requests
from datasets import load_dataset, Dataset
from tqdm import tqdm
import pandas as pd

def download_pubmedqa():
    """
    下载PubMedQA数据集并保存到本地
    """
    print("开始下载PubMedQA数据集...")
    
    # 创建数据目录
    os.makedirs("data", exist_ok=True)
    
    # 使用Hugging Face datasets库下载数据集
    dataset = load_dataset("pubmed_qa", "pqa_labeled")
    
    # 只保存训练集
    df = dataset["train"].to_pandas()
    output_path = "data/pubmedqa_train.csv"
    df.to_csv(output_path, index=False)
    print(f"已保存训练集到: {output_path}")
    
    print("数据集下载完成！")
    return dataset

def load_local_dataset(data_dir="data"):
    """
    从本地加载数据集
    
    Args:
        data_dir: 数据目录路径
    
    Returns:
        dict: 包含训练集的字典
    """
    print("从本地加载数据集...")
    
    # 检查数据文件是否存在
    if not os.path.exists(os.path.join(data_dir, "pubmedqa_train.csv")):
        raise FileNotFoundError("找不到数据文件: pubmedqa_train.csv，请先运行download_pubmedqa()下载数据集")
    
    # 加载训练集
    file_path = os.path.join(data_dir, "pubmedqa_train.csv")
    df = pd.read_csv(file_path)
    dataset = Dataset.from_pandas(df)
    
    # 将数据集分割为训练集和验证集
    train_val = dataset.train_test_split(test_size=0.1, seed=42)
    
    print("数据集加载完成！")
    return train_val

def get_dataset_stats(dataset):
    """
    获取数据集的统计信息
    
    Args:
        dataset: 数据集字典
    
    Returns:
        dict: 包含数据集统计信息的字典
    """
    stats = {}
    for split, data in dataset.items():
        stats[split] = {
            "样本数量": len(data),
            "平均问题长度": sum(len(q) for q in data["question"]) / len(data),
            "平均答案长度": sum(len(a) for a in data["long_answer"]) / len(data)
        }
    return stats

def main():
    # 下载数据集
    dataset = download_pubmedqa()
    
    # 获取数据集统计信息
    stats = get_dataset_stats(dataset)
    
    # 打印统计信息
    print("\n数据集统计信息:")
    for split, stat in stats.items():
        print(f"\n{split}集:")
        for key, value in stat.items():
            print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    main() 