import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from modelscope import snapshot_download

def download_and_save_model(model_name="qwen/Qwen-7B", save_dir="./models"):
    """
    使用 ModelScope 下载模型和分词器并保存到本地
    
    Args:
        model_name: ModelScope 模型名称
        save_dir: 保存目录
    """
    print(f"开始下载模型 {model_name}...")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 使用 ModelScope 下载模型
    print("下载模型...")
    model_dir = snapshot_download(model_name, cache_dir=save_dir)
    print(f"模型已下载到: {model_dir}")
    
    # 加载分词器
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("模型和分词器加载完成！")
    return model, tokenizer

def load_local_model(model_dir="./models", device_map="auto"):
    """
    从本地加载模型和分词器
    
    Args:
        model_dir: 模型目录
        device_map: 设备映射
    
    Returns:
        model: 加载的模型
        tokenizer: 加载的分词器
    """
    print("从本地加载模型和分词器...")
    
    # 检查文件是否存在
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            "找不到模型文件，请先运行 download_and_save_model() 下载模型"
        )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True
    )
    
    print("模型和分词器加载完成！")
    return model, tokenizer

if __name__ == "__main__":
    # 下载并保存模型
    model, tokenizer = download_and_save_model() 