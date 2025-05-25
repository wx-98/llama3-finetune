import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login, HfFolder
import getpass

def login_to_huggingface():
    """
    登录到Hugging Face
    
    Returns:
        bool: 登录是否成功
    """
    try:
        # 检查是否已经登录
        if HfFolder.get_token() is not None:
            print("已经登录到Hugging Face")
            return True
            
        # 获取用户输入的token
        print("请输入您的Hugging Face token (https://huggingface.co/settings/tokens):")
        token = getpass.getpass()
        
        # 登录
        login(token=token)
        print("成功登录到Hugging Face！")
        return True
        
    except Exception as e:
        print(f"登录失败: {str(e)}")
        return False

def verify_huggingface_access(model_name="meta-llama/Llama-3.1-8B"):
    """
    验证是否有权限访问指定的模型
    
    Args:
        model_name: 要验证的模型名称
    
    Returns:
        bool: 是否有权限访问
    """
    try:
        # 尝试获取模型信息
        AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        print(f"成功验证对模型 {model_name} 的访问权限")
        return True
    except Exception as e:
        print(f"验证失败: {str(e)}")
        print("请确保：")
        print("1. 您已经登录到Hugging Face")
        print("2. 您有权限访问该模型")
        print("3. 您已经接受了模型的使用条款")
        return False

def download_and_save_model(model_name="meta-llama/Llama-3.1-8B", save_dir="./models", cache_dir=r"F:\code\huggingface"):
    """
    下载模型和分词器并保存到本地
    
    Args:
        model_name: Hugging Face模型名称
        save_dir: 保存目录
        cache_dir: 缓存目录
    """
    # 首先登录并验证
    if not login_to_huggingface():
        raise Exception("登录失败，无法继续")
    
    if not verify_huggingface_access(model_name):
        raise Exception("没有访问权限，无法继续")
    
    print(f"开始下载模型 {model_name}...")
    
    # 创建保存目录和缓存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # 下载并保存分词器
    print("下载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=True,
        cache_dir=cache_dir
    )
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
    print(f"分词器已保存到: {os.path.join(save_dir, 'tokenizer')}")
    
    # 下载并保存模型
    print("下载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_auth_token=True,
        cache_dir=cache_dir
    )
    model.save_pretrained(os.path.join(save_dir, "model"))
    print(f"模型已保存到: {os.path.join(save_dir, 'model')}")
    
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
    model_path = os.path.join(model_dir, "model")
    tokenizer_path = os.path.join(model_dir, "tokenizer")
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            "找不到模型文件，请先运行 download_and_save_model() 下载模型"
        )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True
    )
    
    print("模型和分词器加载完成！")
    return model, tokenizer

if __name__ == "__main__":
    # 下载并保存模型
    model, tokenizer = download_and_save_model() 