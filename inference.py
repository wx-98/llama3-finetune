import torch
from peft import PeftModel
from model_utils import load_local_model

def load_finetuned_model(model_path="./llama3-medical-finetuned"):
    """
    加载微调后的模型
    
    Args:
        model_path: 微调模型保存路径
    
    Returns:
        model: 加载的模型
        tokenizer: 加载的分词器
    """
    print("正在加载基础模型和分词器...")
    # 加载基础模型和分词器
    base_model, tokenizer = load_local_model(model_dir="./models")
    
    print("正在加载LoRA权重...")
    # 加载LoRA权重
    model = PeftModel.from_pretrained(base_model, model_path)
    return model, tokenizer

def generate_response(model, tokenizer, question, max_length=512):
    # 构建提示
    prompt = f"问题：{question}\n回答："
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,  # 添加重复惩罚
            length_penalty=1.0,  # 添加长度惩罚
            no_repeat_ngram_size=3  # 避免重复的n-gram
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取回答部分
    response = response.split("回答：")[-1].strip()
    return response

def main():
    try:
        # 加载模型和分词器
        print("正在初始化模型...")
        model_path = "./llama3-medical-finetuned"
        model, tokenizer = load_finetuned_model(model_path)
        
        # 交互式问答
        print("\n欢迎使用医学问答系统！输入'退出'结束对话。")
        while True:
            question = input("\n请输入您的问题：")
            if question.lower() in ['退出', 'quit', 'exit']:
                break
                
            print("正在生成回答...")
            response = generate_response(model, tokenizer, question)
            print(f"\n回答：{response}")
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
        print("请确保：")
        print("1. 已经运行过 model_utils.py 下载模型")
        print("2. 已经完成模型训练")
        print("3. 模型文件路径正确")

if __name__ == "__main__":
    main() 