import os
import torch
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import wandb
from tqdm import tqdm
from model_utils import load_local_model
from dataset_utils import load_local_dataset  # 导入本地数据集加载函数

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
def preprocess_function(examples, tokenizer):
    # 构建提示模板
    prompts = []
    for question, answer in zip(examples["question"], examples["long_answer"]):
        prompt = f"问题：{question}\n回答：{answer}"
        prompts.append(prompt)
    
    # 对文本进行编码
    encodings = tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    
    # 确保所有必要的字段都存在
    if "labels" not in encodings:
        encodings["labels"] = encodings["input_ids"].clone()
    
    return encodings

def main():
    # 初始化wandb
    wandb.init(project="llama3-medical-finetuning")
    
    # 加载本地模型和分词器
    print("正在加载本地模型和分词器...")
    model, tokenizer = load_local_model(model_dir="./models")
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=32,  # 增加rank以提高模型容量
        lora_alpha=64,  # 增加alpha值
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 更新目标模块
        lora_dropout=0.1,  # 增加dropout以防止过拟合
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 应用LoRA
    print("正在应用LoRA配置...")
    model = get_peft_model(model, lora_config)
    
    # 加载本地数据集
    print("正在加载本地数据集...")
    dataset = load_local_dataset(data_dir="data")
    
    # 数据预处理
    print("正在进行数据预处理...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./llama3-medical-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # 减小批次大小以适应更大的模型
        gradient_accumulation_steps=8,  # 增加梯度累积步数
        learning_rate=1e-4,  # 降低学习率
        fp16=True,
        logging_steps=10,
        save_strategy="steps",  # 改为steps以匹配评估策略
        save_steps=100,  # 每100步保存一次
        eval_steps=100,  # 每100步评估一次
        eval_strategy="steps",  # 使用步数作为评估策略
        load_best_model_at_end=True,
        report_to="wandb",
        warmup_steps=100,  # 添加预热步数
        weight_decay=0.01,  # 添加权重衰减
        max_grad_norm=1.0,  # 添加梯度裁剪
        save_total_limit=3,  # 最多保存3个检查点
        remove_unused_columns=False,  # 防止删除未使用的列
        push_to_hub=False,  # 不上传到Hub
        ddp_find_unused_parameters=False,  # 分布式训练参数
        dataloader_num_workers=4,  # 数据加载器的工作进程数
        dataloader_pin_memory=True,  # 使用固定内存
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # 添加填充到8的倍数
    )
    
    # 初始化训练器
    print("正在初始化训练器...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],  # 使用分割出的验证集
        data_collator=data_collator
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存模型
    print("正在保存模型...")
    trainer.save_model("./llama3-medical-finetuned")
    tokenizer.save_pretrained("./llama3-medical-finetuned")
    
    # 关闭wandb
    wandb.finish()
    print("训练完成！")

if __name__ == "__main__":
    main() 