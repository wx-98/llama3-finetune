# LLaMA3 医学领域微调

这个项目使用 PubMedQA 数据集对 LLaMA3 模型进行医学领域的微调，使用 LoRA 技术进行高效微调。

## 项目结构

```
.
├── data/                   # 数据集目录
├── models/                 # 模型目录
├── dataset_utils.py       # 数据集处理工具
├── model_utils.py         # 模型加载工具
├── train.py              # 训练脚本
└── requirements.txt      # 项目依赖
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (用于GPU训练)

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/your-username/llama3-medical-finetuning.git
cd llama3-medical-finetuning
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 下载数据集：
```bash
python dataset_utils.py
```

2. 开始训练：
```bash
python train.py
```

## 训练参数

- 使用 LoRA 进行高效微调
- 训练轮数：3 epochs
- 批次大小：2
- 梯度累积步数：8
- 学习率：1e-4
- 使用 FP16 混合精度训练

## 注意事项

- 确保有足够的 GPU 显存（建议至少 16GB）
- 训练过程中会自动保存检查点
- 使用 wandb 进行训练监控

## 许可证

MIT License 