# HCN (Hyper-Cognition Net) 项目

基于LLM微调和双曲空间的社交媒体评论四维认知状态分析系统

## 项目概述

HCN是一个先进的认知状态分析系统，能够从社交媒体评论中识别用户的四维认知状态：
- **情感 (Emotion)**：评论者的情绪状态
- **思维 (Thinking)**：评论者的思维模式
- **意图 (Intent)**：评论者想要达成的目的
- **立场 (Stance)**：评论者对特定话题的立场

### 技术特点

1. **双层架构**：
   - LLM语义层：使用LoRA微调的Qwen2.5-7B作为特征提取器
   - 双曲空间层：利用庞加莱球进行特征解缠，拉开易混淆标签的距离

2. **跨维度融合**：使用Transformer捕获四个认知维度间的依赖关系

3. **端到端推理**：支持实时推理和批量处理

## 项目结构

```
HCN_Project/
├── data/                           # 数据目录
│   ├── raw/                        # 原始数据
│   ├── processed/                  # 处理后的数据
│   └── embeddings/                 # LLM提取的特征
├── 01_data_preparation/            # 模块1：数据准备
├── 02_llm_sft/                     # 模块2：LLM微调
├── 03_feature_extraction/          # 模块3：特征提取
├── 04_hcn_training/                # 模块4：HCN网络训练
│   └── models/                     # 模型定义
├── 05_inference/                   # 模块5：推理部署
├── utils/                          # 工具模块
├── checkpoints/                    # 模型检查点
├── requirements.txt                # 依赖包列表
└── run_pipeline.sh                 # 完整运行脚本
```

## 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <repository_url>
cd HCN/HCN_Project

# 激活conda环境
source ~/.bashrc && conda activate oasis

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行完整流程

```bash
# 一键运行（包含所有步骤）
./run_pipeline.sh
```

### 3. 单独运行各模块

#### 数据准备
```bash
cd 01_data_preparation
python preprocess.py  # 数据格式转换
python data_split.py  # 数据集划分
```

#### LLM微调（可选）
```bash
cd 02_llm_sft
python sft_cognitive.py \
    --model_name_or_path "unsloth/Qwen2.5-7B-Instruct-bnb-4bit" \
    --dataset_path "../data/processed/train.jsonl"
```

#### 特征提取
```bash
cd 03_feature_extraction
python extract_embeddings.py \
    --model_path "path/to/llm" \
    --lora_path "path/to/lora" \
    --data_path "../data/processed/train.jsonl"
```

#### HCN训练
```bash
cd 04_hcn_training
python train_hcn.py \
    --train_path "../data/embeddings/train_embeddings.pt" \
    --val_path "../data/embeddings/val_embeddings.pt"
```

### 4. 推理使用

#### 命令行工具
```bash
cd 05_inference
python demo_cli.py \
    --llm_path "path/to/llm" \
    --hcn_path "path/to/hcn/model"
```

#### API服务
```bash
cd 05_inference
python api_server.py \
    --llm_path "path/to/llm" \
    --hcn_path "path/to/hcn/model" \
    --port 5000
```

#### Python API
```python
from pipeline import HCNInferencePipeline

# 创建推理流水线
pipeline = HCNInferencePipeline(
    llm_base_path="path/to/llm",
    lora_path="path/to/lora",
    hcn_model_path="path/to/hcn/model"
)

# 单条预测
result = pipeline.predict(
    context="背景上下文",
    target="目标评论"
)
print(result)

# 批量预测
data = [
    {"context": "...", "target": "..."},
    {"context": "...", "target": "..."}
]
results = pipeline.predict_batch(data)
```

## 标签体系

### 情感维度 (10个标签)
- 平静、厌恶、信任、喜悦、惊讶、愤怒、期待、其他、悲伤、恐惧

### 思维维度 (2个标签)
- 直觉型：基于直觉和感觉
- 分析型：基于逻辑和分析

### 意图维度 (9个标签)
- 表达主张、信息分享、分歧与冲突、情感表达、寻求信息、认同与联结、号召行动、意图不明、情绪化判断

### 立场维度 (6个标签)
- 不明确、反对美方、支持美方、支持中方、反对贸易战/呼吁合作、反对中方

## 配置说明

所有配置都可以通过命令行参数或配置文件进行调整。主要配置项：

- `model.hidden_dim`: 隐藏层维度（默认512）
- `model.n_layers`: Transformer层数（默认4）
- `training.epochs`: 训练轮数（默认50）
- `training.lr`: 学习率（默认1e-3）
- `loss.lambda_hyper`: 双曲空间正则化权重（默认0.1）

## 性能指标

在测试集上的预期性能：
- 单维度准确率：>85%
- 四维同时准确率：>60%
- 推理速度：<100ms/样本
- 显存占用：<8GB（推理）

## 故障排除

### 常见问题

1. **显存不足**
   - 减小batch_size
   - 使用梯度累积
   - 启用混合精度训练

2. **模型加载失败**
   - 检查模型路径是否正确
   - 确保有足够的磁盘空间
   - 检查网络连接（如果从云端加载）

3. **训练不收敛**
   - 调整学习率
   - 增加训练轮数
   - 检查数据预处理是否正确

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 联系方式

如有问题，请通过GitHub Issues联系。