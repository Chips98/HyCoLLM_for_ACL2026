#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HCN认知状态分类任务微调脚本
基于 unsloth 和 Qwen2.5-7B-Instruct 模型
专注于四维认知状态（情感、思维、意图、立场）的识别
"""

import argparse
import os
import warnings
import json
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# 设置环境变量屏蔽警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainingArguments
)

# 设置环境变量禁用 Unsloth 的网络请求
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["UNSLOTH_DISABLE_STATS"] = "1"

# 尝试导入 Unsloth
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
    print("Unsloth 可用，将使用 FastLanguageModel 进行训练。")
except ImportError:
    HAS_UNSLOTH = False
    print("将使用标准的 Hugging Face Transformers 加载模型。")

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
# 强制禁用 Unsloth 的 SFTTrainer，使用标准版本
import sys
if HAS_UNSLOTH:
    sys.modules.pop('unsloth.trainer', None)

from trl import SFTTrainer, SFTConfig

# 导入统一的提示词管理
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prompts import get_prompts

# ==================== 核心数据处理函数 ====================

def build_cognitive_dataset(data_path: str, tokenizer=None, debug: bool = False) -> Dataset:
    """
    构建认知状态分类任务数据集

    Args:
        data_path: 数据文件路径
        tokenizer: 分词器
        debug: 是否打印调试信息

    Returns:
        认知状态训练数据集
    """
    print("正在构建认知状态分类任务数据集...")

    # 读取原始数据
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = []
        for line in f:
            raw_data.append(json.loads(line))

    print(f"原始数据包含 {len(raw_data)} 个样本")

    cognitive_samples = []
    processed_count = 0

    for entry_idx, entry in enumerate(tqdm(raw_data, desc="处理数据")):
        messages = entry.get('messages', [])

        # 验证消息格式
        if len(messages) < 3:
            continue

        # 提取system、user、assistant消息
        system_msg = None
        user_msg = None
        assistant_msg = None

        for msg in messages:
            if msg.get('role') == 'system':
                system_msg = msg
            elif msg.get('role') == 'user':
                user_msg = msg
            elif msg.get('role') == 'assistant':
                assistant_msg = msg

        if not (system_msg and user_msg and assistant_msg):
            continue

        # 验证助手输出格式 - 只接受新格式
        assistant_content = assistant_msg.get('content', '')

        # 新格式：文本标记
        required_markers = ['<<<EMOTION>>>', '<<<THINKING>>>', '<<<INTENT>>>', '<<<STANCE>>>']

        # 检查是否包含所有必需的新格式标记
        if not all(marker in assistant_content for marker in required_markers):
            if debug and processed_count < 5:
                print(f"跳过格式不正确的样本 {entry_idx}: {assistant_content}")
                print(f"缺少的标记: {[m for m in required_markers if m not in assistant_content]}")
            continue

        # 构建样本
        cognitive_samples.append({
            "messages": messages,
            "task_type": "cognitive_analysis",
            "sample_id": entry_idx
        })

        processed_count += 1

        # 调试打印
        if debug and processed_count <= 3:
            print(f"\n=== 样本 {processed_count} ===")
            print(f"用户输入: {user_msg['content'][:200]}...")
            print(f"助手输出: {assistant_msg}")
            print("="*50)

    print(f"认知数据集构建完成！")
    print(f" - 处理的样本数量: {processed_count}")

    if processed_count == 0:
        print("⚠️ 警告: 没有找到任何有效样本！")

    return Dataset.from_list(cognitive_samples)


def format_chat_messages(example: Dict[str, Any], tokenizer: AutoTokenizer = None) -> str:
    """
    将Chat messages转换为训练文本
    使用聊天模板自动处理角色和内容
    """
    try:
        messages = example.get("messages", [])
        if not messages:
            return ""

        # 使用聊天模板格式化
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        return formatted_text

    except Exception as e:
        print(f"格式化消息时出错: {e}")
        return ""


# ==================== 主训练脚本 ====================

def main():
    parser = argparse.ArgumentParser(description="HCN认知状态分类任务微调脚本")

    # --- 基本配置参数 ---
    parser.add_argument(
        "--model_name_or_path", type=str,
        default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        help="基础模型路径"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./cognitive_lora",
        help="模型输出目录"
    )
    parser.add_argument(
        "--dataset_path", type=str,
        default="../data/processed/train.jsonl",
        help="训练数据路径"
    )
    parser.add_argument(
        "--val_dataset_path", type=str,
        default="../data/processed/val.jsonl",
        help="验证数据路径"
    )
    parser.add_argument(
        "--labels_path", type=str,
        default=None,
        help="标签配置文件路径"
    )

    # --- 调试参数 ---
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--debug_samples", type=int, default=5, help="调试样本数量")

    # --- 训练参数 ---
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="训练批次大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="评估批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--max_steps", type=int, default=-1, help="最大训练步数")
    parser.add_argument("--save_steps", type=int, default=50, help="保存步数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--eval_steps", type=int, default=50, help="评估步数")
    parser.add_argument("--save_total_limit", type=int, default=3, help="保存的检查点数量限制")

    # --- LoRA 参数 ---
    parser.add_argument("--use_lora", action="store_true", default=True, help="使用 LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--use_qlora", action="store_true", default=True, help="使用 QLoRA")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="4位量化")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="梯度检查点")
    parser.add_argument("--disable_unsloth", action="store_true", help="禁用 Unsloth")

    # --- 从检查点恢复训练参数 ---
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从检查点恢复训练")

    args = parser.parse_args()

    # 验证路径
    if not os.path.exists(args.dataset_path):
        print(f"错误: 训练数据文件不存在: {args.dataset_path}")
        return

    # 强制禁用 Unsloth
    if args.disable_unsloth:
        HAS_UNSLOTH = False
        print("强制禁用 Unsloth")
        sys.modules.pop('unsloth.trainer', None)

    # --- 生成带时间戳的输出目录名 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 转换相对路径为绝对路径，确保相对于HCN项目根目录
    if os.path.isabs(args.output_dir):
        prefixed_output_dir = f"{args.output_dir}_{timestamp}"
    else:
        # 获取HCN项目根目录（当前脚本所在目录的上级目录）
        hcn_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 直接在HCN根目录下的checkpoints目录创建模型文件夹
        model_name = os.path.basename(args.output_dir)  # 提取模型名称 'sft_lora'
        checkpoints_dir = os.path.join(hcn_root, "checkpoints")  # HCN根目录/checkpoints
        prefixed_output_dir = os.path.join(checkpoints_dir, f"{model_name}_{timestamp}")  # HCN根目录/checkpoints/sft_lora_20251201_09074

    print(f"模型路径: {args.model_name_or_path}")
    print(f"训练数据: {args.dataset_path}")
    print(f"输出目录: {prefixed_output_dir}")

    # GPU设置
    if torch.cuda.is_available():
        print(f"检测到GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("警告: 未检测到可用的GPU设备")
        return

    # --- 加载tokenizer ---
    print("正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )

    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"设置 pad_token: {tokenizer.pad_token}")

    # 注意：不再添加特殊tokens，使用纯文本标记 <<<EMOTION>>> 等

    # --- 构建数据集 ---
    print("正在构建认知状态分类数据集...")
    train_dataset = build_cognitive_dataset(args.dataset_path, tokenizer, debug=args.debug)

    # 加载验证数据集（如果存在）
    val_dataset = None
    if os.path.exists(args.val_dataset_path):
        val_dataset = build_cognitive_dataset(args.val_dataset_path, tokenizer, debug=False)
        print(f"验证集大小: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("错误: 训练数据集为空，无法开始训练")
        return

    # 格式化数据集
    print("格式化Chat数据...")

    def format_dataset(example):
        formatted_text = format_chat_messages(example, tokenizer)
        return {"text": formatted_text}

    # 应用格式化
    train_dataset = train_dataset.map(format_dataset, remove_columns=train_dataset.column_names)
    if val_dataset:
        val_dataset = val_dataset.map(format_dataset, remove_columns=val_dataset.column_names)

    # 检查格式化结果
    if len(train_dataset) > 0:
        sample_text = train_dataset[0]["text"]
        print(f"格式化样本长度: {len(sample_text)}")
        if args.debug:
            print(f"格式化样本前500字符: {sample_text[:500]}...")

    # --- 加载模型 ---
    print("正在加载模型...")

    if HAS_UNSLOTH:
        # 使用 Unsloth 加载模型
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=args.max_length,
            dtype=None,
            load_in_4bit=args.load_in_4bit,
        )

        # 注意：不再需要调整词汇表大小，因为使用现有tokens

        # 配置 LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
        )

        print("Unsloth 模型和 LoRA 配置完成")
        model.print_trainable_parameters()

    else:
        # 使用标准 Transformers
        bnb_config = None
        if args.use_qlora and args.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            print("启用4位量化")

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # 注意：不再需要调整词汇表大小，因为使用现有tokens

        # 准备LoRA训练
        if args.use_lora:
            if args.use_qlora:
                model = prepare_model_for_kbit_training(
                    model,
                    use_gradient_checkpointing=args.gradient_checkpointing
                )

            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

    # --- 配置训练参数 ---
    training_arguments_dict = {
        "output_dir": prefixed_output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "logging_steps": args.logging_steps,
        "logging_strategy": "steps",
        "logging_first_step": True,
        "eval_strategy": "steps" if val_dataset else "no",
        "eval_steps": args.eval_steps if val_dataset else None,
        "metric_for_best_model": "eval_loss" if val_dataset else None,
        "greater_is_better": False if val_dataset else None,
        "load_best_model_at_end": True if val_dataset else False,
        "report_to": "none",
        # 强制使用fp16，避免与unsloth的bf16配置冲突
        "bf16": False,
        "fp16": True,
        "max_grad_norm": 0.3,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "dataloader_pin_memory": False,
        "dataloader_num_workers": 0,
        "remove_unused_columns": True,
        "dataset_text_field": "text",
        "packing": False,
        "gradient_checkpointing": args.gradient_checkpointing,
        "resume_from_checkpoint": args.resume_from_checkpoint,
    }

    training_arguments = SFTConfig(**training_arguments_dict)

    # --- 初始化训练器 ---
    print("初始化 SFTTrainer...")

    trainer_kwargs = {
        "model": model,
        "args": training_arguments,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
    }

    if val_dataset:
        trainer_kwargs["eval_dataset"] = val_dataset

    trainer = SFTTrainer(**trainer_kwargs)

    # 添加回调
    if val_dataset:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    print("SFTTrainer 初始化完成")

    # --- 开始训练 ---
    print("开始认知状态分类任务训练...")
    try:
        trainer.train()
        print("训练完成！")

        # --- 保存模型 ---
        print(f"保存模型到: {prefixed_output_dir}")
        os.makedirs(prefixed_output_dir, exist_ok=True)

        # 保存LoRA权重和tokenizer
        trainer.save_model(prefixed_output_dir)
        tokenizer.save_pretrained(prefixed_output_dir)

        # 验证保存结果
        print("验证模型保存结果...")
        saved_files = os.listdir(prefixed_output_dir)
        print(f"保存的文件: {saved_files}")

        # 检查关键文件
        key_files = ["adapter_model.safetensors", "adapter_config.json", "tokenizer.json"]
        missing_files = [f for f in key_files if f not in saved_files]
        if missing_files:
            print(f"警告: 缺少关键文件: {missing_files}")
        else:
            print("✓ 所有关键文件已保存")

        print(f"认知状态微调完成！模型保存至: {prefixed_output_dir}")

        # 保存推理示例
        inference_example_path = os.path.join(prefixed_output_dir, "inference_example.txt")
        with open(inference_example_path, 'w', encoding='utf-8') as f:
            f.write("# HCN认知状态分析推理示例\n\n")
            f.write("## 系统提示词:\n")

            # 获取统一的提示词管理
            prompts = get_prompts(args.labels_path)
            f.write(prompts.get_system_prompt() + "\n\n")

            f.write("## 推理格式:\n")
            f.write("- System: 认知状态分析任务定义\n")
            f.write("- User: [背景]上下文\n[评论]目标评论\n[指令]分析请求\n")
            f.write("- Assistant: <<<EMOTION>>>情感\n<<<THINKING>>>思维\n<<<INTENT>>>意图\n<<<STANCE>>>立场\n\n")

    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()