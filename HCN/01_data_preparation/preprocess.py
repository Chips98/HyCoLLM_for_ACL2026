#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本
将原始的JSON数据转换为ChatML格式，用于LLM微调
"""

import json
import os
import random
from typing import List, Dict, Any
from tqdm import tqdm

# 导入统一的提示词管理
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prompts import get_prompts

# 获取统一的提示词和标签
prompts = get_prompts()


def load_raw_data(data_path: str) -> List[Dict]:
    """加载原始数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def build_system_prompt() -> str:
    """构建系统提示词"""
    return prompts.get_system_prompt()


def build_user_prompt(context_post: str, target_post: str) -> str:
    """构建用户提示词"""
    return f"[背景]{context_post}\n[评论]{target_post}\n[指令]请分析该评论的认知状态。"


def build_assistant_response(labels: Dict[str, str]) -> str:
    """构建助手响应"""
    return prompts.build_assistant_response(
        emotion=labels.get('emotion', ''),
        thinking_value=labels.get('thinking_value', ''),
        intent=labels.get('intent', ''),
        stance=labels.get('stance', '')
    )


def convert_to_chatml(data: List[Dict]) -> List[Dict]:
    """转换为ChatML格式"""
    chatml_data = []

    for item in tqdm(data, desc="转换数据"):
        # 验证必要字段
        if not all(key in item for key in ['context_post', 'target_post', 'cognitive_labels']):
            continue

        # 构建对话
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(item['context_post'], item['target_post'])
        assistant_response = build_assistant_response(item['cognitive_labels'])

        # 构建ChatML消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]

        chatml_data.append({
            "messages": messages,
            "source": "us_china_tariff_war",
            "conversation_id": item.get('conversation_id', -1),
            "user_id": item.get('user_id', -1)
        })

    return chatml_data


def save_jsonl(data: List[Dict], output_path: str):
    """保存为JSONL格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def split_dataset(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> tuple:
    """划分数据集，确保同一用户的数据不分散到不同集合"""
    # 按用户ID分组
    user_data = {}
    for item in data:
        user_id = item.get('user_id', -1)
        if user_id not in user_data:
            user_data[user_id] = []
        user_data[user_id].append(item)

    # 随机打乱用户列表
    user_ids = list(user_data.keys())
    random.shuffle(user_ids)

    # 计算每个集合的用户数
    n_users = len(user_ids)
    n_train = int(n_users * train_ratio)
    n_val = int(n_users * (train_ratio + val_ratio))

    # 划分用户
    train_users = user_ids[:n_train]
    val_users = user_ids[n_train:n_val]
    test_users = user_ids[n_val:]

    # 构建数据集
    train_data = []
    val_data = []
    test_data = []

    for user_id in train_users:
        train_data.extend(user_data[user_id])

    for user_id in val_users:
        val_data.extend(user_data[user_id])

    for user_id in test_users:
        test_data.extend(user_data[user_id])

    print(f"数据集划分完成:")
    print(f"  训练集: {len(train_data)} 条, {len(train_users)} 个用户")
    print(f"  验证集: {len(val_data)} 条, {len(val_users)} 个用户")
    print(f"  测试集: {len(test_data)} 条, {len(test_users)} 个用户")

    return train_data, val_data, test_data


def main():
    # 设置随机种子
    random.seed(42)

    # 路径配置
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(data_dir, 'data', 'raw', 'us_china_tariff_war_data_cn.json')
    processed_dir = os.path.join(data_dir, 'data', 'processed')

    # 创建输出目录
    os.makedirs(processed_dir, exist_ok=True)

    # 加载原始数据
    print(f"加载原始数据: {raw_data_path}")
    raw_data = load_raw_data(raw_data_path)
    print(f"原始数据量: {len(raw_data)} 条")

    # 转换为ChatML格式
    chatml_data = convert_to_chatml(raw_data)
    print(f"转换后数据量: {len(chatml_data)} 条")

    # 划分数据集
    train_data, val_data, test_data = split_dataset(chatml_data)

    # 保存数据集
    save_jsonl(train_data, os.path.join(processed_dir, 'train.jsonl'))
    save_jsonl(val_data, os.path.join(processed_dir, 'val.jsonl'))
    save_jsonl(test_data, os.path.join(processed_dir, 'test.jsonl'))

    print("\n数据预处理完成！")
    print(f"输出文件保存在: {processed_dir}")


if __name__ == "__main__":
    main()