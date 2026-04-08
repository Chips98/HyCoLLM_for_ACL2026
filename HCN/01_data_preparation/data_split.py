#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集划分脚本
支持多种划分策略和类别平衡处理
"""

import json
import os
import random
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any
from sklearn.model_selection import StratifiedShuffleSplit


def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """保存为JSONL格式"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def extract_labels(data: List[Dict]) -> List[str]:
    """提取主要标签用于分层抽样"""
    # 使用情感标签作为分层依据
    labels = []
    for item in data:
        # 从assistant消息中提取情感标签
        assistant_msg = item['messages'][-1]['content']
        for line in assistant_msg.split('\n'):
            if line.startswith('[EMO]'):
                labels.append(line[5:])  # 去掉[EMO]前缀
                break
    return labels


def split_by_user(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
                  seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """按用户ID划分数据集，确保同一用户的数据不分散"""
    random.seed(seed)

    # 按用户分组
    user_data = defaultdict(list)
    for item in data:
        user_id = item.get('user_id', -1)
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

    return train_data, val_data, test_data


def split_stratified(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
                     seed: str = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """分层抽样划分数据集，保持标签分布一致"""
    random.seed(seed)
    np.random.seed(seed)

    # 提取标签
    labels = extract_labels(data)

    # 计算划分索引
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))

    # 使用分层抽样
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(sss.split(data, labels))

    # 进一步划分训练和验证集
    train_val_labels = [labels[i] for i in train_val_idx]
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio/(train_ratio+val_ratio), random_state=seed)
    train_idx, val_idx = next(sss_val.split([data[i] for i in train_val_idx], train_val_labels))

    # 构建最终的数据集
    train_data = [data[train_val_idx[i]] for i in train_idx]
    val_data = [data[train_val_idx[i]] for i in val_idx]
    test_data = [data[i] for i in test_idx]

    return train_data, val_data, test_data


def analyze_dataset(data: List[Dict], name: str = "Dataset"):
    """分析数据集统计信息"""
    print(f"\n{name} 统计信息:")
    print(f"  总样本数: {len(data)}")

    # 统计用户数
    users = set()
    for item in data:
        users.add(item.get('user_id', -1))
    print(f"  用户数: {len(users)}")

    # 统计每个维度的标签分布
    emotion_counts = Counter()
    thinking_counts = Counter()
    intent_counts = Counter()
    stance_counts = Counter()

    for item in data:
        assistant_msg = item['messages'][-1]['content']
        for line in assistant_msg.split('\n'):
            if line.startswith('[EMO]'):
                emotion_counts[line[5:]] += 1
            elif line.startswith('[THK]'):
                thinking_counts[line[5:]] += 1
            elif line.startswith('[INT]'):
                intent_counts[line[5:]] += 1
            elif line.startswith('[STN]'):
                stance_counts[line[5:]] += 1

    print(f"\n  情感分布:")
    for emo, count in emotion_counts.most_common():
        print(f"    {emo}: {count}")

    print(f"\n  思维分布:")
    for think, count in thinking_counts.most_common():
        print(f"    {think}: {count}")

    print(f"\n  意图分布:")
    for intent, count in intent_counts.most_common():
        print(f"    {intent}: {count}")

    print(f"\n  立场分布:")
    for stance, count in stance_counts.most_common():
        print(f"    {stance}: {count}")


def balance_dataset(data: List[Dict], target_counts: Dict[str, int] = None) -> List[Dict]:
    """平衡数据集，确保每个类别都有足够的样本"""
    if target_counts is None:
        # 默认每个类别至少100个样本
        target_counts = {label: 100 for label in ['愤怒', '厌恶', '信任', '喜悦', '惊讶', '悲伤', '恐惧', '期待', '平静']}

    # 按情感标签分组
    label_groups = defaultdict(list)
    for item in data:
        assistant_msg = item['messages'][-1]['content']
        for line in assistant_msg.split('\n'):
            if line.startswith('[EMO]'):
                label = line[5:]
                label_groups[label].append(item)
                break

    # 平衡每个类别
    balanced_data = []
    for label, target_count in target_counts.items():
        if label in label_groups:
            samples = label_groups[label]
            if len(samples) > target_count:
                # 随机采样
                balanced_data.extend(random.sample(samples, target_count))
            else:
                # 保留所有样本
                balanced_data.extend(samples)

    return balanced_data


def main():
    # 配置参数
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(data_dir, 'data', 'processed')
    input_file = os.path.join(processed_dir, 'all_data.jsonl')

    # 如果all_data不存在，则合并train.jsonl
    if not os.path.exists(input_file):
        print("合并所有数据...")
        train_file = os.path.join(processed_dir, 'train.jsonl')
        if os.path.exists(train_file):
            # 假设已经有train.jsonl，我们需要重新划分
            data = load_jsonl(train_file)
        else:
            print("错误: 找不到输入数据文件")
            return
    else:
        data = load_jsonl(input_file)

    print(f"原始数据量: {len(data)}")

    # 分析原始数据分布
    analyze_dataset(data, "原始数据")

    # 划分策略选择
    split_strategy = "user"  # 可选: "user" 或 "stratified"

    if split_strategy == "user":
        print("\n使用用户划分策略...")
        train_data, val_data, test_data = split_by_user(data)
    else:
        print("\n使用分层抽样策略...")
        train_data, val_data, test_data = split_stratified(data)

    # 可选：平衡训练集
    balance_train = False
    if balance_train:
        print("\n平衡训练集...")
        train_data = balance_dataset(train_data)

    # 保存划分后的数据集
    save_jsonl(train_data, os.path.join(processed_dir, 'train.jsonl'))
    save_jsonl(val_data, os.path.join(processed_dir, 'val.jsonl'))
    save_jsonl(test_data, os.path.join(processed_dir, 'test.jsonl'))

    # 分析划分后的数据集
    analyze_dataset(train_data, "训练集")
    analyze_dataset(val_data, "验证集")
    analyze_dataset(test_data, "测试集")

    print("\n数据集划分完成！")


if __name__ == "__main__":
    main()