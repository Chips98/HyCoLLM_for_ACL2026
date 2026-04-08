#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一数据预处理脚本
集成数据加载、提示词构建、数据划分功能
从labels.json读取标签定义，删除所有硬编码标签
"""

import json
import os
import random
import numpy as np
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


class LabelManager:
    """标签管理器，从labels.json加载所有标签定义"""

    def __init__(self, labels_path: str):
        self.labels_path = labels_path
        self.labels = self._load_labels()

    def _load_labels(self) -> Dict[str, Any]:
        """加载标签定义"""
        with open(self.labels_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_all_emotions(self) -> List[str]:
        """获取所有情感标签"""
        return list(self.labels['emotion'].keys())

    def get_all_thinking_types(self) -> List[str]:
        """获取所有思维类型标签"""
        return list(self.labels['thinking']['types'].keys())

    def get_all_thinking_values(self) -> List[str]:
        """获取所有思维值标签"""
        return list(self.labels['thinking']['values'].keys())

    def get_all_intents(self) -> List[str]:
        """获取所有意图标签"""
        return list(self.labels['intent'].keys())

    def get_all_stances(self) -> List[str]:
        """获取所有立场标签"""
        return list(self.labels['stance'].keys())

    def analyze_thinking_type(self, thinking_value: str) -> str:
        """根据thinking_value判断思维类型"""
        # 直觉型思维值
        intuitive_values = [
            '主观评价', '认同驱动的顺应', '情绪化判断', '基于经验'
        ]
        # 分析型思维值
        analytical_values = [
            '逻辑', '权衡', '循证', '批判'
        ]

        if thinking_value in intuitive_values:
            return '直觉型'
        elif thinking_value in analytical_values:
            return '分析型'
        else:
            return '直觉型'  # 默认值

    def build_system_prompt(self) -> str:
        """构建系统提示词 - 使用统一的utils.prompts模块"""
        # 导入统一的提示词管理
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.prompts import get_prompts

        prompts = get_prompts(self.labels_path)
        return prompts.get_system_prompt()

    def build_assistant_response(self, emotion: str, thinking_value: str,
                               intent: str, stance: str) -> str:
        """构建助手响应"""
        # 注意：新的标签体系中，thinking_value就是实际的思维值标签
        # 不再转换为思维类型，直接使用原始值
        return f"<<<EMOTION>>>{emotion}\n<<<THINKING>>>{thinking_value}\n<<<INTENT>>>{intent}\n<<<STANCE>>>{stance}"


class UnifiedDataProcessor:
    """统一数据预处理器"""

    def __init__(self, labels_path: str):
        self.label_manager = LabelManager(labels_path)

    def load_raw_data(self, data_path: str) -> List[Dict]:
        """加载原始JSON数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def load_jsonl_data(self, file_path: str) -> List[Dict]:
        """加载JSONL格式数据"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def save_jsonl(self, data: List[Dict], file_path: str):
        """保存为JSONL格式"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"已保存 {len(data)} 条数据到 {file_path}")

    def build_user_prompt(self, context_post: str, target_post: str) -> str:
        """构建用户提示词"""
        return f"[背景]{context_post}\n[评论]{target_post}\n[指令]请分析该评论的认知状态。"

    def convert_to_chatml(self, data: List[Dict]) -> List[Dict]:
        """将原始数据转换为ChatML格式"""
        chatml_data = []

        for item in tqdm(data, desc="转换为ChatML格式"):
            # 验证必要字段
            if not all(key in item for key in ['context_post', 'target_post', 'cognitive_labels']):
                continue

            # 构建对话
            system_prompt = self.label_manager.build_system_prompt()
            user_prompt = self.build_user_prompt(item['context_post'], item['target_post'])
            assistant_response = self.label_manager.build_assistant_response(
                emotion=item['cognitive_labels'].get('emotion', ''),
                thinking_value=item['cognitive_labels'].get('thinking_value', ''),
                intent=item['cognitive_labels'].get('intent', ''),
                stance=item['cognitive_labels'].get('stance', '')
            )

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

    def extract_primary_labels(self, data: List[Dict]) -> List[str]:
        """提取主要标签用于分层抽样（使用情感标签）"""
        labels = []
        for item in data:
            # 从assistant消息中提取情感标签
            assistant_msg = item['messages'][-1]['content']
            for line in assistant_msg.split('\n'):
                if line.startswith('<<<EMOTION>>>'):
                    labels.append(line[12:])  # 去掉<<<EMOTION>>>前缀
                    break
        return labels

    def split_by_user(self, data: List[Dict], train_ratio: float = 0.8,
                     val_ratio: float = 0.1, test_ratio: float = 0.1,
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

    def split_stratified(self, data: List[Dict], train_ratio: float = 0.8,
                        val_ratio: float = 0.1, test_ratio: float = 0.1,
                        seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """分层抽样划分数据集，保持标签分布一致"""
        random.seed(seed)
        np.random.seed(seed)

        # 提取标签
        labels = self.extract_primary_labels(data)

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

    def analyze_dataset(self, data: List[Dict], name: str = "Dataset"):
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
                if line.startswith('<<<EMOTION>>>'):
                    emotion_counts[line[12:]] += 1
                elif line.startswith('<<<THINKING>>>'):
                    thinking_counts[line[14:]] += 1
                elif line.startswith('<<<INTENT>>>'):
                    intent_counts[line[11:]] += 1
                elif line.startswith('<<<STANCE>>>'):
                    stance_counts[line[12:]] += 1

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

    def process_data(self, input_path: str, output_dir: str,
                    split_data: bool = True, split_strategy: str = "user",
                    train_ratio: float = 0.8, val_ratio: float = 0.1,
                    test_ratio: float = 0.1, output_filename: str = "all_data.jsonl"):
        """
        完整数据处理流程

        Args:
            input_path: 输入数据路径
            output_dir: 输出目录
            split_data: 是否划分数据集
            split_strategy: 划分策略 ("user" 或 "stratified")
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        """
        # 设置随机种子
        random.seed(42)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print("统一数据预处理流程")
        print("=" * 60)

        # 判断输入格式并加载数据
        if input_path.endswith('.json'):
            print(f"加载JSON数据: {input_path}")
            raw_data = self.load_raw_data(input_path)
        elif input_path.endswith('.jsonl'):
            print(f"加载JSONL数据: {input_path}")
            raw_data = self.load_jsonl_data(input_path)
        else:
            raise ValueError("输入文件格式必须是 .json 或 .jsonl")

        print(f"原始数据量: {len(raw_data)} 条")

        # 如果已经是ChatML格式（通过检查messages字段），跳过转换
        if 'messages' in raw_data[0] if raw_data else False:
            print("检测到数据已经是ChatML格式，跳过转换步骤")
            chatml_data = raw_data
        else:
            print("转换为ChatML格式...")
            chatml_data = self.convert_to_chatml(raw_data)
            print(f"转换后数据量: {len(chatml_data)} 条")

        # 保存完整的ChatML数据
        all_output_path = os.path.join(output_dir, output_filename)
        self.save_jsonl(chatml_data, all_output_path)

        if split_data:
            print(f"\n使用 {split_strategy} 策略划分数据集...")

            if split_strategy == "user":
                train_data, val_data, test_data = self.split_by_user(
                    chatml_data, train_ratio, val_ratio, test_ratio
                )
            else:  # stratified
                train_data, val_data, test_data = self.split_stratified(
                    chatml_data, train_ratio, val_ratio, test_ratio
                )

            # 保存划分后的数据集
            self.save_jsonl(train_data, os.path.join(output_dir, 'train.jsonl'))
            self.save_jsonl(val_data, os.path.join(output_dir, 'val.jsonl'))
            self.save_jsonl(test_data, os.path.join(output_dir, 'test.jsonl'))

            # 分析数据集
            self.analyze_dataset(chatml_data, "完整数据集")
            self.analyze_dataset(train_data, "训练集")
            self.analyze_dataset(val_data, "验证集")
            self.analyze_dataset(test_data, "测试集")
        else:
            print("\n跳过数据集划分")
            self.analyze_dataset(chatml_data, "处理后的数据集")

        print(f"\n数据处理完成！输出目录: {output_dir}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='统一数据预处理脚本')

    # 必需参数
    parser.add_argument('--labels', type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cut', 'labels.json'),
                       help='标签配置文件路径')
    parser.add_argument('--output-dir', type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed'),
                       help='输出目录路径')

    # 训练集参数
    parser.add_argument('--train-input', type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cut', 'train_data.json'),
                       help='训练集输入文件路径')
    parser.add_argument('--train-output', type=str, default='train.jsonl',
                       help='训练集输出文件名')

    # 测试集参数
    parser.add_argument('--test-input', type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cut', 'test_data.json'),
                       help='测试集输入文件路径')
    parser.add_argument('--test-output', type=str, default='test.jsonl',
                       help='测试集输出文件名')

    # 处理选项
    parser.add_argument('--process-train', action='store_true', default=True,
                       help='是否处理训练集')
    parser.add_argument('--process-test', action='store_true', default=True,
                       help='是否处理测试集')
    parser.add_argument('--skip-train', action='store_true',
                       help='跳过训练集处理')
    parser.add_argument('--skip-test', action='store_true',
                       help='跳过测试集处理')

    # 划分选项（仅当需要划分时使用）
    parser.add_argument('--split-data', action='store_true',
                       help='是否对数据进行划分')
    parser.add_argument('--split-strategy', type=str, default='user',
                       choices=['user', 'stratified'],
                       help='数据集划分策略')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='测试集比例')

    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()

    # 检查标签文件是否存在
    if not os.path.exists(args.labels):
        print(f"错误: 标签文件不存在: {args.labels}")
        return

    # 初始化处理器
    processor = UnifiedDataProcessor(args.labels)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("统一数据预处理流程")
    print("=" * 60)
    print(f"标签配置文件: {args.labels}")
    print(f"输出目录: {args.output_dir}")
    print(f"是否划分数据集: {args.split_data}")
    if args.split_data:
        print(f"划分策略: {args.split_strategy}")
        print(f"训练集比例: {args.train_ratio}")
        print(f"验证集比例: {args.val_ratio}")
        print(f"测试集比例: {args.test_ratio}")
    print()

    # 处理训练集
    if not args.skip_train and args.process_train:
        print("处理训练集...")
        if not os.path.exists(args.train_input):
            print(f"警告: 训练集文件不存在: {args.train_input}")
        else:
            processor.process_data(
                input_path=args.train_input,
                output_dir=args.output_dir,
                split_data=args.split_data,
                split_strategy=args.split_strategy,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                output_filename=args.train_output
            )
        print()

    # 处理测试集
    if not args.skip_test and args.process_test:
        print("处理测试集...")
        if not os.path.exists(args.test_input):
            print(f"警告: 测试集文件不存在: {args.test_input}")
        else:
            processor.process_data(
                input_path=args.test_input,
                output_dir=args.output_dir,
                split_data=False,  # 测试集不进行划分
                output_filename=args.test_output
            )
        print()

    print("数据处理完成！")
    print(f"输出文件保存在: {args.output_dir}")

    # 列出生成的文件
    output_files = []
    if not args.skip_train and args.process_train:
        output_files.append(args.train_output)
    if not args.skip_test and args.process_test:
        output_files.append(args.test_output)

    print("生成的文件:")
    for filename in output_files:
        filepath = os.path.join(args.output_dir, filename)
        if os.path.exists(filepath):
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} (文件不存在)")


if __name__ == "__main__":
    main()