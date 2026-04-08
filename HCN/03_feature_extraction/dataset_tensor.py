#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集构建工具
将提取的特征构建为PyTorch数据集，用于HCN训练
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CognitiveTensorDataset(Dataset):
    """HCN认知状态PyTorch数据集"""

    def __init__(self, features_path, normalize=True):
        """
        初始化数据集

        Args:
            features_path: 特征文件路径(.pt)
            normalize: 是否对特征进行标准化
        """
        # 加载特征和标签
        data = torch.load(features_path, map_location='cpu')
        self.features = data['features']  # [num_samples, 4, hidden_dim]
        self.labels = data['labels']      # dict with keys: emotion, thinking, intent, stance
        self.label_maps = data['label_maps']

        # 特征标准化
        if normalize:
            self._normalize_features()

        print(f"加载数据集: {len(self.features)} 个样本")
        print(f"特征维度: {self.features.shape}")

    def _normalize_features(self):
        """对特征进行标准化"""
        # 计算每个维度的均值和标准差
        # 展平特征以便计算统计量
        features_flat = self.features.view(-1, self.features.shape[-1])
        mean = features_flat.mean(dim=0)
        std = features_flat.std(dim=0, unbiased=False)

        # 避免除零
        std = torch.clamp(std, min=1e-8)

        # 标准化
        self.features = (self.features - mean) / std

        # 保存标准化参数
        self.norm_mean = mean
        self.norm_std = std

        print("特征已标准化")

    def get_norm_stats(self):
        """获取归一化统计量，用于其他数据集使用"""
        return self.norm_mean, self.norm_std

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """获取单个样本"""
        return {
            'features': self.features[idx],  # [4, hidden_dim]
            'emotion': self.labels['emotion'][idx],
            'thinking': self.labels['thinking'][idx],
            'intent': self.labels['intent'][idx],
            'stance': self.labels['stance'][idx]
        }


def create_data_loaders(train_path, val_path=None, test_path=None,
                       batch_size=32, num_workers=4, shuffle=True):
    """
    创建数据加载器

    Args:
        train_path: 训练特征路径
        val_path: 验证特征路径
        test_path: 测试特征路径
        batch_size: 批次大小
        num_workers: 工作进程数
        shuffle: 是否打乱训练数据

    Returns:
        train_loader, val_loader, test_loader
    """
    # 创建训练数据集
    train_dataset = CognitiveTensorDataset(train_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = None
    if val_path:
        val_dataset = CognitiveTensorDataset(val_path)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    test_loader = None
    if test_path:
        test_dataset = CognitiveTensorDataset(test_path)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader, test_loader


def compute_class_weights(dataset_path):
    """计算类别权重，用于处理不平衡数据"""
    data = torch.load(dataset_path, map_location='cpu')
    labels = data['labels']

    # 计算每个维度的类别权重
    class_weights = {}

    for task, label_list in labels.items():
        # 统计每个类别的样本数
        unique, counts = torch.unique(torch.tensor(label_list), return_counts=True)
        total_samples = len(label_list)

        # 计算权重（逆频率）
        weights = total_samples / (len(unique) * counts.float())

        # 创建权重映射
        weight_dict = {}
        for idx, weight in zip(unique, weights):
            weight_dict[idx.item()] = weight.item()

        class_weights[task] = weight_dict

        print(f"\n{task} 类别权重:")
        for idx, weight in sorted(weight_dict.items()):
            print(f"  类别 {idx}: {weight:.4f}")

    return class_weights


def analyze_features(features_path):
    """分析特征统计信息"""
    data = torch.load(features_path, map_location='cpu')
    features = data['features']

    print("\n特征分析:")
    print(f"  形状: {features.shape}")
    print(f"  数据类型: {features.dtype}")
    print(f"  设备: {features.device}")

    # 计算统计量
    features_flat = features.view(-1, features.shape[-1])
    print(f"\n特征统计 (所有维度合并):")
    print(f"  均值: {features_flat.mean().item():.6f}")
    print(f"  标准差: {features_flat.std().item():.6f}")
    print(f"  最小值: {features_flat.min().item():.6f}")
    print(f"  最大值: {features_flat.max().item():.6f}")

    # 分析每个认知维度的特征
    print("\n各认知维度特征分析:")
    dim_names = ['Emotion', 'Thinking', 'Intent', 'Stance']
    for i, name in enumerate(dim_names):
        dim_features = features[:, i, :]
        print(f"\n{name} 维度:")
        print(f"  均值: {dim_features.mean().item():.6f}")
        print(f"  标准差: {dim_features.std().item():.6f}")
        print(f"  范围: [{dim_features.min().item():.6f}, {dim_features.max().item():.6f}]")

    # 计算维度间的相关性（使用平均池化降维）
    print("\n维度间相关性:")

    # 将每个维度的特征平均池化到较低维度，避免维度灾难
    # 这里我们简单地取每个维度特征的平均值作为代表
    dim_means = []
    for i in range(4):
        dim_feature = features[:, i, :]  # [batch, hidden_dim]
        dim_mean = dim_feature.mean(dim=1)  # [batch] - 对hidden_dim求平均
        dim_means.append(dim_mean)

    # 计算维度间的相关性
    for i in range(4):
        for j in range(i+1, 4):
            feat_i = dim_means[i]
            feat_j = dim_means[j]

            # 转换为float32提高数值稳定性
            feat_i = feat_i.float()
            feat_j = feat_j.float()

            # 检查标准差
            std_i = feat_i.std()
            std_j = feat_j.std()

            print(f"  调试 {dim_names[i]}-{dim_names[j]}: std_i={std_i.item():.8f}, std_j={std_j.item():.8f}")

            if std_i.item() < 1e-8 or std_j.item() < 1e-8:
                print(f"  {dim_names[i]}-{dim_names[j]}: 无法计算（标准差接近0）")
                print(f"    {dim_names[i]} 范围: [{feat_i.min().item():.6f}, {feat_i.max().item():.6f}]")
                print(f"    {dim_names[j]} 范围: [{feat_j.min().item():.6f}, {feat_j.max().item():.6f}]")
            else:
                # 使用更稳定的皮尔逊相关系数公式
                feat_i_centered = feat_i - feat_i.mean()
                feat_j_centered = feat_j - feat_j.mean()

                numerator = (feat_i_centered * feat_j_centered).mean()
                denominator = std_i * std_j

                if denominator.item() < 1e-8:
                    print(f"  {dim_names[i]}-{dim_names[j]}: 无法计算（分母太小）")
                else:
                    corr = numerator / denominator
                    if torch.isnan(corr):
                        print(f"  {dim_names[i]}-{dim_names[j]}: 无法计算（仍然NaN）")
                        print(f"    numerator: {numerator.item():.8f}")
                        print(f"    denominator: {denominator.item():.8f}")
                    else:
                        print(f"  {dim_names[i]}-{dim_names[j]}: {corr.item():.6f}")

    # 额外：使用样本级别的特征统计
    print(f"\n样本级别特征统计:")
    for i in range(4):
        dim_feature = features[:, i, :].float()  # 转换为float32避免精度问题
        sample_norms = torch.norm(dim_feature, dim=1)  # 每个样本的L2范数
        print(f"  {dim_names[i]}: 平均范数={sample_norms.mean().item():.4f}, "
              f"范数标准差={sample_norms.std().item():.4f}")


def main():
    """测试数据集加载功能"""
    import argparse

    parser = argparse.ArgumentParser(description="测试数据集加载")
    parser.add_argument("--train_path", type=str, required=True, help="训练特征路径")
    parser.add_argument("--val_path", type=str, help="验证特征路径")
    parser.add_argument("--test_path", type=str, help="测试特征路径")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--analyze", action="store_true", help="分析特征统计")

    args = parser.parse_args()

    if args.analyze:
        analyze_features(args.train_path)
        return

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        args.train_path,
        args.val_path,
        args.test_path,
        batch_size=args.batch_size,
        shuffle=True
    )

    # 测试数据加载
    print("\n测试数据加载...")
    for batch in train_loader:
        print(f"特征形状: {batch['features'].shape}")
        print(f"情感标签形状: {batch['emotion'].shape}")
        print(f"思维标签形状: {batch['thinking'].shape}")
        print(f"意图标签形状: {batch['intent'].shape}")
        print(f"立场标签形状: {batch['stance'].shape}")
        break

    # 计算类别权重
    print("\n计算类别权重...")
    class_weights = compute_class_weights(args.train_path)


if __name__ == "__main__":
    main()