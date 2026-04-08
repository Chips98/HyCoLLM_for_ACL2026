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
    """HCN认知状态PyTorch数据集 (支持外部统计量归一化)"""

    def __init__(self, features_path, normalize=True, mean=None, std=None):
        """
        Args:
            features_path: 特征文件路径
            normalize: 是否归一化
            mean: (可选) 外部传入的均值
            std: (可选) 外部传入的标准差
        """
        # 加载特征和标签
        data = torch.load(features_path, map_location='cpu')
        self.features = data['features'].float() # 确保是float32
        self.labels = data['labels']
        self.label_maps = data['label_maps']
        
        # 1. 保存从外部传入的统计量
        self.norm_mean = mean
        self.norm_std = std

        # 2. 如果开启归一化，则执行
        if normalize:
            self._normalize_features()

        print(f"加载数据集: {len(self.features)} 个样本")
        print(f"特征维度: {self.features.shape}")

    def _normalize_features(self):
        """对特征进行标准化"""
        # 展平特征以便计算统计量
        features_flat = self.features.view(-1, self.features.shape[-1])
        
        # 3. 核心逻辑：如果有外部统计量，就用外部的；否则计算自己的
        if self.norm_mean is None or self.norm_std is None:
            print("📊 (Dataset) 计算当前数据集自身的统计量...")
            mean = features_flat.mean(dim=0)
            std = features_flat.std(dim=0, unbiased=False)
            # 避免除零
            std = torch.clamp(std, min=1e-8)
            
            self.norm_mean = mean
            self.norm_std = std
        else:
            print("🔄 (Dataset) 使用外部传入的统计量(通常来自训练集)...")

        # 4. 执行标准化: (X - mu) / sigma
        # 注意：这里一定要用 self.norm_mean 和 self.norm_std
        self.features = (self.features - self.norm_mean) / self.norm_std

        print("✅ 特征已标准化")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'emotion': self.labels['emotion'][idx],
            'thinking': self.labels['thinking'][idx],
            'intent': self.labels['intent'][idx],
            'stance': self.labels['stance'][idx]
        }
        

def create_data_loaders(train_path, val_path=None, test_path=None,
                       batch_size=32, num_workers=4, shuffle=True):
    """
    创建数据加载器 (修改版：确保Val/Test使用Train的统计量)

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
    # 1. 创建训练数据集 (它会计算并在内部保存 mean/std)
    print(f"📉 (Loader) 加载训练集: {train_path}")
    train_dataset = CognitiveTensorDataset(train_path, normalize=True)

    # 获取训练集的统计量
    train_mean = train_dataset.norm_mean
    train_std = train_dataset.norm_std

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = None
    if val_path:
        print(f"📉 (Loader) 加载验证集 (使用训练集统计量): {val_path}")
        # 关键修改：传入 mean 和 std
        val_dataset = CognitiveTensorDataset(
            val_path,
            normalize=True,
            mean=train_mean,
            std=train_std
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    test_loader = None
    if test_path:
        print(f"📉 (Loader) 加载测试集 (使用训练集统计量): {test_path}")
        # 关键修改：传入 mean 和 std
        test_dataset = CognitiveTensorDataset(
            test_path,
            normalize=True,
            mean=train_mean,
            std=train_std
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader, test_loader



def compute_class_weights(dataset_path, smoothing=0.3, clip_max=5.0):
    """
    计算温和的类别权重
    
    Args:
        dataset_path: 数据集路径
        smoothing: 平滑系数 (0.0-1.0)。
                   1.0 = 原始逆频率（非常激进）
                   0.5 = 平方根平滑（温和）
                   0.3 = 更加温和（推荐）
                   0.0 = 所有权重为1（无权重）
        clip_max: 权重上限截断值，防止极端值破坏训练
    """
    data = torch.load(dataset_path, map_location='cpu', weights_only=False)
    labels = data['labels']

    class_weights = {}

    print(f"\n📊 计算温和类别权重 (Smoothing={smoothing}, Clip_Max={clip_max}):")

    for task, label_list in labels.items():
        if isinstance(label_list, torch.Tensor):
            label_tensor = label_list
        else:
            label_tensor = torch.tensor(label_list)
            
        unique, counts = torch.unique(label_tensor, return_counts=True)
        total_samples = len(label_list)
        num_classes = len(unique)

        # 1. 基础计算：逆类别频率
        # base_weights = total / (n_classes * count)
        base_weights = total_samples / (num_classes * counts.float())
        print(f"  ➤ {task}: 原始权重范围 ({base_weights.min():.2f} - {base_weights.max():.2f})")
        # 2. 核心改进：幂次平滑 (Power Smoothing)
        # 将权重进行幂次运算。例如 weights^0.5 是开根号，能显著压缩数值范围
        smoothed_weights = torch.pow(base_weights, smoothing)
        print(f"    平滑后权重范围 ({smoothed_weights.min():.2f} - {smoothed_weights.max():.2f})")
        # 3. 核心改进：均值归一化 (Mean Normalization)
        # 强制让权重的平均值等于 1.0。
        # 这样做的目的是保持总 Loss 的量级不变，不需要调整学习率
        normalized_weights = smoothed_weights / smoothed_weights.mean()
        print(f"    归一化后权重范围 ({normalized_weights.min():.2f} - {normalized_weights.max():.2f})")
        # 4. 安全网：数值截断 (Clipping)
        # 即使平滑后，如果有只有1个样本的极端类别，权重仍可能过大
        # 强制将最大权重限制在 clip_max (例如 5.0 或 10.0) 以内
        final_weights = torch.clamp(normalized_weights, max=clip_max)
        print(f"    截断后权重范围 ({final_weights.min():.2f} - {final_weights.max():.2f})")    
        # 再次归一化以确保截断后均值仍约为1 (可选，但推荐)
        final_weights = final_weights / final_weights.mean()
        print(f"    最终权重范围 ({final_weights.min():.2f} - {final_weights.max():.2f})")
        # 创建权重映射
        weight_dict = {}
        for idx, weight in zip(unique, final_weights):
            weight_dict[idx.item()] = weight.item()

        class_weights[task] = weight_dict

        print(f"\n  ➤ {task} (Range: {final_weights.min():.2f} - {final_weights.max():.2f}):")
        for idx, weight in sorted(weight_dict.items()):
            print(f"    类别 {idx}: {weight:.4f}")

    return class_weights

def analyze_features(features_path):
    """分析特征统计信息"""
    data = torch.load(features_path, map_location='cpu', weights_only=False)
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