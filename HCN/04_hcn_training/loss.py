#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HCN模型的损失函数实现
包含多任务损失、双曲空间正则化损失等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import itertools
from models.hyperbolic import PoincareBall


class HCNMutliTaskLoss(nn.Module):
    """HCN多任务损失函数"""

    def __init__(self,
                 task_weights=[1.0, 1.0, 1.0, 1.0],
                 use_uncertainty_weighting=False,
                 use_dynamic_weighting=False,
                 temperature=2.0,
                 lambda_hyper=0.1,
                 lambda_contrastive=0.1):
        super().__init__()

        self.task_names = ['emotion', 'thinking', 'intent', 'stance']
        self.n_tasks = len(self.task_names)

        # 基础交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss()

        # 任务权重
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.use_dynamic_weighting = use_dynamic_weighting

        if use_uncertainty_weighting:
            # 可学习的权重（基于不确定性）
            self.log_vars = nn.Parameter(torch.zeros(self.n_tasks))
        else:
            self.register_buffer('task_weights', torch.tensor(task_weights))

        # 损失权重
        self.lambda_hyper = lambda_hyper  # 双曲空间正则化权重
        self.lambda_contrastive = lambda_contrastive  # 对比学习权重
        self.temperature = temperature

        # 自动调整温度参数：如果使用默认高温则降低
        if temperature > 1.0:
            self.temperature = 0.5  # 更合适的温度参数
            print(f"🔧 自动调整温度参数: {temperature} -> {self.temperature} (提升对比学习效果)")

        # 类别权重（处理不平衡）
        self.class_weights = None

        # 双曲流形实例将在forward中动态传入，避免创建独立的参数

    def set_class_weights(self, class_weights: Dict[str, torch.Tensor]):
        """设置类别权重"""
        self.class_weights = class_weights

    def compute_task_loss(self, logits, labels, task_name):
        """计算单个任务的损失"""
        # 应用类别权重（如果有）
        if self.class_weights and task_name in self.class_weights:
            class_weight_dict = self.class_weights[task_name]

            # 获取当前批次中出现的类别
            unique_labels = torch.unique(labels)

            # 创建权重tensor
            num_classes = logits.size(-1)
            weight_tensor = torch.ones(num_classes, device=logits.device)

            # 为已知类别设置权重
            for class_idx, weight in class_weight_dict.items():
                if class_idx < num_classes:
                    weight_tensor[class_idx] = weight

            # 创建带权重的损失函数
            ce_loss = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            ce_loss = nn.CrossEntropyLoss()

        loss = ce_loss(logits, labels)
        return loss

    def compute_task_weights(self):
        """计算当前的任务权重"""
        if self.use_uncertainty_weighting:
            # 基于不确定性的权重：1 / (2 * sigma^2)
            weights = torch.exp(-self.log_vars)
            return weights
        else:
            return self.task_weights

    def hyperbolic_regularization_loss(self, feature_dict, labels, hyperbolic_layer=None):
        """
        改进版双曲空间正则化损失：包含正样本拉近和负样本推开
        使用双曲对比学习增强特征表示能力
        """
        if feature_dict is None or 'input_hyperbolic' not in feature_dict:
            return torch.tensor(0.0, device=labels['emotion'].device)

        features = feature_dict['input_hyperbolic'] # [batch_size, 4, hidden_dim]
        device = features.device

        # 获取流形参数（消融实验兼容：检查是否为Identity层）
        if hyperbolic_layer is not None and hasattr(hyperbolic_layer, 'curvature'):
            # 双曲模式：使用实际的曲率参数
            c = hyperbolic_layer.curvature
            eps = hyperbolic_layer.eps
        else:
            # 欧氏模式或Identity层：不计算双曲损失，直接返回0
            # 这是因为欧氏空间中没有双曲几何概念
            return torch.tensor(0.0, device=device)

        total_loss = torch.tensor(0.0, device=device)
        n_tasks = 0

        # 对每个任务维度分别计算
        for dim_idx, task_name in enumerate(self.task_names):
            dim_features = features[:, dim_idx, :]  # [B, D]
            dim_labels = labels[task_name]          # [B]

            # 跳过空任务
            if len(dim_labels) == 0: continue

            # === 计算所有样本对之间的庞加莱距离矩阵 ===
            x = dim_features
            N = x.size(0)

            # 如果样本数量太少，跳过
            if N < 2:
                continue

            # 1. 计算欧氏范数平方 ||x||^2 [N, 1]
            x_norm_sq = torch.sum(x ** 2, dim=1, keepdim=True)
            x_norm_sq = torch.clamp(x_norm_sq, max=(1.0 - 1e-5)/c)

            # 2. 计算欧氏距离平方矩阵 ||u-v||^2 [N, N]
            euclidean_dist_sq = x_norm_sq + x_norm_sq.t() - 2 * torch.matmul(x, x.t())
            euclidean_dist_sq = torch.clamp(euclidean_dist_sq, min=0.0)

            # 3. 计算分母矩阵
            alpha = 1 - c * x_norm_sq
            alpha = torch.clamp(alpha, min=eps)
            denominator = torch.matmul(alpha, alpha.t())
            denominator = torch.clamp(denominator, min=eps)

            # 4. 计算双曲距离矩阵
            delta = 2 * c * euclidean_dist_sq / denominator
            delta = torch.clamp(delta, min=0.0, max=1e5)
            arg = 1 + delta
            arg = torch.clamp(arg, min=1.0 + 1e-5)
            sqrt_c = torch.sqrt(c)
            dist_matrix = (1 / sqrt_c) * torch.acosh(arg)

            # 构建正负样本掩码
            # label_matrix: [N, N], (i, j)为True表示i和j同类
            label_matrix = dim_labels.unsqueeze(0) == dim_labels.unsqueeze(1)
            # 排除对角线
            mask_diag = torch.eye(N, device=device).bool()
            label_matrix = label_matrix & (~mask_diag)

            # 正样本对距离 (拉近)
            pos_dist = dist_matrix[label_matrix]
            pos_loss = pos_dist.mean() if pos_dist.numel() > 0 else torch.tensor(0.0, device=device)

            # 负样本对距离 (推开) - 使用 Margin Ranking
            neg_dist = dist_matrix[~label_matrix & ~mask_diag]

            # 简单的推开损失：exp(-neg_dist)
            # 或者 triplet loss 风格
            if neg_dist.numel() > 0 and pos_dist.numel() > 0:
                # 采样负样本以匹配正样本数量，避免类别不平衡影响
                if neg_dist.numel() > pos_dist.numel():
                     idx = torch.randperm(neg_dist.nelement())[:pos_dist.nelement()]
                     sampled_neg_dist = neg_dist[idx]
                else:
                     sampled_neg_dist = neg_dist

                # Margin Loss: max(0, pos - neg + margin)
                margin = 0.5
                # 使用平均正样本距离与平均负样本距离计算triplet loss
                triplet_loss = F.relu(pos_loss - sampled_neg_dist.mean() + margin)

                # 组合损失：正样本拉近 + 负样本推开
                task_loss = pos_loss + triplet_loss
            else:
                # 如果只有正样本或只有负样本，使用简单损失
                task_loss = pos_loss

            total_loss += task_loss
            n_tasks += 1

        if n_tasks > 0:
            total_loss = total_loss / n_tasks

        return total_loss

    def contrastive_loss(self, feature_dict, labels):
        """
        对比学习损失
        增强不同认知维度之间的区分度
        """
        if feature_dict is None or 'encoded_euclidean' not in feature_dict:
            return torch.tensor(0.0, device=labels['emotion'].device)

        # 从字典中取出Transformer编码后的特征
        features = feature_dict['encoded_euclidean']  # [batch_size, 4, hidden_dim]
        batch_size = features.size(0)
        device = features.device

        # 将特征归一化，添加数值稳定性保护
        features_norm = F.normalize(features, p=2, dim=-1, eps=1e-8)

        # 检查归一化后特征的有效性
        if not torch.isfinite(features_norm).all():
            print(f"警告：归一化后特征包含无效值，跳过对比损失")
            return torch.tensor(0.0, device=device)

        # 计算维度间的相似度矩阵
        similarity_matrix = torch.matmul(
            features_norm, features_norm.transpose(-2, -1)
        )  # [batch_size, 4, 4]

        # 检查相似度矩阵的有效性，如果包含无效值则返回0损失
        if not torch.isfinite(similarity_matrix).all():
            print(f"警告：相似度矩阵包含无效值，对比损失设为0")
            return torch.tensor(0.0, device=device)

        total_loss = torch.tensor(0.0, device=device)

        # 对于每个样本，希望不同维度的特征区分明显
        for i in range(batch_size):
            sim = similarity_matrix[i]  # [4, 4]

            # 对角线元素应该是高的（同维度相似）
            # 非对角线元素应该是低的（不同维度不相似）
            mask = torch.eye(4, device=device).bool()

            # 正样本：对角线
            pos_sim = sim[mask]
            # 负样本：非对角线
            neg_sim = sim[~mask]

            # InfoNCE风格的对比损失
            logits = torch.cat([pos_sim, neg_sim]) / self.temperature
            labels_tensor = torch.cat([
                torch.ones(pos_sim.size(0), device=device),
                torch.zeros(neg_sim.size(0), device=device)
            ])

            total_loss += F.binary_cross_entropy_with_logits(
                logits, labels_tensor
            )

        return total_loss / batch_size

    def forward(self, outputs, labels, feature_dict=None, hyperbolic_layer=None):
        """
        计算总损失

        Args:
            outputs: dict containing logits for each task
            labels: dict containing labels for each task
            feature_dict: optional dict containing intermediate features for regularization
            hyperbolic_layer: optional hyperbolic layer instance for distance calculation
        """
        # 获取设备（从任意一个输出tensor获取）
        device = next(iter(outputs.values())).device

        # 计算各任务的基础损失
        task_losses = []
        for i, task_name in enumerate(self.task_names):
            if task_name in outputs and task_name in labels:
                loss = self.compute_task_loss(
                    outputs[task_name],
                    labels[task_name],
                    task_name
                )
                task_losses.append(loss)
            else:
                task_losses.append(torch.tensor(0.0, device=device))

        # 堆叠任务损失
        task_losses = torch.stack(task_losses)

        # 计算任务权重并确保在正确设备上
        weights = self.compute_task_weights()
        if not self.use_uncertainty_weighting:
            weights = weights.to(device)

        # 加权任务损失
        weighted_task_loss = torch.sum(weights * task_losses)

        # 添加不确定性正则项（如果使用不确定性权重）
        if self.use_uncertainty_weighting:
            uncertainty_reg = torch.sum(self.log_vars.to(device))
            weighted_task_loss += uncertainty_reg

        # 双曲空间正则化损失
        hyper_loss = self.hyperbolic_regularization_loss(feature_dict, labels, hyperbolic_layer)
        hyper_loss = self.lambda_hyper * hyper_loss

        # 对比学习损失
        contrastive_loss = self.contrastive_loss(feature_dict, labels)
        contrastive_loss = self.lambda_contrastive * contrastive_loss

        # 总损失
        total_loss = weighted_task_loss + hyper_loss + contrastive_loss

        # 检查总损失的有效性，如果无效则直接报错
        if torch.isnan(total_loss):
            raise ValueError(f"总损失为NaN! 任务损失: {weighted_task_loss.item():.6f}, 双曲损失: {hyper_loss.item():.6f}, 对比损失: {contrastive_loss.item():.6f}")
        if torch.isinf(total_loss):
            raise ValueError(f"总损失为Inf! 任务损失: {weighted_task_loss.item():.6f}, 双曲损失: {hyper_loss.item():.6f}, 对比损失: {contrastive_loss.item():.6f}")

        # 记录各项损失
        self.last_losses = {
            'total': total_loss.item(),
            'task_loss': weighted_task_loss.item(),
            'hyper_loss': hyper_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'emotion': task_losses[0].item(),
            'thinking': task_losses[1].item(),
            'intent': task_losses[2].item(),
            'stance': task_losses[3].item()
        }

        return total_loss


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


class TripletLoss(nn.Module):
    """三元组损失用于特征学习"""

    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """计算三元组损失"""
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


def test_loss_functions():
    """测试损失函数"""
    print("测试损失函数...")

    # 创建模拟数据
    batch_size = 4
    outputs = {
        'emotion': torch.randn(batch_size, 10),
        'thinking': torch.randn(batch_size, 2),
        'intent': torch.randn(batch_size, 9),
        'stance': torch.randn(batch_size, 6)
    }
    labels = {
        'emotion': torch.randint(0, 10, (batch_size,)),
        'thinking': torch.randint(0, 2, (batch_size,)),
        'intent': torch.randint(0, 9, (batch_size,)),
        'stance': torch.randint(0, 6, (batch_size,))
    }

    # 创建特征字典，包含双曲空间和欧氏空间特征
    feature_dict = {
        'input_hyperbolic': torch.randn(batch_size, 4, 512),  # 双曲空间特征
        'encoded_euclidean': torch.randn(batch_size, 4, 512)  # 编码后的欧氏空间特征
    }

    # 测试多任务损失
    loss_fn = HCNMutliTaskLoss(
        task_weights=[1.0, 1.0, 1.0, 1.0],
        lambda_hyper=0.1,
        lambda_contrastive=0.1
    )

    loss = loss_fn(outputs, labels, feature_dict)
    print(f"总损失: {loss.item():.4f}")
    print("各项损失:", loss_fn.last_losses)

    print("测试完成！")


if __name__ == "__main__":
    test_loss_functions()