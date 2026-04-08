#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双曲空间数学操作实现
实现庞加莱球模型的核心运算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PoincareBall(nn.Module):
    """庞加莱球模型实现"""

    def __init__(self, dim=768, curvature=1.0, learnable_curvature=False):
        """
        初始化庞加莱球

        Args:
            dim: 嵌入维度
            curvature: 曲率参数c
            learnable_curvature: 是否学习曲率参数
        """
        super().__init__()
        self.dim = dim

        if learnable_curvature:
            self.curvature = nn.Parameter(torch.tensor(curvature))
        else:
            self.register_buffer('curvature', torch.tensor(curvature))

        # 缓存常用的值
        self.register_buffer('eps', torch.tensor(1e-8))
        self.register_buffer('max_norm', torch.tensor(1.0 - 1e-5))

    def exp_map(self, x, c=None):
        """
        指数映射：欧氏空间 -> 双曲空间
        公式: exp_0^c(x) = tanh(sqrt(c)||x||) * x / (sqrt(c)||x||)
        """
        if c is None:
            c = self.curvature

        # 计算范数
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x = torch.clamp(norm_x, min=self.eps, max=1e6)

        # 计算指数映射
        sqrt_c = torch.sqrt(c)
        scaled_norm = sqrt_c * norm_x
        tanh_scaled = torch.tanh(scaled_norm)

        # 避免除零
        denominator = sqrt_c * norm_x
        denominator = torch.clamp(denominator, min=self.eps)

        result = tanh_scaled * x / denominator

        # 检查结果的有效性
        if not torch.isfinite(result).all():
            print(f"警告：exp_map结果包含无效值，使用输入作为替代")
            return torch.clamp(x, max=self.max_norm)

        # 确保结果在单位球内
        norm_result = torch.norm(result, p=2, dim=-1, keepdim=True)
        scale = torch.clamp(self.max_norm / norm_result, max=1.0)
        result = result * scale

        return result

    def log_map(self, y, c=None):
        """
        对数映射：双曲空间 -> 欧氏空间
        公式: log_0^c(y) = (1/sqrt(c)) * atanh(sqrt(c)||y||) * y / ||y||
        """
        if c is None:
            c = self.curvature

        # 计算范数
        norm_y = torch.norm(y, p=2, dim=-1, keepdim=True)
        # 确保范数在有效范围内
        norm_y = torch.clamp(norm_y, min=self.eps, max=self.max_norm)

        # 计算对数映射
        sqrt_c = torch.sqrt(c)
        scaled_norm = sqrt_c * norm_y

        # 确保atanh输入有效 (必须小于1)
        scaled_norm = torch.clamp(scaled_norm, max=1.0 - 1e-8)
        atanh_scaled = torch.atanh(scaled_norm)

        # 避免除零
        denominator = sqrt_c * norm_y
        denominator = torch.clamp(denominator, min=self.eps)

        result = (1 / sqrt_c) * atanh_scaled * y / denominator

        # 检查结果的有效性
        if not torch.isfinite(result).all():
            print(f"警告：log_map结果包含无效值，使用输入作为替代")
            return y

        return result

    def poincare_distance(self, u, v, c=None):
        """
        计算庞加莱距离（数值稳定版本）
        公式: d_D(u,v) = (1/sqrt(c)) * arccosh(1 + 2c||u-v||^2 / ((1-c||u||^2)(1-c||v||^2)))
        """
        if c is None:
            c = self.curvature

        sqrt_c = torch.sqrt(c)

        # 计算范数
        norm_u = torch.norm(u, p=2, dim=-1)
        norm_v = torch.norm(v, p=2, dim=-1)

        # 计算差向量
        diff = u - v
        norm_diff = torch.norm(diff, p=2, dim=-1)

        # 严格限制范数，确保在单位球内
        max_norm = (1.0 - 1e-6) / torch.sqrt(c)
        norm_u = torch.clamp(norm_u, max=max_norm)
        norm_v = torch.clamp(norm_v, max=max_norm)

        # 计算庞加莱距离
        numerator = 2 * c * norm_diff ** 2
        denominator = (1 - c * norm_u ** 2) * (1 - c * norm_v ** 2)
        denominator = torch.clamp(denominator, min=self.eps, max=1e10)

        ratio = numerator / denominator
        # 限制ratio范围，避免arccosh数值不稳定
        ratio = torch.clamp(ratio, min=0.0, max=1e6)
        arg = 1 + ratio

        # === 关键修改 ===
        # acosh(x) 在 x=1 处导数无穷大。
        # 我们必须强制 arg >= 1 + epsilon，避免梯度爆炸。
        # 1e-5 是一个经验值，既保证数值稳定又不明显影响距离计算
        arg = torch.clamp(arg, min=1.0 + 1e-5, max=1e6)
        # ===============

        # 使用数值稳定的acosh实现：acosh(x) = log(x + sqrt(x^2 - 1))
        distance = (1 / sqrt_c) * torch.acosh(arg)

        # 检查结果的有效性
        distance = torch.clamp(distance, min=0.0, max=1e6)

        return distance

    def mobius_add(self, u, v, c=None):
        """
        Möbius加法：双曲空间中的向量加法
        """
        if c is None:
            c = self.curvature

        sqrt_c = torch.sqrt(c)
        norm_u_sq = torch.sum(u ** 2, dim=-1, keepdim=True)
        norm_v_sq = torch.sum(v ** 2, dim=-1, keepdim=True)
        dot_uv = torch.sum(u * v, dim=-1, keepdim=True)

        numerator = (1 + 2 * sqrt_c * dot_uv + c * norm_v_sq) * u + (1 - c * norm_u_sq) * v
        denominator = 1 + 2 * sqrt_c * dot_uv + c * norm_u_sq * norm_v_sq
        denominator = torch.clamp(denominator, min=self.eps)

        result = numerator / denominator

        # 确保结果在单位球内
        return self.project_to_ball(result, c)

    def mobius_scalar_mul(self, r, x, c=None):
        """
        Möbius标量乘法：双曲空间中的标量乘法
        """
        if c is None:
            c = self.curvature

        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x = torch.clamp(norm_x, min=self.eps)

        # 计算缩放因子
        sqrt_c = torch.sqrt(c)
        denom = sqrt_c * norm_x
        denom = torch.clamp(denom, min=1e-8)

        result = torch.tanh(r * torch.atanh(denom)) * x / denom

        return self.project_to_ball(result, c)

    def project_to_ball(self, x, c=None):
        """将向量投影到单位球内"""
        if c is None:
            c = self.curvature

        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        max_norm = 1.0 / torch.sqrt(c) - self.eps

        # 如果向量范数超过最大值，进行缩放
        scale = torch.clamp(max_norm / norm, max=1.0)
        return x * scale

    def gyration(self, u, v, w, c=None):
        """
        陀螺运动（gyration）：双曲空间中的变换
        """
        if c is None:
            c = self.curvature

        # Möbius加法的非交换性修正
        neg_v = self.mobius_scalar_mul(-1, v, c)
        sum_uv = self.mobius_add(u, v, c)
        sum_vneg_w = self.mobius_add(neg_v, w, c)
        sum_uv_vneg_w = self.mobius_add(sum_uv, sum_vneg_w, c)
        sum_vneg_u = self.mobius_add(neg_v, u, c)
        sum_vneg_u_v = self.mobius_add(sum_vneg_u, v, c)

        return self.mobius_add(sum_uv_vneg_w, self.mobius_scalar_mul(-1, sum_vneg_u_v, c), c)

    def parallel_transport(self, u, v, w, c=None):
        """
        平行传输：将向量w从点u平行传输到点v
        """
        if c is None:
            c = self.curvature

        # 使用gyration进行平行传输
        return self.gyration(v, self.mobius_scalar_mul(-1, u, c), w, c)

    def hyperplane_to_poincare(self, a, b, c=None):
        """
        将超平面转换为庞加莱球中的点
        """
        if c is None:
            c = self.curvature

        norm_a = torch.norm(a, p=2, dim=-1, keepdim=True)
        norm_a = torch.clamp(norm_a, min=self.eps)

        # 计算双曲空间中的点
        sqrt_c = torch.sqrt(c)
        denom = sqrt_c * b
        denom = torch.clamp(denom, min=self.eps)

        result = torch.tanh(b / sqrt_c) * a / norm_a
        return result / denom


class HyperbolicLinear(nn.Module):
    """双曲空间线性层"""

    def __init__(self, in_features, out_features, curvature=1.0, use_bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        # 双曲空间
        self.poincare = PoincareBall(dim=in_features, curvature=curvature)

        # 权重参数（在切空间中定义）
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 双曲空间中的输入 [batch_size, in_features]
        Returns:
            双曲空间中的输出 [batch_size, out_features]
        """
        # 将输入映射到切空间
        x_tangent = self.poincare.log_map(x)

        # 在切空间中进行线性变换
        output_tangent = F.linear(x_tangent, self.weight, self.bias)

        # 将输出映射回双曲空间
        output = self.poincare.exp_map(output_tangent)

        return output


class HyperbolicDistance(nn.Module):
    """双曲空间距离层"""

    def __init__(self, num_embeddings, embedding_dim, curvature=1.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 双曲空间
        self.poincare = PoincareBall(dim=embedding_dim, curvature=curvature)

        # 嵌入层（在欧氏空间中初始化）
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-0.001, 0.001)

        # 将嵌入向量投影到双曲空间
        with torch.no_grad():
            self.embedding.weight.copy_(
                self.poincare.exp_map(self.embedding.weight)
            )

    def forward(self, idx1, idx2):
        """
        计算两个嵌入之间的庞加莱距离
        """
        # 获取嵌入向量
        emb1 = self.embedding(idx1)
        emb2 = self.embedding(idx2)

        # 确保向量在单位球内
        emb1 = self.poincare.project_to_ball(emb1)
        emb2 = self.poincare.project_to_ball(emb2)

        # 计算距离
        distance = self.poincare.poincare_distance(emb1, emb2)

        return distance

    def get_embedding(self, idx):
        """获取指定索引的嵌入向量"""
        emb = self.embedding(idx)
        return self.poincare.project_to_ball(emb)


def test_hyperbolic_operations():
    """测试双曲空间操作"""
    print("测试双曲空间操作...")

    # 创建庞加莱球
    poincare = PoincareBall(dim=10, curvature=1.0)

    # 生成随机向量
    x = torch.randn(5, 10) * 0.1
    y = torch.randn(5, 10) * 0.1

    # 测试指数映射和对数映射
    x_hyper = poincare.exp_map(x)
    x_euclid = poincare.log_map(x_hyper)
    print(f"指数映射误差: {torch.norm(x - x_euclid).item():.6f}")

    # 测试Möbius加法
    z = poincare.mobius_add(x_hyper, y_hyper)
    print(f"Möbius加法结果范数: {torch.norm(z, dim=-1).mean().item():.6f}")

    # 测试距离计算
    distance = poincare.poincare_distance(x_hyper, y_hyper)
    print(f"庞加莱距离: {distance.mean().item():.6f}")

    print("测试完成！")


if __name__ == "__main__":
    test_hyperbolic_operations()