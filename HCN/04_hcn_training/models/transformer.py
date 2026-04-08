#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨维度注意力Transformer模块
用于捕获四个认知维度之间的相互关系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DimensionEmbedding(nn.Module):
    """维度嵌入层，用于区分四个认知维度（无序）"""

    def __init__(self, d_model, num_dims=4, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 可学习的维度嵌入 [1, 4, d_model] - 使用正态分布初始化
        self.emb = nn.Parameter(torch.randn(1, num_dims, d_model) * 0.02)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model], seq_len=4 (四个维度)
        """
        # x = x + self.emb[:, :x.size(1), :]  # 广播相加
        x = x + self.emb  # 由于维度固定为4，直接相加
        return self.dropout(x)


class CrossDimensionAttention(nn.Module):
    """跨维度注意力机制"""

    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # 注意力权重可视化
        self.attention_weights = None

    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: [batch_size, 4, d_model] - 四个认知维度的特征
            mask: [batch_size, 4, 4] - 可选的注意力掩码
            return_attention: 是否返回注意力权重
        """
        batch_size = x.size(0)
        seq_len = x.size(1)  # 应该是4

        # 生成Q, K, V
        Q = self.w_q(x)  # [batch_size, 4, d_model]
        K = self.w_k(x)
        V = self.w_v(x)

        # 重塑为多头注意力格式
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # [batch_size, n_heads, 4, d_k]

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # [batch_size, n_heads, 4, 4]

        # 应用掩码（如果有）
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        self.attention_weights = attn_weights.mean(dim=1)  # 平均多头权重

        # 应用注意力权重
        context = torch.matmul(attn_weights, V)
        # [batch_size, n_heads, 4, d_k]

        # 重塑并应用输出投影
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        output = self.w_o(context)
        output = self.dropout(output)

        # 残差连接和层归一化
        output = self.layer_norm(output + x)

        if return_attention:
            return output, self.attention_weights
        return output


class CognitiveInteractionLayer(nn.Module):
    """认知交互层，建模四个维度间的特定关系"""

    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.attention = CrossDimensionAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)

        # 维度特定的交互权重
        self.dimension_interaction = nn.Parameter(torch.ones(4, 4))
        with torch.no_grad():
            # 初始化为更强调对角线（自交互）
            self.dimension_interaction.fill_(0.1)
            self.dimension_interaction.fill_diagonal_(1.0)

    def forward(self, x):
        """
        Args:
            x: [batch_size, 4, d_model]
        Returns:
            output: [batch_size, 4, d_model]
        """
        # 跨维度注意力
        attn_output = self.attention(x)

        # 应用维度交互权重
        # weighted_x = torch.matmul(self.dimension_interaction, x)
        # attn_output = attn_output + weighted_x

        # 前馈网络
        ff_output = self.feed_forward(attn_output)

        # 残差连接和层归一化
        output = self.layer_norm(ff_output + attn_output)

        return output


class CognitiveTransformerEncoder(nn.Module):
    """认知状态Transformer编码器"""

    def __init__(self, d_model=768, n_layers=6, n_heads=8, d_ff=3072, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 维度嵌入（无序）
        self.pos_encoder = DimensionEmbedding(d_model, num_dims=4, dropout=dropout)

        # 多层认知交互
        self.layers = nn.ModuleList([
            CognitiveInteractionLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)

        # 可学习的维度权重
        self.dimension_weights = nn.Parameter(torch.ones(4))

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_all_layers=False):
        """
        Args:
            x: [batch_size, 4, d_model] - 四个认知维度的特征
            return_all_layers: 是否返回所有层的输出
        Returns:
            output: [batch_size, 4, d_model]
            (optional) layer_outputs: list of [batch_size, 4, d_model]
        """
        # 添加位置编码
        x = self.pos_encoder(x)
        x = self.dropout(x)

        layer_outputs = []
        for layer in self.layers:
            x = layer(x)
            if return_all_layers:
                layer_outputs.append(x)

        # 最终层归一化
        x = self.layer_norm(x)

        if return_all_layers:
            return x, layer_outputs
        return x


class CognitiveFusionModule(nn.Module):
    """认知融合模块，将四个维度的信息融合用于分类"""

    def __init__(self, d_model=768, fusion_type='attention'):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'attention':
            # 注意力融合
            self.fusion_attention = nn.MultiheadAttention(
                d_model, num_heads=8, batch_first=True
            )
            self.query_token = nn.Parameter(torch.randn(1, 1, d_model))

        elif fusion_type == 'mlp':
            # MLP融合
            self.fusion_mlp = nn.Sequential(
                nn.Linear(d_model * 4, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(0.1)
            )

        elif fusion_type == 'bilinear':
            # 双线性融合
            self.bilinear = nn.Bilinear(d_model, d_model, d_model)

    def forward(self, x):
        """
        Args:
            x: [batch_size, 4, d_model]
        Returns:
            fused_features: [batch_size, d_model] - 融合后的特征
        """
        if self.fusion_type == 'attention':
            # 使用注意力融合
            batch_size = x.size(0)
            query = self.query_token.expand(batch_size, -1, -1)
            fused, _ = self.fusion_attention(query, x, x)
            return fused.squeeze(1)

        elif self.fusion_type == 'mlp':
            # 展平并通过MLP
            flattened = x.view(x.size(0), -1)
            return self.fusion_mlp(flattened)

        elif self.fusion_type == 'bilinear':
            # 双线性融合（两两组合）
            fused = torch.zeros(x.size(0), x.size(-1), device=x.device)
            for i in range(4):
                for j in range(i+1, 4):
                    fused = fused + self.bilinear(x[:, i], x[:, j])
            return fused

        else:
            # 简单平均
            return x.mean(dim=1)


def test_transformer():
    """测试Transformer模块"""
    print("测试认知Transformer...")

    # 创建模型
    encoder = CognitiveTransformerEncoder(d_model=512, n_layers=2, n_heads=8)
    fusion = CognitiveFusionModule(d_model=512, fusion_type='attention')

    # 生成测试数据
    batch_size = 4
    x = torch.randn(batch_size, 4, 512)  # 4个认知维度

    # 前向传播
    encoded = encoder(x)
    fused = fusion(encoded)

    print(f"输入形状: {x.shape}")
    print(f"编码后形状: {encoded.shape}")
    print(f"融合后形状: {fused.shape}")

    print("测试完成！")


if __name__ == "__main__":
    test_transformer()