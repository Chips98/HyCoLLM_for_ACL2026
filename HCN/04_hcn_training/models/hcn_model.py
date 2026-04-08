#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HCN (Hyper-Cognition Net) 主模型实现
结合双曲空间和Transformer的认知状态分类模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .hyperbolic import PoincareBall, HyperbolicLinear
from .transformer import CognitiveTransformerEncoder, CognitiveFusionModule


class HCNModel(nn.Module):
    """
    HCN (Hyper-Cognition Net) 主模型
    架构：
    1. 输入层：接收4个维度的特征 [B, 4, D]
    2. 位置编码：区分4个认知维度
    3. 双曲空间映射：将特征映射到双曲空间
    4. Transformer编码器：跨维度注意力机制
    5. 映射回欧氏空间
    6. 多任务分类头：分别预测4个维度
    """

    def __init__(self,
                 input_dim=768,
                 hidden_dim=512,
                 n_layers=6,
                 n_heads=8,
                 dropout=0.1,
                 curvature=1.0,
                 learnable_curvature=False,
                 fusion_type='attention',
                 hyperbolic_scale=0.05,
                 use_hyperbolic=True,      # 消融实验参数：是否使用双曲空间
                 use_transformer=True):    # 消融实验参数：是否使用Transformer交互
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hyperbolic_scale = hyperbolic_scale

        # 消融实验标志
        self.use_hyperbolic = use_hyperbolic
        self.use_transformer = use_transformer

        # 特征降维（如果需要）
        if input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.input_norm = nn.LayerNorm(hidden_dim)
        else:
            self.input_projection = nn.Identity()

        # 双曲空间层（消融实验：欧氏模式下使用Identity）
        if self.use_hyperbolic:
            self.hyperbolic_layer = PoincareBall(
                dim=hidden_dim,
                curvature=curvature,
                learnable_curvature=learnable_curvature
            )
        else:
            # 欧氏模式：使用简单的线性层或恒等映射
            self.hyperbolic_layer = nn.Identity()

        # 认知Transformer编码器（消融实验：无Transformer交互时使用Identity）
        if self.use_transformer:
            self.transformer_encoder = CognitiveTransformerEncoder(
                d_model=hidden_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout
            )
        else:
            # 无Transformer交互：直接传递，不做跨维度交互
            self.transformer_encoder = nn.Identity()

        # 认知融合模块（可选）
        self.use_fusion = True
        self.fusion_module = CognitiveFusionModule(
            d_model=hidden_dim,
            fusion_type=fusion_type
        )

        # 分类头 - 精确匹配实际数据的标签范围
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 9)    # 数据标签范围 0-8 = 9类
        )

        self.thinking_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 8)    # 数据标签范围 0-7 = 8类
        )

        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 7)    # 数据标签范围 0-6 = 7类
        )

        self.stance_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)    # 数据标签范围 0-2 = 3类
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用更保守的初始化
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

                # 立即检查初始化后的权重
                if not torch.isfinite(m.weight).all():
                    raise ValueError(f"权重初始化后包含无效值: {m}")
                if m.bias is not None and not torch.isfinite(m.bias).all():
                    raise ValueError(f"偏置初始化后包含无效值: {m}")

    def forward(self, x, return_features=False):
        """nor
        前向传播
        Args:
            x: [batch_size, 4, input_dim] - 四个认知维度的特征
            return_features: 是否返回中间特征用于可视化
        Returns:
            logits: dict containing logits for each dimension
            features: dict containing intermediate features (if return_features=True)
        """
        # 检查输入的有效性
        if not torch.isfinite(x).all():
            raise ValueError(f"模型输入包含无效值! min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}, std={x.std():.6f}")

        batch_size = x.size(0)

        # 1. 输入投影（如果需要）
        x = self.input_projection(x)  # [B, 4, hidden_dim]
        if hasattr(self, 'input_norm'):
            x = self.input_norm(x)
        # 双曲缩放因子，控制输入到双曲空间的幅度
        x = x * self.hyperbolic_scale

        # 2. 双曲空间映射逻辑修改（消融实验）
        if self.use_hyperbolic:
            # 双曲模式：映射到双曲空间并映射回切空间
            x_hyper = self.hyperbolic_layer.exp_map(x)  # [B, 4, hidden_dim]

            # 检查双曲空间映射的有效性
            if not torch.isfinite(x_hyper).all():
                raise ValueError(f"双曲空间映射后包含无效值! min={x_hyper.min():.6f}, max={x_hyper.max():.6f}, mean={x_hyper.mean():.6f}")

            # 必须映射回欧氏空间（切空间）才能使用Transformer
            x_tangent = self.hyperbolic_layer.log_map(x_hyper)  # [B, 4, hidden_dim]
        else:
            # 欧氏模式：直接使用原始特征（或经过一个LayerNorm）
            x_hyper = x
            x_tangent = x

        # 检查切空间映射的有效性（仅双曲模式）
        # if not torch.isfinite(x_tangent).all():
        #     raise ValueError(f"切空间映射后包含无效值! min={x_tangent.min():.6f}, max={x_tangent.max():.6f}, mean={x_tangent.mean():.6f}")

        # 3. Transformer编码逻辑修改（消融实验）
        if self.use_transformer:
            # 有Transformer交互：使用跨维度注意力机制
            x_encoded = self.transformer_encoder(x_tangent)  # [B, 4, hidden_dim]
        else:
            # 无Transformer交互：直接跳过交互，各维度独立处理
            x_encoded = x_tangent # 直接跳过交互

        # 4. 分别从四个维度提取特征进行分类
        emotion_features = x_encoded[:, 0, :]  # [B, hidden_dim]
        thinking_features = x_encoded[:, 1, :]
        intent_features = x_encoded[:, 2, :]
        stance_features = x_encoded[:, 3, :]

        # 5. 分类预测
        logits = {
            'emotion': self.emotion_classifier(emotion_features),
            'thinking': self.thinking_classifier(thinking_features),
            'intent': self.intent_classifier(intent_features),
            'stance': self.stance_classifier(stance_features)
        }

        if return_features:
            features = {
                'input_original': x,
                'input_hyperbolic': x_hyper,
                'input_tangent': x_tangent,
                'encoded_euclidean': x_encoded,
                'emotion_features': emotion_features,
                'thinking_features': thinking_features,
                'intent_features': intent_features,
                'stance_features': stance_features
            }

            if self.use_fusion:
                # 生成融合特征
                fused_features = self.fusion_module(x_encoded)
                features['fused'] = fused_features

            return logits, features

        return logits


class HCNWithMultiTaskLearning(HCNModel):
    """带多任务学习的HCN模型"""

    def __init__(self,
                 task_weights=[1.0, 1.0, 1.0, 1.0],
                 use_uncertainty_weighting=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.task_weights = task_weights
        self.use_uncertainty_weighting = use_uncertainty_weighting

        if use_uncertainty_weighting:
            # 可学习的任务权重（基于不确定性）
            self.log_vars = nn.Parameter(torch.zeros(4))

    def compute_task_weights(self):
        """计算任务权重"""
        if self.use_uncertainty_weighting:
            # 基于不确定性的权重：1 / (2 * sigma^2)
            weights = torch.exp(-self.log_vars)
            return weights
        else:
            return torch.tensor(self.task_weights)

    def get_attention_weights(self, x):
        """获取注意力权重用于可视化"""
        # 通过transformer获取注意力权重
        # 这需要修改transformer模块以返回注意力权重
        pass


class HierarchicalHCN(HCNModel):
    """分层HCN模型，考虑认知维度的层次关系"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 思维类型的层次约束
        self.thinking_hierarchy = nn.Parameter(torch.randn(2, self.hidden_dim))

        # 认知维度间的关系建模
        self.dimension_relation = nn.Parameter(torch.randn(4, 4))

    def forward(self, x, return_features=False):
        """前向传播，添加层次约束"""
        # 调用父类的前向传播
        logits, features = super().forward(x, return_features=True)

        # 添加层次约束（在训练时通过损失函数应用）
        if self.training:
            # 这里可以添加层次约束的计算
            pass

        if return_features:
            return logits, features
        return logits


def test_hcn_model():
    """测试HCN模型"""
    print("测试HCN模型...")

    # 创建模型
    model = HCNModel(
        input_dim=768,
        hidden_dim=512,
        n_layers=2,
        n_heads=8,
        curvature=1.0
    )

    # 生成测试数据
    batch_size = 4
    x = torch.randn(batch_size, 4, 768)

    # 前向传播
    logits, features = model(x, return_features=True)

    print(f"输入形状: {x.shape}")
    print(f"情感logits形状: {logits['emotion'].shape}")
    print(f"思维logits形状: {logits['thinking'].shape}")
    print(f"意图logits形状: {logits['intent'].shape}")
    print(f"立场logits形状: {logits['stance'].shape}")
    print(f"编码特征形状: {features['encoded_euclidean'].shape}")

    # 测试推理
    model.eval()
    with torch.no_grad():
        logits = model(x)
        predictions = {
            key: torch.argmax(F.softmax(logits[key], dim=-1), dim=-1)
            for key in logits
        }
        print(f"\n预测结果:")
        for key, pred in predictions.items():
            print(f"  {key}: {pred}")

    print("\n模型参数量:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("测试完成！")


if __name__ == "__main__":
    test_hcn_model()