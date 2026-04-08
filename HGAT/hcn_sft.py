#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HCN-SFT: 双曲引导的认知指令微调 (Hyperbolic-Guided Cognitive Instruction Tuning)
完整实现研究方案中的几何引导和表征对齐双重机制

主要组件：
1. CognitiveProjector: 特征投影器 v_cog -> Soft Prompts
2. AlignmentProjector: 对齐投影器 LLM Hidden -> Tangent Space
3. HyperbolicGuidedLLM: HCN-LLM融合模型
4. HCNAwareDataCollator: 智能数据整理器，正确处理assistant部分masking
5. HCNTuner: 自定义Trainer，联合优化SFT loss + SCT loss
"""

import argparse
import os
import sys
import json
import copy
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import warnings

# 环境变量设置
os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)

# 尝试导入DataCollatorForCompletionOnlyLM，如果不存在则跳过
try:
    from transformers import DataCollatorForCompletionOnlyLM
    print("✓ DataCollatorForCompletionOnlyLM 导入成功")
except ImportError:
    print("⚠️ DataCollatorForCompletionOnlyLM 不可用，将使用自定义实现")
    DataCollatorForCompletionOnlyLM = None

from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 动态导入HCN模块的函数
def setup_hcn_imports(hcn_module_dir):
    """动态导入HCN模块，修复相对导入问题"""
    import importlib.util

    # HCN模块文件路径
    hcn_model_path = os.path.join(hcn_module_dir, "hcn_model.py")
    hyperbolic_path = os.path.join(hcn_module_dir, "hyperbolic.py")
    transformer_path = os.path.join(hcn_module_dir, "transformer.py")

    # 检查文件是否存在
    if not os.path.exists(hcn_model_path):
        raise FileNotFoundError(f"HCN模型文件不存在: {hcn_model_path}")

    print(f"使用HCN模块目录: {hcn_module_dir}")

    # 先导入依赖模块并创建包结构
    import types
    hcn_package = types.ModuleType("hcn_package")
    sys.modules["hcn_package"] = hcn_package

    # 导入hyperbolic模块作为包的子模块
    if os.path.exists(hyperbolic_path):
        spec = importlib.util.spec_from_file_location("hcn_package.hyperbolic", hyperbolic_path)
        hyperbolic_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hyperbolic_module)
        sys.modules["hcn_package.hyperbolic"] = hyperbolic_module
        hcn_package.hyperbolic = hyperbolic_module
        print(f"✓ hyperbolic模块导入成功")

    # 导入transformer模块作为包的子模块
    if os.path.exists(transformer_path):
        spec = importlib.util.spec_from_file_location("hcn_package.transformer", transformer_path)
        transformer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transformer_module)
        sys.modules["hcn_package.transformer"] = transformer_module
        hcn_package.transformer = transformer_module
        print(f"✓ transformer模块导入成功")

    # 读取HCN模型代码并修复相对导入
    with open(hcn_model_path, 'r', encoding='utf-8') as f:
        hcn_code = f.read()

    # 替换相对导入为绝对导入
    hcn_code = hcn_code.replace("from .hyperbolic import", "from hcn_package.hyperbolic import")
    hcn_code = hcn_code.replace("from .transformer import", "from hcn_package.transformer import")

    # 执行修复后的代码
    hcn_module = types.ModuleType("hcn_package.hcn_model")
    sys.modules["hcn_package.hcn_model"] = hcn_module
    hcn_package.hcn_model = hcn_module

    exec(hcn_code, hcn_module.__dict__)

    HCNModel = hcn_module.HCNModel
    print("✓ HCN模型导入成功")
    return HCNModel

# 全局变量，将在main函数中初始化
HCNModel = None

# ==================== 1. 复用 SFT 的数据处理逻辑 ====================

class CognitivePromptBuilderSFT:
    """
    完全移植自 cognitive_sft.py 的提示词构建器，确保训练与推理一致
    """
    def __init__(self, labels_config):
        self.labels = labels_config

    def build_prompt(self, history, current_item):
        # 构建各个部分，逻辑与推理保持100%一致
        task_desc = self._build_task_description()
        label_scope = self._build_label_scope()
        history_sec = self._build_history_section(history)
        current_sec = self._build_current_section(current_item)
        notes = self._build_notes()
        output_fmt = self._build_output_format()

        return f"""{task_desc}

{label_scope}

{history_sec}

# 目标用户的当前对话上下文
{current_sec}

# 注意事项
{notes}

# 输出格式
{output_fmt}

请严格按照以上格式输出分析结果。"""

    def _build_task_description(self):
        return """# 任务描述
你是一个社会心理学计算专家，正在进行严格的数据标注任务。
你的任务是根据给定的标签体系，判断用户的认知状态。

**重要警告**：
- **绝对禁止**创造任何新的标签或词汇
- **必须**且只能从给定的选项列表中选择一个最匹配的标签
- **严禁**输出"讽刺"、"怀疑"、"反讽"、"嘲讽"等任何不在列表中的词汇
- 如果感到模糊，请选择语义最接近的现有标签，而不是发明新词
- 每个维度只能选择一个选项，不能多选也不能留空

你是一个专业的认知状态分析专家。你的任务是分析用户在社交媒体对话中表现出的认知状态，从四个维度进行判断：

1. **情感（Emotion）**: 识别用户当前的情绪状态
2. **立场（Stance）**: 判断用户在话题中的立场倾向
3. **思维（Thinking）**: 分析用户的思维方式和价值判断
4. **意图（Intent）**: 识别用户发言的主要目的

你需要仔细分析用户的发言内容，理解其背后的情感、立场、思维模式和意图，并**严格**从给定的标签中选择最符合的选项。"""

    def _build_label_scope(self):
        prompt_parts = ["# 标签范围\n", "请从以下标签中选择最符合的选项：\n"]

        prompt_parts.append("## 情感（Emotion）")
        for label, info in self.labels["emotion"].items():
            prompt_parts.append(f"- {label}: {info['description']}")

        prompt_parts.append("\n## 立场（Stance）")
        for label, info in self.labels["stance"].items():
            prompt_parts.append(f"- {label}: {info['description']}")

        prompt_parts.append("\n## 思维（Thinking）")
        prompt_parts.append("注意：思维维度仅使用以下价值判断标签：")
        thinking_vals = self.labels.get("thinking", {}).get("values", {})
        for label, info in thinking_vals.items():
            prompt_parts.append(f"- {label}: {info['description']}")

        prompt_parts.append("\n## 意图（Intent）")
        for label, info in self.labels["intent"].items():
            prompt_parts.append(f"- {label}: {info['description']}")

        return "\n".join(prompt_parts)

    def _build_history_section(self, history):
        if not history:
            return "# 目标用户的历史发言\n无历史发言记录"

        prompt_parts = ["# 目标用户的历史发言（含context_post和target_post）"]
        for i, turn in enumerate(history, 1):
            prompt_parts.append(f"\n第{i}轮发言:")
            if "context_post" in turn and turn["context_post"]:
                prompt_parts.append(f"上下文: {turn['context_post']}")
            prompt_parts.append(f"发言: {turn['target_post']}")
        return "\n".join(prompt_parts)

    def _build_current_section(self, current):
        prompt_parts = []
        prompt_parts.append("## 原始帖子")
        prompt_parts.append(f"```\n{current.get('original_post', '')}\n```")
        prompt_parts.append("\n## 上下文帖子")
        prompt_parts.append(f"```\n{current.get('context_post', '')}\n```")
        prompt_parts.append("\n## 当前发言（待分析的内容）")
        prompt_parts.append(f"```\n{current.get('target_post', '')}\n```")
        prompt_parts.append("\n**注意：请重点分析'当前发言'中反映的用户认知状态。**")
        return "\n".join(prompt_parts)

    def _build_notes(self):
        notes = [
            "1. 仔细分析用户的发言内容，理解其言外之意",
            "2. 考虑历史对话的上下文，理解用户的立场和情感变化",
            "3. 思维维度要区分'直觉型'和'分析型'，并进一步判断具体的思维价值",
            "4. 每个维度只能选择一个最符合的标签，不能多选也不能留空",
            "5. 如果无法确定，选择最可能的选项，不要留空 no think",
            "**严重警告**：",
            "6. **绝对禁止**输出'讽刺'、'怀疑'、'反讽'、'嘲讽'、'开心'、'难过'等任何不在标签列表中的词汇",
            "7. **必须**严格从标签列表中选择，即使模型认为有更合适的词汇也不得使用",
            "8. 如果遇到讽刺意图，请归类为'分歧与冲突'或'情感表达'等现有标签",
            "9. 违反输出限制将导致结果无效" 
        ]
        return "\n".join(notes)

    def _build_output_format(self):
        # 动态获取标签Key
        emotion_labels = list(self.labels["emotion"].keys())
        stance_labels = list(self.labels["stance"].keys())
        thinking_labels = list(self.labels.get("thinking", {}).get("values", {}).keys())
        intent_labels = list(self.labels["intent"].keys())

        return f"""请严格按照以下格式输出，每行一个维度：

情感: 从 [{', '.join(emotion_labels)}] 中选择唯一一项
立场: 从 [{', '.join(stance_labels)}] 中选择唯一一项
思维: 从 [{', '.join(thinking_labels)}] 中选择唯一一项
意图: 从 [{', '.join(intent_labels)}] 中选择唯一一项

示例：
情感: 厌恶
立场: 支持中方
思维: 主观评价
意图: 表达主张"""

def preprocess_dataset_with_history(raw_dataset, max_history_rounds=10):
    """
    与 cognitive_sft.py 保持完全一致的预处理
    关键：必须保证这里的排序逻辑与提取特征时的数据顺序一致！
    """
    print("正在按用户分组并构建历史上下文 (必须与特征提取时的顺序一致)...")
    data_list = [item for item in raw_dataset]
    user_groups = {}
    for item in data_list:
        uid = item['user_id']
        if uid not in user_groups: user_groups[uid] = []
        user_groups[uid].append(item)

    processed_samples = []
    for uid, items in user_groups.items():
        # 排序逻辑必须一致
        items.sort(key=lambda x: int(x['sub_id']) if isinstance(x['sub_id'], (int, str)) and str(x['sub_id']).isdigit() else x.get('timestep', 0))
        history = []
        for current_item in items:
            sample = copy.deepcopy(current_item)
            sample['history'] = copy.deepcopy(history)
            processed_samples.append(sample)
            history.append({
                "context_post": current_item.get("context_post", ""),
                "target_post": current_item.get("target_post", "")
            })
            if len(history) > max_history_rounds:
                history = history[-max_history_rounds:]
    return processed_samples

# ==================== 2. 模型架构：HCN-LLM 融合 ====================

class CognitiveProjector(nn.Module):
    """
    特征投影器: v_cog (HCN) -> Soft Prompts (LLM)
    实现论文中的公式: e_prior = P(v_cog) ∈ R^(L × d_model)

    修复：添加输出归一化和数值限制，防止梯度爆炸和NaN损失

    参数:
    - input_dim: HCN认知状态向量维度
    - output_dim: LLM隐藏层维度
    - num_tokens: Soft prompt token数量 (L)
    """
    def __init__(self, input_dim, output_dim, num_tokens=4, dropout=0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim

        # 两层MLP投影，GELU激活，适应双曲特征的分布特性
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),  # 添加中间层LayerNorm
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim * num_tokens),
            nn.Dropout(dropout)
        )

        # 输出层的LayerNorm，保持分布稳定
        self.output_norm = nn.LayerNorm(output_dim)

        # 新增：Tanh激活函数，强制限制输出幅度在[-1,1]之间
        self.final_act = nn.Tanh()

        # 新增：可学习的缩放因子，用于精细调整输出范围，初始化为较小值
        self.learnable_scale = nn.Parameter(torch.tensor(0.1))  # 减小初始值

    def forward(self, x):
        """
        Args:
            x: [B, input_dim] HCN认知状态向量
        Returns:
            out: [B, num_tokens, output_dim] Soft prompts，数值范围限制在合理区间
        """
        # 输入检查：确保输入没有NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("⚠️ CognitiveProjector 输入包含 NaN/Inf，使用零输入")
            x = torch.zeros_like(x)

        out = self.net(x)
        out = out.view(x.size(0), self.num_tokens, self.output_dim)

        # LayerNorm归一化
        out = self.output_norm(out)

        # Tanh激活函数强制限制幅度，防止数值爆炸
        out = self.final_act(out)

        # 可学习缩放因子进行精细调整，限制缩放范围防止过大
        scale = torch.clamp(self.learnable_scale, min=-1.0, max=1.0)
        out = out * scale

        # 最终数值检查
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("⚠️ CognitiveProjector 输出包含 NaN/Inf，重置为零")
            out = torch.zeros_like(out)

        return out

class AlignmentProjector(nn.Module):
    """
    对齐投影器: LLM Hidden -> Tangent Space
    实现论文中的公式: v_sem = W_align(h_last^LLM)

    参数:
    - input_dim: LLM隐藏层维度
    - output_dim: 切空间维度（与HCN输出维度一致）
    """
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: [B, input_dim] LLM最后一层隐藏状态
        Returns:
            out: [B, output_dim] 投影到切空间的语义向量
        """
        return self.net(x)

class HyperbolicGuidedLLM(nn.Module):
    """
    HCN-SFT核心融合模型
    实现论文中的双重机制：几何引导 + 表征对齐
    支持多种消融研究模式
    """
    def __init__(self, llm_model, hcn_model, llm_hidden_size, hcn_hidden_size,
                 num_soft_tokens=None, lambda_sct=None, dropout=0.1,
                 enable_soft_prompt=True, enable_sct_loss=True,
                 enable_hcn_fusion=True, enable_cosine_alignment=True,
                 enable_alignment_projector=True):
        super().__init__()
        self.llm = llm_model
        self.hcn = hcn_model

        # 修复：使用可配置的默认值，避免硬编码
        self.num_soft_tokens = num_soft_tokens if num_soft_tokens is not None else 4
        self.lambda_sct = lambda_sct if lambda_sct is not None else 0.3  # SCT损失权重

        # 消融研究标志
        self.enable_soft_prompt = enable_soft_prompt
        self.enable_sct_loss = enable_sct_loss
        self.enable_hcn_fusion = enable_hcn_fusion
        self.enable_cosine_alignment = enable_cosine_alignment
        self.enable_alignment_projector = enable_alignment_projector

        # 冻结HCN参数，只作为特征提取器
        for param in self.hcn.parameters():
            param.requires_grad = False
        self.hcn.eval()

        # 根据消融标志初始化组件
        if self.enable_soft_prompt:
            # 特征注入投影器
            self.input_projector = CognitiveProjector(
                hcn_hidden_size, llm_hidden_size, num_soft_tokens, dropout
            )
        else:
            self.input_projector = None

        if self.enable_alignment_projector:
            # 表征对齐投影器
            self.align_projector = AlignmentProjector(
                llm_hidden_size, hcn_hidden_size, dropout
            )
        else:
            self.align_projector = None

    def gradient_checkpointing_enable(self, **kwargs):
        """启用梯度检查点以节省显存"""
        if hasattr(self.llm, 'gradient_checkpointing_enable'):
            self.llm.gradient_checkpointing_enable(**kwargs)

    def get_hcn_cognitive_anchor(self, hcn_features):
        """
        从HCN获取认知锚点 v_cog (切空间向量)

        根据方法论，v_cog 是在切空间 (Tangent Space) 中通过 Transformer 融合得到的认知状态向量。
        HCNModel 的前向传播会：
        1. exp_map: 输入特征 -> 双曲空间
        2. Transformer: 在切空间中进行特征融合 (HCN内部已自动log_map回切空间)
        3. 返回切空间中的融合特征 v_cog

        消融模式支持：
        - enable_hcn_fusion: 使用融合特征 vs 各维度平均

        Args:
            hcn_features: [B, 4, D] 输入特征
        Returns:
            v_cog: [B, hcn_dim] 切空间中的认知状态向量，可直接用于余弦相似度计算
        """
        with torch.no_grad():
            _, hcn_out_dict = self.hcn(hcn_features, return_features=True)

            if self.enable_hcn_fusion and 'fused' in hcn_out_dict:
                # 使用融合特征作为认知锚点
                v_cog = hcn_out_dict['fused']
            else:
                # 使用各维度在切空间的平均（消融模式）
                encoded = hcn_out_dict['encoded_euclidean']  # [B, 4, hcn_dim] 切空间向量
                v_cog = encoded.mean(dim=1)  # [B, hcn_dim]

        return v_cog

    def extract_llm_last_hidden(self, hidden_states, attention_mask):
        """
        提取LLM每个序列的最后一个有效token的隐藏状态
        Args:
            hidden_states: [B, L, D] 最后一层隐藏状态
            attention_mask: [B, L] 注意力掩码
        Returns:
            last_hidden: [B, D] 每个序列最后一个有效token的隐藏状态
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 计算每个序列的实际长度（考虑软prompt）
        sequence_lengths = attention_mask.sum(dim=1) - 1  # 最后一个有效token的索引

        # 确保索引有效
        sequence_lengths = torch.clamp(sequence_lengths, 0, seq_len - 1)

        # 提取最后一个有效token的隐藏状态
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, sequence_lengths]

        return last_hidden

    def compute_sct_loss(self, v_sem, v_cog):
        """
        计算语义-认知对齐损失 (SCT Loss) - 数值稳定版本
        实现论文公式: L_SCT = 1 - CosSim(v_sem, v_cog)

        Args:
            v_sem: [B, D] LLM语义向量
            v_cog: [B, D] HCN认知锚点
        Returns:
            sct_loss: 标量损失值
        """
        # 【关键修改】强制转为 float32 计算相似度，避免 fp16 下溢
        v_sem_f32 = v_sem.float()
        v_cog_f32 = v_cog.float()

        # 计算余弦相似度，添加 epsilon 防止除零
        v_sem_norm = F.normalize(v_sem_f32, p=2, dim=-1, eps=1e-8)
        v_cog_norm = F.normalize(v_cog_f32, p=2, dim=-1, eps=1e-8)

        cos_sim = (v_sem_norm * v_cog_norm).sum(dim=-1)
        sct_loss = 1.0 - cos_sim.mean()

        # 检查数值稳定性
        if torch.isnan(sct_loss) or torch.isinf(sct_loss):
            print("⚠️ SCT Loss 产生 NaN/Inf，返回零损失")
            return torch.tensor(0.0, device=v_sem.device, dtype=v_sem.dtype)

        return sct_loss.to(v_sem.dtype)  # 转回原始dtype保持一致性

    def forward(self, input_ids, attention_mask=None, labels=None, hcn_features=None, **kwargs):
        """
        前向传播 - 支持消融研究模式

        Args:
            input_ids: [B, L] 输入token ids
            attention_mask: [B, L] 注意力掩码
            labels: [B, L] 训练标签
            hcn_features: [B, 4, D] HCN输入特征
        Returns:
            dict: 包含loss和中间结果
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # 1. 获取HCN认知锚点 (几何先验)
        v_cog = self.get_hcn_cognitive_anchor(hcn_features)  # [B, hcn_dim]

        # 2. 准备LLM输入 - 根据消融模式决定是否使用软提示
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, L, llm_dim]

        if self.enable_soft_prompt and self.input_projector is not None:
            # 生成软提示并拼接
            soft_prompts = self.input_projector(v_cog)  # [B, num_tokens, llm_dim]
            inputs_embeds = torch.cat([soft_prompts, inputs_embeds], dim=1)  # [B, L+num_tokens, llm_dim]

            # 扩展注意力掩码
            if attention_mask is not None:
                prompt_mask = torch.ones(batch_size, self.num_soft_tokens, device=device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)  # [B, L+num_tokens]

            # 扩展标签（软提示部分不计算损失）
            if labels is not None:
                ignore_labels = torch.full((batch_size, self.num_soft_tokens), -100, device=device, dtype=labels.dtype)
                labels = torch.cat([ignore_labels, labels], dim=1)  # [B, L+num_tokens]

        # 3. LLM前向传播
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )

        # 4. 计算SCT对齐损失 (如果启用)
        sct_loss = torch.tensor(0.0, device=device)
        v_sem = None

        if self.enable_sct_loss and self.enable_alignment_projector and self.align_projector is not None:
            # 提取最后一层隐藏状态
            last_hidden_states = outputs.hidden_states[-1]  # [B, L+num_tokens, llm_dim]

            # 获取每个序列最后一个有效token的隐藏状态
            llm_last_hidden = self.extract_llm_last_hidden(last_hidden_states, attention_mask)  # [B, llm_dim]

            # 投影到切空间
            v_sem = self.align_projector(llm_last_hidden)  # [B, hcn_dim]

            # 计算SCT损失
            if self.enable_cosine_alignment:
                sct_loss = self.compute_sct_loss(v_sem, v_cog)
            else:
                # 消融模式：使用MSE损失替代余弦相似度
                sct_loss = F.mse_loss(v_sem, v_cog)

        # 5. 联合损失
        total_loss = outputs.loss + self.lambda_sct * sct_loss

        return {
            "loss": total_loss,
            "logits": outputs.logits,
            "sft_loss": outputs.loss.detach() if hasattr(outputs.loss, 'detach') else outputs.loss,
            "sct_loss": sct_loss.detach(),
            "v_cog": v_cog.detach(),  # 用于可视化
            "v_sem": v_sem.detach() if v_sem is not None else None  # 用于可视化
        }

# ==================== 3. 数据集与 Collator ====================

class HybridCognitiveDataset(Dataset):
    def __init__(self, data_samples, features_path, tokenizer, prompt_builder, max_length=2048):
        self.data = data_samples
        self.tokenizer = tokenizer
        self.prompt_builder = prompt_builder
        self.max_length = max_length
        
        # 加载 HCN 特征 (必须与 data_samples 顺序一一对应)
        print(f"加载 HCN 特征: {features_path}")
        features_data = torch.load(features_path, map_location='cpu')
        self.hcn_features = features_data['features'] # Tensor [N, 4, D]
        
        if len(self.data) != len(self.hcn_features):
            print(f"⚠️ 警告: 样本数 {len(self.data)} 与 特征数 {len(self.hcn_features)} 不匹配!")
            # 严格对齐检查：这里假设预处理流程是严格受控的
            # 在实际工程中，建议在生成特征时保存 sample_id 以便校验
            min_len = min(len(self.data), len(self.hcn_features))
            self.data = self.data[:min_len]
            self.hcn_features = self.hcn_features[:min_len]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. 构建 Prompt (完全复用 SFT 逻辑)
        user_content = self.prompt_builder.build_prompt(item.get('history', []), item)

        # 构建标签部分
        labels_dict = item.get('cognitive_labels', {})
        thinking_val = labels_dict.get('thinking_value', labels_dict.get('thinking', '未知'))
        assistant_content = f"情感: {labels_dict.get('emotion', '未知')}\n" \
                            f"立场: {labels_dict.get('stance', '未知')}\n" \
                            f"思维: {thinking_val}\n" \
                            f"意图: {labels_dict.get('intent', '未知')}"

        # 2. Tokenize (使用 Chat Template)
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False, # 在 Collator 中 Padding
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encodings.input_ids[0]
        attention_mask = encodings.attention_mask[0]
        labels = input_ids.clone() # 修复：后续在 Collator 中正确处理 Masking

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "hcn_features": self.hcn_features[idx]
        }

class HCNAwareDataCollator:
    """
    HCN感知的数据整理器 (修复版 V3 - 强力序列匹配)
    直接在 Token ID 层面搜索锚点，无视解码问题，完美兼容 Qwen3 的 Thinking 模式
    """
    def __init__(self, tokenizer, pad_to_multiple_of=None, debug_samples=3):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.debug_samples = debug_samples  # 需要详细打印的样本数量

        # 1. 预计算锚点序列 IDs: <|im_start|>assistant
        # Qwen2.5/3 的 <|im_start|> ID 是 151644
        im_start_id = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
        if im_start_id is None: # 防御性回退
            im_start_id = self.tokenizer.bos_token_id

        # 'assistant' 的 ID (查表得 77091)
        # 我们不加特殊符，只编码文本，确保拿到纯粹的ID
        assistant_ids = self.tokenizer.encode("assistant", add_special_tokens=False)

        # 组合成绝对锚点: [151644, 77091]
        self.response_template_ids = [im_start_id] + assistant_ids

        # 结束符 IDs
        self.im_end_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        self.eos_id = self.tokenizer.eos_token_id

        print(f"🔧 [Collator] 锚点序列 IDs: {self.response_template_ids}")

        # 计数器，用于控制调试打印数量
        self.debug_counter = 0
        
    def __call__(self, features):
        batch = {}

        # 1. 提取 Input IDs
        input_ids = [f['input_ids'] for f in features]
        
        # 2. Padding
        batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        
        attention_mask = [f['attention_mask'] for f in features]
        batch['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        
        # 3. 核心修复：基于序列匹配的 Label Masking
        # 默认全设为 -100 (忽略所有)
        labels = batch['input_ids'].clone()
        labels[:] = -100
        
        for i, feature_input_ids in enumerate(input_ids):
            # 转为 list 方便搜索
            if isinstance(feature_input_ids, torch.Tensor):
                seq = feature_input_ids.tolist()
            else:
                seq = feature_input_ids
                
            # 寻找所有 assistant 回复的起始位置
            # 搜索 [151644, 77091] 出现的所有位置
            matches = []
            n = len(seq)
            m = len(self.response_template_ids)
            for j in range(n - m + 1):
                if seq[j:j+m] == self.response_template_ids:
                    matches.append(j)
            
            if not matches:
                # 依然没找到？打印前20个token帮助调试
                print(f"⚠️ 样本 {i} 未找到锚点! 前20个token: {seq[:20]}")
                continue

            # 对每一轮回复进行 Unmask (恢复 Label)
            for start_idx in matches:
                # 有效内容从 "assistant" 之后开始 (包含可能的 \n 和 <think>)
                # 我们希望模型学习 <think>，所以从这里开始计算 Loss 是完全正确的
                content_start = start_idx + m
                
                # 寻找这一轮的结束位置 (<|im_end|>)
                end_idx = n
                if self.im_end_id is not None:
                    try:
                        # 找到当前回复之后的第一个 im_end
                        relative_end = seq[content_start:].index(self.im_end_id)
                        end_idx = content_start + relative_end + 1 # +1 包含 im_end 本身
                    except ValueError:
                        pass
                
                # 恢复这段区域的 label
                # 注意：batch['input_ids'] 已经 pad 过了，我们要操作对应的位置
                labels[i, content_start:end_idx] = batch['input_ids'][i, content_start:end_idx]
        
        batch['labels'] = labels

        # 4. 堆叠 HCN 特征
        batch['hcn_features'] = torch.stack([f['hcn_features'] for f in features])

        # 5. 详细打印前几个样本的匹配内容
        if self.debug_counter < self.debug_samples:
            for i in range(min(len(input_ids), self.debug_samples - self.debug_counter)):
                self._debug_detailed_matching(input_ids[i], labels[i], i, self.debug_counter + i)
            self.debug_counter += min(len(input_ids), self.debug_samples - self.debug_counter)

        return batch

    def _debug_detailed_matching(self, input_ids, labels, sample_idx, debug_idx):
        """简化版本：检查LLM回复的关键信息"""
        # 获取有效token的位置
        valid_positions = (labels != -100).nonzero(as_tuple=True)[0]

        if len(valid_positions) == 0:
            print(f"  ❌ 样本 {debug_idx+1}: 未找到有效训练内容")
            return

        # 提取完整回复内容
        full_reply_ids = input_ids[valid_positions[0]:valid_positions[-1]+1]
        full_reply_text = self.tokenizer.decode(full_reply_ids)

        # 检查认知标签格式
        expected_dimensions = ["情感", "立场", "思维", "意图"]
        found_dimensions = [dim for dim in expected_dimensions if dim in full_reply_text]

        # 简化输出
        print(f"  ✅ 样本 {debug_idx+1}: 找到 {len(found_dimensions)}/4 个认知维度, 有效token数 {len(valid_positions)}")

        # 只在有缺失维度时才警告
        if len(found_dimensions) < 4:
            missing = [d for d in expected_dimensions if d not in found_dimensions]
            print(f"     ⚠️ 缺失维度: {missing}")

    def _debug_labels(self, input_ids, labels):
        print("\n🔍 [Collator Debug] 检查 Label Masking (V3):")
        valid_count = (labels != -100).sum().item()
        print(f"  有效 Token 数: {valid_count} / {labels.numel()}")
        
        if valid_count > 0:
            # 找到第一个有效 token 的位置
            first_valid = (labels != -100).nonzero(as_tuple=True)[0][0].item()
            # 打印该位置附近的 Token ID 和解码文本
            context_ids = input_ids[max(0, first_valid-5):first_valid+10]
            print(f"  训练起始上下文 (IDs): {context_ids.tolist()}")
            print(f"  训练起始文本: {self.tokenizer.decode(context_ids)}")
            
            # 检查是否包含 <think>
            full_text = self.tokenizer.decode(input_ids[labels != -100])
            if "<think>" in full_text:
                print("  ✓ 检测到 <think> 标签，思维链将被纳入训练。")
            else:
                print("  ℹ️ 未检测到 <think> (可能被截断或样本不包含)。")
        else:
            print("  ❌ 错误: 仍未找到有效 Token!")
            


class HCNTuner(Trainer):
    """
    HCN-SFT训练器 - 带详细日志记录
    负责打印和保存SFT/SCT损失分解信息，以及梯度统计
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 创建日志文件
        self.loss_log_file = os.path.join(self.args.output_dir, "training_losses_detailed.txt")
        self.gradient_log_file = os.path.join(self.args.output_dir, "gradient_stats.txt")

        # 初始化日志文件
        os.makedirs(self.args.output_dir, exist_ok=True)

        with open(self.loss_log_file, 'w', encoding='utf-8') as f:
            f.write("# HCN-SFT 训练损失详细记录\n")
            f.write("# 格式: step,epoch,sft_loss,sct_loss,total_loss,sft_ratio(%),sct_ratio(%)\n")
            f.write("# 生成时间: " + str(datetime.now()) + "\n\n")

        with open(self.gradient_log_file, 'w', encoding='utf-8') as f:
            f.write("# HCN-SFT 梯度统计记录\n")
            f.write("# 格式: step,epoch,total_norm,max_norm,mean_norm,std_norm,grad_nan_count\n")
            f.write("# 生成时间: " + str(datetime.now()) + "\n\n")

    @property
    def tokenizer_or_processor(self):
        """获取tokenizer或processing_class，兼容不同版本的transformers"""
        # 优先使用新的processing_class，如果不可用则使用tokenizer
        if hasattr(self, 'processing_class') and self.processing_class is not None:
            return self.processing_class
        elif hasattr(self, 'tokenizer') and self.tokenizer is not None:
            return self.tokenizer
        else:
            raise AttributeError("既没有 processing_class 也没有 tokenizer 可用")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算损失并记录SFT/SCT分解信息
        """
        # 调用父类的compute_loss
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        # 提取HCN特定的损失信息
        sft_loss = outputs.get("sft_loss", loss)
        sct_loss = outputs.get("sct_loss", 0.0)

        # 安全地转换为标量值
        def safe_to_value(tensor):
            if isinstance(tensor, torch.Tensor):
                if torch.isnan(tensor).any():
                    return 0.0
                return tensor.item()
            return float(tensor) if tensor is not None else 0.0

        # 每步都记录损失
        total_loss_val = safe_to_value(loss)
        sft_loss_val = safe_to_value(sft_loss)
        sct_loss_val = safe_to_value(sct_loss)

        # 计算安全比值
        def safe_ratio(numerator, denominator):
            if denominator == 0 or not (0 < denominator < float('inf')):
                return 0.0
            return numerator / denominator

        sct_ratio = safe_ratio(sct_loss_val, total_loss_val)
        sft_ratio = safe_ratio(sft_loss_val, total_loss_val)

        # 记录到文件
        step = self.state.global_step
        epoch = self.state.epoch if hasattr(self.state, 'epoch') else step / self.args.max_steps

        with open(self.loss_log_file, 'a', encoding='utf-8') as f:
            f.write(f"{step},{epoch:.6f},{sft_loss_val:.8f},{sct_loss_val:.8f},{total_loss_val:.8f},{sft_ratio*100:.2f},{sct_ratio*100:.2f}\n")

        # 在logging_steps时打印详细信息
        if step % self.args.logging_steps == 0:
            print(f"\n🔍 [HCN损失分解] Step {step}, Epoch {epoch:.3f}:")
            print(f"  SFT损失: {sft_loss_val:.6f} ({sft_ratio*100:.1f}%)")
            print(f"  SCT损失: {sct_loss_val:.6f} ({sct_ratio*100:.1f}%)")
            print(f"  总损失: {total_loss_val:.6f}")
            print(f"  📁 损失已保存到: {self.loss_log_file}")

        return loss if not return_outputs else (loss, outputs)


    def _save_checkpoint(self, model, trial, metrics=None):
        """
        重写保存逻辑 (修复版):
        1. 正常保存完整 Checkpoint (受 save_total_limit 控制，会被轮替删除)
        2. 扁平化保存 LoRA 和 Projectors 到 lora_history/step-X (方便 vLLM 直接加载)
        """
        # 1. 正常保存 Checkpoint (Optimizer, Scheduler 等大文件)
        try:
            super()._save_checkpoint(model, trial, metrics)
        except TypeError:
            try:
                super()._save_checkpoint(model, trial)
            except TypeError:
                super()._save_checkpoint(model)

        # 2. 获取当前步数和输出目录
        step = self.state.global_step
        run_dir = self._get_output_dir(trial=trial)
        
        # 创建永久保存目录: .../lora_history/step-X
        permanent_save_dir = os.path.join(run_dir, "lora_history", f"step-{step}")
        os.makedirs(permanent_save_dir, exist_ok=True)

        if self.args.should_save:
            # print(f"  💾 [HCN-SFT] 正在保存 LoRA 快照至: {permanent_save_dir}...")

            # (A) 保存 LLM LoRA [关键修改：直接保存到 permanent_save_dir，不再创建子文件夹]
            # 这样 adapter_config.json 会直接出现在 step-X 目录下
            model.llm.save_pretrained(permanent_save_dir)

            # 保存 Tokenizer (方便推理时直接指向该目录)
            try:
                self.tokenizer_or_processor.save_pretrained(permanent_save_dir)
            except Exception:
                pass

            # (B) 保存 Projector (作为 .pt 文件保存在同一目录下)
            # vLLM 会忽略 .pt 文件，但科研评估代码可以加载它们
            if model.input_projector is not None:
                torch.save(model.input_projector.state_dict(),
                           os.path.join(permanent_save_dir, "input_projector.pt"))

            if model.align_projector is not None:
                torch.save(model.align_projector.state_dict(),
                           os.path.join(permanent_save_dir, "align_projector.pt"))

            print(f"  ✅ [Step {step}] LoRA与轻量级组件已保存至: {permanent_save_dir}")

    

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        在每步结束时记录梯度统计
        """
        super().on_step_end(args, state, control, model, **kwargs)

        # 记录梯度统计
        if model is not None and state.global_step % args.logging_steps == 0:
            try:
                # 计算梯度统计
                total_norm = 0.0
                max_norm = 0.0
                all_grads = []
                nan_count = 0

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_norm += grad_norm ** 2
                        max_norm = max(max_norm, grad_norm)
                        all_grads.append(grad_norm)

                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            nan_count += 1

                total_norm = total_norm ** 0.5
                mean_norm = sum(all_grads) / len(all_grads) if all_grads else 0.0
                std_norm = (sum((g - mean_norm)**2 for g in all_grads) / len(all_grads))**0.5 if all_grads else 0.0

                # 记录到梯度文件
                epoch = state.epoch if hasattr(state, 'epoch') else state.global_step / args.max_steps
                with open(self.gradient_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{state.global_step},{epoch:.6f},{total_norm:.6f},{max_norm:.6f},{mean_norm:.6f},{std_norm:.6f},{nan_count}\n")

                # 打印梯度信息
                print(f"  📊 梯度统计: 总范数={total_norm:.4f}, 最大范数={max_norm:.4f}, NaN数={nan_count}")
                print(f"  📁 梯度已保存到: {self.gradient_log_file}")

            except Exception as e:
                print(f"  ⚠️ 梯度统计记录失败: {e}")

        return control


def get_target_modules(model):
    """
    动态获取模型的LoRA目标模块
    """
    target_modules = []

    # 收集所有Linear层名称
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            target_modules.append(name)

    # 根据模型类型选择合适的模块名称模式
    model_name_lower = model.config._name_or_path.lower()

    if 'internlm' in model_name_lower:
        patterns = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'w1', 'w2', 'w3']
    elif 'llama' in model_name_lower or 'qwen' in model_name_lower:
        patterns = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        # 默认模式，适用于大多数transformer模型
        patterns = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # 筛选匹配的模块
    selected_modules = []
    for module_name in target_modules:
        for pattern in patterns:
            if pattern in module_name:
                selected_modules.append(module_name)
                break

    return selected_modules

# ==================== 4. 自动特征提取功能 ====================

def extract_hcn_features_auto(data_samples, model_path, output_path, batch_size=8,
                              cache_dir="./feature_cache", cache_features=True,
                              lora_path=None):
    """
    自动提取HCN特征

    Args:
        data_samples: 预处理后的数据样本
        model_path: LLM模型路径
        output_path: 输出特征文件路径
        batch_size: 批次大小
        cache_dir: 缓存目录
        cache_features: 是否使用缓存
        lora_path: SFT LoRA适配器路径（可选）

    Returns:
        features_path: 特征文件路径
    """
    import hashlib
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print("🔍 检查HCN特征文件...")

    # 如果文件已存在且启用缓存，直接返回
    if os.path.exists(output_path) and cache_features:
        print(f"✓ HCN特征文件已存在: {output_path}")
        return output_path

    # 生成缓存文件名（基于数据内容和模型路径）
    model_identifier = f"{model_path}_{lora_path}" if lora_path else model_path
    data_str = json.dumps([sample.get('sub_id', str(i)) for i, sample in enumerate(data_samples[:10])],
                         sort_keys=True)
    cache_hash = hashlib.md5(f"{data_str}_{model_identifier}".encode()).hexdigest()[:8]
    cache_filename = f"hcn_features_{cache_hash}.pt"
    cache_path = os.path.join(cache_dir, cache_filename)

    # 检查缓存
    if cache_features and os.path.exists(cache_path):
        print(f"✓ 从缓存加载HCN特征: {cache_path}")
        # 复制到目标路径
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        import shutil
        shutil.copy2(cache_path, output_path)
        return output_path

    print(f"⚠️ HCN特征文件不存在，开始自动提取...")
    print(f"  基础模型: {model_path}")
    if lora_path:
        print(f"  SFT LoRA: {lora_path}")
    print(f"  样本数: {len(data_samples)}")
    print(f"  批次大小: {batch_size}")

    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 4-bit量化加载
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto"
        )

        # 如果提供LoRA路径，加载SFT训练后的适配器
        if lora_path and os.path.exists(lora_path):
            print(f"📈 加载SFT LoRA适配器: {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)
            print("✓ SFT LoRA适配器加载成功")

        model.eval()

        # 特征提取
        all_features = []
        max_length = 512  # 特征提取的序列长度限制

        for i in range(0, len(data_samples), batch_size):
            batch_samples = data_samples[i:i+batch_size]
            batch_texts = []

            # 准备文本输入
            for sample in batch_samples:
                # 构建简化的输入文本（使用原始帖子和当前发言）
                text_parts = []
                if sample.get('original_post'):
                    text_parts.append(f"原始帖子: {sample['original_post']}")
                if sample.get('target_post'):
                    text_parts.append(f"当前发言: {sample['target_post']}")
                if sample.get('context_post'):
                    text_parts.append(f"上下文: {sample['context_post']}")

                batch_text = " | ".join(text_parts)
                batch_texts.append(batch_text)

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 提取隐藏状态
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                # 使用最后一层的平均池化作为特征
                hidden_states = outputs.hidden_states[-1]  # [B, L, D]
                attention_mask = inputs['attention_mask']

                # 平均池化，考虑attention mask
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                batch_features = sum_embeddings / sum_mask  # [B, D]

                all_features.append(batch_features.cpu())

            if (i // batch_size + 1) % 10 == 0:
                print(f"  已处理: {min(i+batch_size, len(data_samples))}/{len(data_samples)}")

        # 合并所有特征
        all_features = torch.cat(all_features, dim=0)  # [N, D]

        # 扩展为4个认知维度（HCN需要的格式）
        # 这里使用相同的特征，实际中应该使用不同方法提取每个维度的特征
        features_4d = all_features.unsqueeze(1).expand(-1, 4, -1)  # [N, 4, D]

        # 保存特征
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save({
            'features': features_4d,
            'metadata': {
                'num_samples': len(data_samples),
                'feature_dim': all_features.shape[-1],
                'model_path': model_path,
                'extraction_date': datetime.now().isoformat()
            }
        }, output_path)

        print(f"✓ HCN特征提取完成: {output_path}")
        print(f"  特征形状: {features_4d.shape}")

        # 缓存特征文件
        if cache_features:
            os.makedirs(cache_dir, exist_ok=True)
            import shutil
            shutil.copy2(output_path, cache_path)
            print(f"✓ 特征已缓存: {cache_path}")

        return output_path

    except Exception as e:
        print(f"✗ HCN特征提取失败: {e}")
        raise

# ==================== 5. 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="HCN-SFT: 双曲引导的认知指令微调")

    # --- 基本配置参数 (完全对齐 cognitive_sft.py) ---
    parser.add_argument(
        "--model_name_or_path", type=str,
        default="Qwen3-8B",
        help="基础模型路径"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./hcn_sft_results",
        help="模型输出目录（会自动添加模型名-时间戳前缀）"
    )
    parser.add_argument(
        "--data_name", type=str,
        default="hcn_cognitive",
        help="数据集名称"
    )
    parser.add_argument(
        "--model_name", type=str,
        default="Qwen3-8B",
        help="模型名称"
    )
    parser.add_argument(
        "--dataset_path", type=str,
        default="dataset/cut/train_data.json",
        help="认知分析训练数据路径"
    )
    parser.add_argument(
        "--cognitive_labels_path", type=str,
        default="dataset/cut/labels.json",
        help="认知维度标签配置文件路径"
    )

    # HCN特定参数
    parser.add_argument(
        "--hcn_model_path", type=str,
        required=True,
        help="训练好的 HCN 模型路径 (.pt)"
    )
    parser.add_argument(
        "--hcn_code_dir", type=str,
        default="../HCN/04_hcn_training/models",
        help="HCN源代码目录 (包含hcn_model.py, hyperbolic.py, transformer.py)"
    )
    parser.add_argument(
        "--train_features_path", type=str,
        default=None,
        help="对应的 HCN 特征文件路径 (.pt) - 使用--auto_extract_features时自动生成"
    )

    # --- 训练参数 (对齐 cognitive_sft.py) ---
    parser.add_argument("--max_length", type=int, default=4096, help="最大序列长度")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="批次大小（HCN模型显存占用大）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率（建议值：5e-5，数值稳定性优化后的推荐值）")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--max_steps", type=int, default=1000, help="最大训练步数")
    parser.add_argument("--save_steps", type=int, default=200, help="保存步数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="梯度裁剪")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="热身比例")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="学习率调度器")
    parser.add_argument("--fp16", action="store_true", default=False, help="使用混合精度训练")
    parser.add_argument("--bf16", action="store_true", default=True, help="使用bf16精度训练")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="梯度检查点")
    parser.add_argument("--report_to", type=str, default="none", help="报告集成 (none, wandb, tensorboard等)")

    # --- LoRA 参数 (对齐 cognitive_sft.py) ---
    parser.add_argument("--use_lora", action="store_true", default=True, help="使用 LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--use_qlora", action="store_true", default=True, help="使用 QLoRA")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="4位量化")

    # --- 继续训练参数 (对齐 cognitive_sft.py) ---
    parser.add_argument("--continue_from_lora", action="store_true", default=False,
                       help="是否从已有LoRA适配器继续训练")
    parser.add_argument("--lora_path", type=str, default=None,
                       help="已有LoRA适配器的路径，当continue_from_lora为True时使用")

    # --- HCN-SFT 检查点继续训练 ---
    parser.add_argument("--continue_from_hcn_sft", action="store_true", default=False,
                       help="是否从已有HCN-SFT检查点继续训练")
    parser.add_argument("--hcn_sft_checkpoint_path", type=str, default=None,
                       help="已有HCN-SFT检查点的路径，包含final_llm_lora和projectors")
    parser.add_argument("--resume_training", action="store_true", default=False,
                       help="从上一个训练检查点恢复（包括训练状态）")

    # --- HCN-SFT 特定参数 ---
    # 修复：将默认值定义为常量，便于统一管理
    DEFAULT_NUM_SOFT_TOKENS = 4
    DEFAULT_LAMBDA_SCT = 0.3
    DEFAULT_PROJECTOR_DROPOUT = 0.1

    parser.add_argument("--num_soft_tokens", type=int, default=DEFAULT_NUM_SOFT_TOKENS,
                       help=f"Soft prompt token数量 (默认: {DEFAULT_NUM_SOFT_TOKENS})")
    parser.add_argument("--lambda_sct", type=float, default=DEFAULT_LAMBDA_SCT,
                       help=f"SCT损失权重 (默认: {DEFAULT_LAMBDA_SCT})")
    parser.add_argument("--projector_dropout", type=float, default=DEFAULT_PROJECTOR_DROPOUT,
                       help=f"投影器dropout率 (默认: {DEFAULT_PROJECTOR_DROPOUT})")

    # --- 特征文件处理参数 ---
    parser.add_argument("--auto_extract_features", action="store_true", default=False,
                       help="如果特征文件不存在，自动提取HCN特征")
    parser.add_argument("--cache_features", action="store_true", default=True,
                       help="缓存提取的特征文件，避免重复计算")
    parser.add_argument("--feature_cache_dir", type=str, default="./feature_cache",
                       help="特征文件缓存目录")
    parser.add_argument("--extraction_batch_size", type=int, default=8,
                       help="特征提取时的批次大小")
    parser.add_argument("--feature_extraction_model", type=str, default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
                       help="用于特征提取的基础LLM模型路径")
    parser.add_argument("--feature_extraction_lora", type=str, default=None,
                       help="用于特征提取的SFT LoRA适配器路径")
    parser.add_argument("--use_sft_for_features", action="store_true", default=False,
                       help="使用SFT训练后的LLM模型提取特征")

    # --- 消融研究参数 (Ablation Study Parameters) ---
    parser.add_argument("--enable_soft_prompt", action="store_true", default=True, help="启用软提示特征注入")
    parser.add_argument("--disable_soft_prompt", action="store_true", help="禁用软提示特征注入")
    parser.add_argument("--enable_sct_loss", action="store_true", default=True, help="启用语义-认知对齐损失")
    parser.add_argument("--disable_sct_loss", action="store_true", help="禁用语义-认知对齐损失")
    parser.add_argument("--enable_hcn_fusion", action="store_true", default=True, help="启用HCN融合特征")
    parser.add_argument("--disable_hcn_fusion", action="store_true", help="禁用HCN融合特征，使用平均特征")
    parser.add_argument("--enable_cosine_alignment", action="store_true", default=True, help="使用余弦相似度对齐")
    parser.add_argument("--disable_cosine_alignment", action="store_true", help="禁用余弦相似度对齐")
    parser.add_argument("--enable_alignment_projector", action="store_true", default=True, help="启用对齐投影器")
    parser.add_argument("--disable_alignment_projector", action="store_true", help="禁用对齐投影器")
    parser.add_argument("--ablation_mode", type=str, default="full",
                       choices=["full", "no_soft_prompt", "no_sct", "no_alignment", "baseline_sft"],
                       help="预定义消融模式")

    args = parser.parse_args()

    # --- 处理消融模式 ---
    # 解析消融模式并设置相应的标志
    if args.ablation_mode != "full":
        if args.ablation_mode == "no_soft_prompt":
            args.disable_soft_prompt = True
        elif args.ablation_mode == "no_sct":
            args.disable_sct_loss = True
        elif args.ablation_mode == "no_alignment":
            args.disable_cosine_alignment = True
            args.disable_alignment_projector = True
        elif args.ablation_mode == "baseline_sft":
            args.disable_soft_prompt = True
            args.disable_sct_loss = True
            args.disable_cosine_alignment = True
            args.disable_alignment_projector = True

    # 参数验证
    if not args.auto_extract_features and args.train_features_path is None:
        parser.error("当不使用 --auto_extract_features 时，必须指定 --train_features_path")

    # 处理enable/disable标志
    enable_soft_prompt = args.enable_soft_prompt and not args.disable_soft_prompt
    enable_sct_loss = args.enable_sct_loss and not args.disable_sct_loss
    enable_hcn_fusion = args.enable_hcn_fusion and not args.disable_hcn_fusion
    enable_cosine_alignment = args.enable_cosine_alignment and not args.disable_cosine_alignment
    enable_alignment_projector = args.enable_alignment_projector and not args.disable_alignment_projector

    # --- 检查特征文件维度 ---
    feat_dim = None
    if args.train_features_path is not None:
        print(f"\n检查HCN特征文件维度: {args.train_features_path}")
        try:
            feat_data = torch.load(args.train_features_path, map_location='cpu')
            if 'features' in feat_data:
                feat_dim = feat_data['features'].shape[-1]
            elif 'embeddings' in feat_data:
                feat_dim = feat_data['embeddings'].shape[-1]
            else:
                feat_dim = feat_data.shape[-1] if isinstance(feat_data, torch.Tensor) else None

            print(f"✓ 特征文件维度: {feat_dim}")

        except Exception as e:
            print(f"⚠️ 无法加载特征文件: {e}")
            feat_dim = None
    else:
        print(f"\n使用自动特征提取模式")

    # 验证维度一致性
    # 修复：将硬编码的默认值提取为常量
    DEFAULT_HCN_INPUT_DIM = 4096
    DEFAULT_HCN_HIDDEN_DIM = 512

    hcn_input_dim = DEFAULT_HCN_INPUT_DIM
    try:
        hcn_ckpt = torch.load(args.hcn_model_path, map_location='cpu')
        hcn_input_dim = hcn_ckpt.get('model_config', {}).get('input_dim', DEFAULT_HCN_INPUT_DIM)
        print(f"✓ HCN模型配置维度: {hcn_input_dim}")

        if feat_dim is not None and feat_dim != hcn_input_dim:
            print(f"⚠️ 警告: 特征维度 ({feat_dim}) 与 HCN 配置维度 ({hcn_input_dim}) 不匹配，将使用特征维度覆盖配置。")
            hcn_input_dim = feat_dim

    except Exception as e:
        print(f"⚠️ 无法检查HCN模型配置: {e}，使用默认值 {hcn_input_dim}")

    # --- 生成带前缀的输出目录名 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = f"-{args.ablation_mode}" if args.ablation_mode != "full" else ""
    prefixed_output_dir = f"{args.output_dir}/{args.model_name}-{args.data_name}{mode_suffix}-{timestamp}"

    print("=" * 60)
    print("HCN-SFT: 双曲引导的认知指令微调")
    print("=" * 60)
    print(f"LLM模型: {args.model_name_or_path}")
    print(f"HCN模型: {args.hcn_model_path}")
    print(f"训练数据: {args.dataset_path}")
    print(f"HCN特征: {args.train_features_path}")
    print(f"输出目录: {prefixed_output_dir}")
    print(f"使用LoRA: {args.use_lora}")
    print(f"使用QLoRA: {args.use_qlora}")
    print(f"4位量化: {args.load_in_4bit}")
    print(f"软提示tokens: {args.num_soft_tokens}")
    print(f"SCT损失权重: {args.lambda_sct}")
    print(f"批次大小: {args.per_device_train_batch_size}")
    print(f"梯度累积: {args.gradient_accumulation_steps}")
    print(f"有效批次大小: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"学习率: {args.learning_rate} (已优化数值稳定性)")
    print("=" * 60)

    # 数值稳定性提示
    print("🔧 数值稳定性增强措施:")
    print("  ✓ SCT Loss计算使用float32精度")
    print("  ✓ CognitiveProjector增强LayerNorm和数值限制")
    print("  ✓ 梯度NaN/Inf实时检测与处理")
    print("  ✓ 学习率已调整为数值稳定推荐值")
    print("  ✓ 可学习缩放因子初始值减小")
    print("=" * 60)

    # --- 初始化HCN模块 ---
    print(f"\n正在初始化HCN模块...")
    global HCNModel
    HCNModel = setup_hcn_imports(args.hcn_code_dir)

    # --- 加载认知标签配置 ---
    print(f"\n正在加载认知标签配置: {args.cognitive_labels_path}")
    try:
        with open(args.cognitive_labels_path, 'r', encoding='utf-8') as f:
            labels_config = json.load(f)
        print("✓ 认知标签配置加载成功")
    except Exception as e:
        print(f"✗ 认知标签配置加载失败: {e}")
        return

    # --- 加载分词器 ---
    print(f"\n正在加载分词器: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right"
    )

    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"设置pad_token: {tokenizer.pad_token}")

    # --- 加载LLM ---
    print(f"\n正在加载LLM: {args.model_name_or_path}")

    # 配置量化
    bnb_config = None
    if args.use_qlora and args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("✓ 启用4位量化")

    # 加载模型
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 启用梯度检查点
    if args.gradient_checkpointing:
        llm_model.gradient_checkpointing_enable()
        print("✓ 梯度检查点已启用")

    print("✓ LLM加载成功")

    # --- LoRA配置 ---
    if args.use_lora:
        print("\n配置LoRA...")
        target_modules = get_target_modules(llm_model)
        print(f"目标模块: {target_modules[:5]}... (共{len(target_modules)}个)")

        # 处理继续训练
        if args.continue_from_lora and args.lora_path:
            print(f"从已有LoRA适配器继续训练: {args.lora_path}")
            try:
                # 准备模型用于量化训练
                if args.use_qlora:
                    llm_model = prepare_model_for_kbit_training(llm_model)

                # 加载已有的LoRA适配器
                llm_model = PeftModel.from_pretrained(llm_model, args.lora_path, is_trainable=True)
                print("✓ LoRA适配器加载成功")
            except Exception as e:
                print(f"✗ LoRA适配器加载失败: {e}")
                print("创建新的LoRA适配器...")
                args.continue_from_lora = False
        else:
            # 创建新的LoRA配置
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )

            # 准备模型用于量化训练
            if args.use_qlora:
                llm_model = prepare_model_for_kbit_training(llm_model)

            # 应用LoRA
            llm_model = get_peft_model(llm_model, lora_config)

        llm_model.print_trainable_parameters()
        print("✓ LoRA配置完成")

    # --- 加载HCN模型 ---
    print(f"\n正在加载HCN模型: {args.hcn_model_path}")
    try:
        hcn_ckpt = torch.load(args.hcn_model_path, map_location='cpu')
        print("✓ HCN检查点加载成功")

        # 修复：使用定义的常量，避免硬编码
        hcn_input_dim = DEFAULT_HCN_INPUT_DIM  # 使用前面定义的常量
        hcn_hidden_dim = DEFAULT_HCN_HIDDEN_DIM

        if 'model_config' in hcn_ckpt:
            config = hcn_ckpt['model_config']
            hcn_input_dim = config.get('input_dim', hcn_input_dim)
            hcn_hidden_dim = config.get('hidden_dim', hcn_hidden_dim)
            print(f"从检查点恢复配置: input_dim={hcn_input_dim}, hidden_dim={hcn_hidden_dim}")

        # 检查是否为消融实验模型
        model_keys = hcn_ckpt['model_state_dict'].keys()

        # 检查双曲层类型
        has_hyperbolic_params = any('hyperbolic_layer.' in key and 'curvature' in key for key in model_keys)
        use_hyperbolic = has_hyperbolic_params

        # 检查Transformer层
        has_transformer_params = any('transformer_encoder.' in key and 'layers' in key for key in model_keys)
        use_transformer = has_transformer_params

        print(f"🔍 检测到模型配置:")
        print(f"  双曲空间: {'✅ 启用' if use_hyperbolic else '❌ 禁用 (欧氏模式)'}")
        print(f"  Transformer: {'✅ 启用' if use_transformer else '❌ 禁用'}")

        # 创建HCN模型（传入消融实验参数）
        hcn_model = HCNModel(
            input_dim=hcn_input_dim,
            hidden_dim=hcn_hidden_dim,
            n_layers=4,
            n_heads=8,
            dropout=0.1,
            use_hyperbolic=use_hyperbolic,
            use_transformer=use_transformer
        )

        # 加载权重
        hcn_model.load_state_dict(hcn_ckpt['model_state_dict'])
        print(f"✓ HCN模型创建成功: input_dim={hcn_input_dim}, hidden_dim={hcn_hidden_dim}")

    except Exception as e:
        print(f"✗ HCN模型加载失败: {e}")
        raise

    # --- HCN-SFT检查点加载 ---
    checkpoint_loaded = False
    if args.continue_from_hcn_sft and args.hcn_sft_checkpoint_path:
        print(f"\n从HCN-SFT检查点继续训练: {args.hcn_sft_checkpoint_path}")
        try:
            # 加载检查点配置
            checkpoint_config_path = os.path.join(args.hcn_sft_checkpoint_path, "hcn_sft_config.json")
            if os.path.exists(checkpoint_config_path):
                with open(checkpoint_config_path, 'r', encoding='utf-8') as f:
                    checkpoint_config = json.load(f)

                # 从检查点恢复配置
                hcn_sft_config = checkpoint_config.get('hcn_sft_config', {})
                print(f"✓ 从检查点恢复配置:")
                print(f"  num_soft_tokens: {hcn_sft_config.get('num_soft_tokens', 'N/A')}")
                print(f"  lambda_sct: {hcn_sft_config.get('lambda_sct', 'N/A')}")
                print(f"  hcn_hidden_dim: {hcn_sft_config.get('hcn_hidden_dim', 'N/A')}")

                # 修复：使用常量进行默认值检测，避免硬编码
                # 更新参数（优先使用命令行参数）
                # 如果命令行参数是默认值且检查点有值，则使用检查点的值
                if args.num_soft_tokens == DEFAULT_NUM_SOFT_TOKENS and 'num_soft_tokens' in hcn_sft_config:
                    args.num_soft_tokens = hcn_sft_config['num_soft_tokens']
                if args.lambda_sct == DEFAULT_LAMBDA_SCT and 'lambda_sct' in hcn_sft_config:
                    args.lambda_sct = hcn_sft_config['lambda_sct']
                if hcn_hidden_dim == DEFAULT_HCN_HIDDEN_DIM and 'hcn_hidden_dim' in hcn_sft_config:
                    hcn_hidden_dim = hcn_sft_config['hcn_hidden_dim']

            # 加载投影器权重
            input_proj_path = os.path.join(args.hcn_sft_checkpoint_path, "input_projector.pt")
            align_proj_path = os.path.join(args.hcn_sft_checkpoint_path, "align_projector.pt")

            checkpoint_loaded = True

        except Exception as e:
            print(f"⚠️ HCN-SFT检查点加载失败: {e}")
            print("将从头开始训练")
            args.continue_from_hcn_sft = False

    # --- 构建HCN-LLM融合模型 ---
    print("\n构建HCN-LLM融合模型...")
    print(f"消融配置:")
    print(f"  软提示注入: {'✓' if enable_soft_prompt else '✗'}")
    print(f"  SCT损失: {'✓' if enable_sct_loss else '✗'}")
    print(f"  HCN融合特征: {'✓' if enable_hcn_fusion else '✗'}")
    print(f"  余弦对齐: {'✓' if enable_cosine_alignment else '✗'}")
    print(f"  对齐投影器: {'✓' if enable_alignment_projector else '✗'}")

    model = HyperbolicGuidedLLM(
        llm_model=llm_model,
        hcn_model=hcn_model,
        llm_hidden_size=llm_model.config.hidden_size,
        hcn_hidden_size=hcn_hidden_dim,
        num_soft_tokens=args.num_soft_tokens,
        lambda_sct=args.lambda_sct,
        dropout=args.projector_dropout,
        enable_soft_prompt=enable_soft_prompt,
        enable_sct_loss=enable_sct_loss,
        enable_hcn_fusion=enable_hcn_fusion,
        enable_cosine_alignment=enable_cosine_alignment,
        enable_alignment_projector=enable_alignment_projector
    )

    # 加载投影器权重（如果从检查点继续）
    if checkpoint_loaded:
        try:
            if enable_soft_prompt and model.input_projector is not None and os.path.exists(input_proj_path):
                model.input_projector.load_state_dict(torch.load(input_proj_path, map_location='cpu'))
                print("✓ 输入投影器权重加载成功")

            if enable_alignment_projector and model.align_projector is not None and os.path.exists(align_proj_path):
                model.align_projector.load_state_dict(torch.load(align_proj_path, map_location='cpu'))
                print("✓ 对齐投影器权重加载成功")
        except Exception as e:
            print(f"⚠️ 投影器权重加载失败: {e}")

    # 启用梯度检查点
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 激活投影器的梯度（仅当投影器存在时）
    trainable_params = []
    if enable_soft_prompt and model.input_projector is not None:
        for p in model.input_projector.parameters():
            p.requires_grad = True
        trainable_params.extend(list(model.input_projector.parameters()))
        print("✓ 输入投影器已激活")

    if enable_alignment_projector and model.align_projector is not None:
        for p in model.align_projector.parameters():
            p.requires_grad = True
        trainable_params.extend(list(model.align_projector.parameters()))
        print("✓ 对齐投影器已激活")

    print(f"✓ HCN-LLM融合模型构建完成，可训练参数: {sum(p.numel() for p in trainable_params)}")

    # --- 数据准备 ---
    print("\n准备训练数据...")

    # 初始化提示词构建器
    prompt_builder = CognitivePromptBuilderSFT(labels_config)
    print("✓ 提示词构建器初始化完成")

    # 加载并预处理原始数据
    print(f"加载原始数据: {args.dataset_path}")
    raw_dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    print(f"✓ 原始数据加载成功，样本数量: {len(raw_dataset)}")

    # 预处理数据（构建历史上下文）
    processed_samples = preprocess_dataset_with_history(raw_dataset)
    print(f"✓ 历史上下文构建完成，处理后样本数量: {len(processed_samples)}")

    # --- 自动特征提取 (如果需要) ---
    if args.auto_extract_features:
        # 生成自动特征文件路径
        feature_dir = os.path.join(args.feature_cache_dir, "auto_extracted")
        os.makedirs(feature_dir, exist_ok=True)

        # 根据是否使用SFT模型生成不同的缓存名
        if args.use_sft_for_features and args.feature_extraction_lora:
            auto_features_path = os.path.join(feature_dir, f"train_features_sft_{len(processed_samples)}samples.pt")
            lora_path = args.feature_extraction_lora
        else:
            auto_features_path = os.path.join(feature_dir, f"train_features_base_{len(processed_samples)}samples.pt")
            lora_path = None

        # 自动提取HCN特征
        actual_features_path = extract_hcn_features_auto(
            data_samples=processed_samples,
            model_path=args.feature_extraction_model,
            output_path=auto_features_path,
            batch_size=args.extraction_batch_size,
            cache_dir=args.feature_cache_dir,
            cache_features=args.cache_features,
            lora_path=lora_path
        )
        # 使用实际的特征文件路径
        train_features_path = actual_features_path
    else:
        train_features_path = args.train_features_path

    # 构建HCN感知的数据集
    train_dataset = HybridCognitiveDataset(
        processed_samples,
        train_features_path,
        tokenizer,
        prompt_builder,
        max_length=args.max_length
    )

    # --- 训练配置 ---
    print("\n配置训练参数...")
    training_args = TrainingArguments(
        output_dir=prefixed_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_first_step=True,
        save_total_limit=1,  # 完整的大检查点只保留最新的1个
        save_strategy="steps",  # 确保保存策略按步数
        save_only_model=False,  # 保存完整的模型状态（包括tokenizer等）
        remove_unused_columns=False,  # 重要：保留hcn_features
        report_to=args.report_to,
        dataloader_pin_memory=False,  # HCN特征可能很大，禁用pin_memory
        gradient_checkpointing=args.gradient_checkpointing
    )

    # 初始化训练器 - 使用HCNTuner打印损失分解
    # 兼容新版本transformers，需要明确传递tokenizer或processing_class
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": HCNAwareDataCollator(tokenizer)
    }

    # 尝试传递tokenizer或processing_class（新版本推荐processing_class）
    try:
        # 先尝试使用processing_class（新版本）
        trainer_kwargs["processing_class"] = tokenizer
    except Exception:
        # 如果失败，使用tokenizer（旧版本）
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = HCNTuner(**trainer_kwargs)

    print("✓ 训练器初始化完成")

    # --- 开始训练 ---
    print(f"\n🚀 开始HCN-SFT联合训练...")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"总步数预估: {len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps) * args.num_train_epochs}")

    try:
        # 检查是否需要从检查点恢复
        if args.resume_training and args.continue_from_hcn_sft and os.path.exists(args.hcn_sft_checkpoint_path):
            # 寻找最新的检查点
            checkpoint_dir = os.path.join(args.hcn_sft_checkpoint_path, "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                    print(f"从检查点恢复: {checkpoint_path}")
                    trainer.train(resume_from_checkpoint=checkpoint_path)
                    print("✓ 训练状态恢复成功")
                else:
                    print("⚠️ 未找到检查点，将从头开始训练")
                    trainer.train()
            else:
                print("⚠️ 检查点目录不存在，将从头开始训练")
                trainer.train()
        else:
            trainer.train()
        print("✓ 训练完成")
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        raise

    # --- 保存模型 ---
    print(f"\n💾 保存模型到: {prefixed_output_dir}")

    # 确保输出目录存在
    os.makedirs(prefixed_output_dir, exist_ok=True)

    try:
        # 保存LLM LoRA权重
        if hasattr(llm_model, 'save_pretrained'):
            llm_model.save_pretrained(os.path.join(prefixed_output_dir, "final_llm_lora"))
            tokenizer.save_pretrained(os.path.join(prefixed_output_dir, "final_llm_lora"))

        # 保存投影器权重
        torch.save(model.input_projector.state_dict(),
                  os.path.join(prefixed_output_dir, "input_projector.pt"))
        torch.save(model.align_projector.state_dict(),
                  os.path.join(prefixed_output_dir, "align_projector.pt"))

        # 保存训练配置
        config_to_save = {
            "model_name_or_path": args.model_name_or_path,
            "hcn_model_path": args.hcn_model_path,
            "dataset_path": args.dataset_path,
            "train_features_path": args.train_features_path,
            "cognitive_labels_path": args.cognitive_labels_path,
            "training_args": training_args.to_dict(),
            "hcn_sft_config": {
                "num_soft_tokens": args.num_soft_tokens,
                "lambda_sct": args.lambda_sct,
                "projector_dropout": args.projector_dropout,
                "hcn_input_dim": hcn_input_dim,
                "hcn_hidden_dim": hcn_hidden_dim,
                "llm_hidden_size": llm_model.config.hidden_size
            },
            "lora_config": {
                "use_lora": args.use_lora,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "use_qlora": args.use_qlora,
                "load_in_4bit": args.load_in_4bit
            } if args.use_lora else None,
            "training_summary": {
                "final_global_step": trainer.state.global_step,
                "final_training_loss": trainer.state.log_history[-1].get("train_loss", None) if trainer.state.log_history else None
            }
        }

        with open(f"{prefixed_output_dir}/hcn_sft_config.json", "w", encoding="utf-8") as f:
            json.dump(config_to_save, f, ensure_ascii=False, indent=2)

        print("✓ 模型、配置和损失历史保存成功")

    except Exception as e:
        print(f"✗ 模型保存失败: {e}")

    print(f"\n🎉 HCN-SFT训练完成！")
    print("=" * 60)
    print(f"📁 输出目录: {prefixed_output_dir}")
    print(f"")
    print(f"🤖 模型权重:")
    print(f"  LLM LoRA: {prefixed_output_dir}/final_llm_lora")
    print(f"  输入投影器: {prefixed_output_dir}/input_projector.pt")
    print(f"  对齐投影器: {prefixed_output_dir}/align_projector.pt")
    print(f"")
    print(f"📊 训练数据:")
    print(f"  📄 详细损失记录(TXT): {prefixed_output_dir}/training_losses_detailed.txt")
    print(f"  📈 梯度统计(TXT): {prefixed_output_dir}/gradient_stats.txt")
    print(f"  📊 损失历史(CSV): {prefixed_output_dir}/training_losses.csv")
    print(f"  📋 损失历史(JSON): {prefixed_output_dir}/training_losses.json")
    print(f"  ⚙️ 训练配置: {prefixed_output_dir}/hcn_sft_config.json")
    print(f"")
    print(f"📈 训练统计:")
    final_loss = trainer.state.log_history[-1].get("train_loss", None) if trainer.state.log_history else None
    if final_loss is not None:
        print(f"  最终训练损失: {final_loss:.6f}")
    print(f"  最终训练步数: {trainer.state.global_step}")
    print(f"  训练轮数: {trainer.state.epoch}")
    print("=" * 60)


if __name__ == "__main__":
    main()