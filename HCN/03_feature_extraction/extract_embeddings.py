#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取脚本
从微调后的LLM中提取四维认知特征向量
"""

import argparse
import os
import json
import torch
import numpy as np
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 导入统一的标签管理
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
from prompts import get_prompts


class CognitiveDataset(Dataset):
    """认知状态数据集 - 特征提取专用 (修复版)"""

    def __init__(self, data_path, tokenizer, max_length=2048, labels_path=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompts = get_prompts(labels_path)

        # 定义需要提取特征的锚点标记
        # 注意：顺序必须与 SFT 训练时的输出顺序一致
        self.markers = ['<<<EMOTION>>>', '<<<THINKING>>>', '<<<INTENT>>>', '<<<STANCE>>>']

        # 预先对标记进行分词，用于后续拼接
        # add_special_tokens=False 确保纯文本分词
        self.marker_tokens = {}
        for m in self.markers:
            self.marker_tokens[m] = self.tokenizer.encode(m, add_special_tokens=False)

        # 强制使用排序后的标签，确保一致性
        # 确保标签顺序在训练集和测试集之间完全一致
        self.label_maps = {
            'emotion': {l: i for i, l in enumerate(sorted(self.prompts.get_label_list('emotion')))},
            'thinking': {l: i for i, l in enumerate(sorted(self.prompts.get_label_list('thinking')))},
            'intent': {l: i for i, l in enumerate(sorted(self.prompts.get_label_list('intent')))},
            'stance': {l: i for i, l in enumerate(sorted(self.prompts.get_label_list('stance')))}
        }

        # 为了向后兼容，保留原有的属性名
        self.emotion_labels = self.prompts.get_label_list('emotion')
        self.thinking_labels = self.prompts.get_label_list('thinking')
        self.intent_labels = self.prompts.get_label_list('intent')
        self.stance_labels = self.prompts.get_label_list('stance')
        self.emotion_label2id = self.label_maps['emotion']
        self.thinking_label2id = self.label_maps['thinking']
        self.intent_label2id = self.label_maps['intent']
        self.stance_label2id = self.label_maps['stance']

        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                labels = self.parse_labels(item)
                if labels:
                    data.append({
                        'messages': item['messages'],
                        'labels': labels
                    })
        return data

    def parse_labels(self, item):
        # 解析 assistant 消息中的标签
        assistant_content = ""
        for msg in item['messages']:
            if msg['role'] == 'assistant':
                assistant_content = msg['content']
                break
        if not assistant_content:
            return None
        import re
        labels = {}
        patterns = {
            'emotion': r'<<<EMOTION>>>(.+?)(?=<<<|$|\n)',
            'thinking': r'<<<THINKING>>>(.+?)(?=<<<|$|\n)',
            'intent': r'<<<INTENT>>>(.+?)(?=<<<|$|\n)',
            'stance': r'<<<STANCE>>>(.+?)(?=<<<|$|\n)'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, assistant_content)
            if match:
                val = match.group(1).strip()
                if val in self.label_maps[key]:
                    labels[key] = val
        return labels if len(labels) == 4 else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. 构建基础 Context (System + User)
        input_messages = [msg for msg in item['messages'] if msg['role'] in ['system', 'user']]

        # 使用 apply_chat_template 生成直到 Assistant 开始前的文本
        # add_generation_prompt=True 会添加 "<|im_start|>assistant\n" (针对 Qwen)
        context_text = self.tokenizer.apply_chat_template(
            input_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 2. 手动构建 Input IDs 并记录锚点位置
        # 我们不再依赖搜索，而是通过构建过程确切知道位置
        input_ids = self.tokenizer.encode(context_text, add_special_tokens=False)

        marker_indices = {} # 记录每个标记最后一个token的索引

        # 模拟 Assistant 的输出过程：依次追加标记
        # 格式示例: ...<|im_start|>assistant\n<<<EMOTION>>>\n<<<THINKING>>>...
        # 这里为了提取特征，我们只拼接标记 Key，不拼接 Value

        for m in self.markers:
            # 追加换行符（如果需要模拟每行一个）
            # 注意：根据 sft_cognitive.py，输出格式是 <<<EMOTION>>>val\n<<<THINKING>>>val
            # 特征提取时，我们希望提取 "<<<EMOTION>>>" 这个整体被编码后的向量

            # 为了确保分词连贯，我们先加一个换行符(或空格)，但这取决于SFT数据的具体格式
            # 假设 SFT 数据中标签之间有换行
            if len(marker_indices) > 0: # 不是第一个标记，加换行
                newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
                input_ids.extend(newline_ids)

            # 追加标记本身的 Token IDs
            m_ids = self.marker_tokens[m]
            input_ids.extend(m_ids)

            # 记录该标记最后一个 Token 的绝对位置 (当前的长度 - 1)
            # 这是提取 Embedding 的最佳位置，代表模型"读完这个标签"后的状态
            marker_indices[m] = len(input_ids) - 1

        # 3. 截断与 Padding
        # 注意：因为我们要提取特定位置，Padding 必须小心处理
        # 转换为 Tensor
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)

        # 如果超过最大长度，进行截断 (保留最后的标记部分，截断前面的 context)
        if len(input_ids_tensor) > self.max_length:
            # 这种情况比较危险，标记索引会乱，简单起见我们截断前面
            # 但为了脚本稳健，建议 max_length 设大一点
            input_ids_tensor = input_ids_tensor[-self.max_length:]
            # 重新计算索引偏移量
            offset = len(input_ids) - len(input_ids_tensor)
            for k in marker_indices:
                marker_indices[k] -= offset

        # Left Padding (为了批处理对齐)
        padding_len = self.max_length - len(input_ids_tensor)
        if padding_len > 0:
            pad_ids = torch.full((padding_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            # Left padding: [PAD, ..., PAD, ID1, ID2, ...]
            final_input_ids = torch.cat([pad_ids, input_ids_tensor])
            attention_mask = torch.cat([torch.zeros(padding_len), torch.ones(len(input_ids_tensor))])

            # 修正索引：所有索引都要加上 padding_len
            for k in marker_indices:
                marker_indices[k] += padding_len
        else:
            final_input_ids = input_ids_tensor
            attention_mask = torch.ones(len(input_ids_tensor))

        # 4. 转换标签 ID
        label_ids = {k: self.label_maps[k].get(v, 0) for k, v in item['labels'].items()}

        # 将索引位置也作为数据返回
        indices_tensor = torch.tensor([marker_indices[m] for m in self.markers], dtype=torch.long)

        return {
            'input_ids': final_input_ids,
            'attention_mask': attention_mask,
            'labels': label_ids,
            'marker_indices': indices_tensor # [4]
        }


def load_model(model_path, lora_path=None, device='auto'):
    """加载模型和tokenizer"""
    print(f"加载基础模型: {model_path}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )

    # 注意：不再添加特殊tokens，使用纯文本标记 <<<EMOTION>>> 等

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    # 注意：不再需要调整词汇表大小，因为使用现有tokens

    # 如果有LoRA适配器，加载它
    if lora_path and os.path.exists(lora_path):
        print(f"加载LoRA适配器: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()

    return model, tokenizer




def extract_features_batch(model, tokenizer, batch, device, marker_ids_map=None):
    """
    批量提取特征 - 直接使用预计算的索引
    """
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    marker_indices = batch['marker_indices'].to(device) # [B, 4]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # 获取最后一层隐状态
        last_hidden_states = outputs.hidden_states[-1] # [B, Seq, Dim]

        batch_features = []

        for i in range(input_ids.size(0)):
            # 直接从 marker_indices 获取位置
            # indices: [idx_emotion, idx_thinking, idx_intent, idx_stance]
            indices = marker_indices[i]

            # 提取对应位置的向量
            # [4, Dim]
            features = last_hidden_states[i, indices, :]
            batch_features.append(features)

        return torch.stack(batch_features) # [B, 4, Dim]


def main():
    parser = argparse.ArgumentParser(description="HCN特征提取脚本")

    parser.add_argument(
        "--model_path", type=str, required=True,
        help="基础模型路径"
    )
    parser.add_argument(
        "--lora_path", type=str,
        help="LoRA适配器路径"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="数据文件路径（train.jsonl/val.jsonl/test.jsonl）"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="输出文件路径（.pt格式）"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="批次大小"
    )
    parser.add_argument(
        "--max_length", type=int, default=8192,
        help="最大序列长度（建议8192以上，避免Qwen3思考过程被截断）"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="设备选择"
    )
    parser.add_argument(
        "--labels_path", type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cut', 'labels.json'),
        help="标签配置文件路径"
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # 加载模型和tokenizer
    model, tokenizer = load_model(args.model_path, args.lora_path, args.device)

    # 创建数据集
    print(f"加载数据集: {args.data_path}")
    dataset = CognitiveDataset(args.data_path, tokenizer, args.max_length, args.labels_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"数据集大小: {len(dataset)}")

    # 提取特征
    device = next(model.parameters()).device
    all_features = []
    all_labels = {'emotion': [], 'thinking': [], 'intent': [], 'stance': []}

    print("开始提取特征...")
    print(f"总共 {len(dataloader)} 个批次，批次大小: {args.batch_size}")

    batch_count = 0
    successful_samples = 0
    failed_samples = 0

    for batch in tqdm(dataloader, desc="提取特征"):
        batch_count += 1

        # 每10个批次打印一次进度信息
        if batch_count % 10 == 0:
            print(f"  正在处理第 {batch_count}/{len(dataloader)} 批次...")

        # 提取特征向量
        features = extract_features_batch(model, tokenizer, batch, device)

        # 检查批次是否完全失败（返回空张量）
        if features.numel() == 0:
            print(f"  ❌ 第 {batch_count} 批次完全失败，跳过")
            failed_samples += batch['input_ids'].size(0)
            continue

        # 打印标记统计信息（如果是第一批）
        if batch_count == 1 and hasattr(extract_features_batch, 'marker_stats'):
            stats = extract_features_batch.marker_stats
            print(f"  📊 首批次标记查找: {stats['found']}/{stats['total']} 找到 "
                  f"({stats['found']/stats['total']*100:.1f}%)")

        # 检查特征提取是否成功
        if torch.isnan(features).any():
            print(f"  ❌ 第 {batch_count} 批次特征包含NaN！")
            failed_samples += batch['input_ids'].size(0)
            continue

        if torch.isinf(features).any():
            print(f"  ❌ 第 {batch_count} 批次特征包含Inf！")
            failed_samples += batch['input_ids'].size(0)
            continue

        # 收集成功的特征和对应的标签
        # 注意：features的batch_size可能小于原始batch的size，因为有些样本失败了
        actual_batch_size = features.size(0)
        if actual_batch_size > 0:
            all_features.append(features.cpu())

            # 只收集成功样本的标签（前actual_batch_size个）
            for key in all_labels:
                all_labels[key].extend(batch['labels'][key][:actual_batch_size])

            successful_samples += actual_batch_size

        # 打印前几批次的特征统计
        if batch_count <= 3 and actual_batch_size > 0:
            print(f"  ✅ 批次 {batch_count}: 特征形状 {features.shape}, "
                  f"范围 [{features.min():.6f}, {features.max():.6f}], "
                  f"均值 {features.mean():.6f}")

        # 统计失败的样本数
        failed_in_batch = batch['input_ids'].size(0) - actual_batch_size
        if failed_in_batch > 0:
            failed_samples += failed_in_batch

    # 打印处理统计信息
    total_samples = successful_samples + failed_samples
    template_samples = getattr(extract_features_batch, 'template_count', 0)
    print(f"\n📊 处理统计:")
    print(f"  总样本数: {total_samples}")
    print(f"  成功提取: {successful_samples} ({successful_samples/total_samples*100:.1f}%)")
    print(f"  模板输出: {template_samples} ({template_samples/total_samples*100:.1f}%)")
    print(f"  其他失败: {failed_samples - template_samples} ({(failed_samples - template_samples)/total_samples*100:.1f}%)")

    if template_samples > 0:
        print(f"  ⚠️  警告: {template_samples} 个样本输出模板文本，未进行实际分类")
        print(f"  💡 建议:")
        print(f"     1. 检查SFT训练数据是否包含实际分类标签")
        print(f"     2. 确认模型训练是否完成")
        print(f"     3. 检查生成参数设置是否正确")

    # 检查是否没有成功提取的特征
    if len(all_features) == 0:
        print("❌ 严重错误: 没有成功提取任何特征！请检查以下问题:")
        print("   1. max_length 是否太小，导致截断")
        print("   2. 数据中是否包含所需的标记 <<<EMOTION>>> 等")
        print("   3. 模型输出格式是否正确")
        return False

    # 拼接所有特征
    all_features = torch.cat(all_features, dim=0)  # [num_samples, 4, hidden_dim]

    # 转换标签为tensor
    for key in all_labels:
        all_labels[key] = torch.tensor(all_labels[key], dtype=torch.long)

    print(f"\n特征形状: {all_features.shape}")
    print(f"特征维度: {all_features.shape[2]}")

    # 保存特征和标签
    output_data = {
        'features': all_features,
        'labels': all_labels,
        'num_samples': len(all_features),
        'feature_dim': all_features.shape[2],
        'label_maps': {
            'emotion': dataset.label_maps['emotion'],  # 保存标签到ID的映射字典
            'thinking': dataset.label_maps['thinking'],
            'intent': dataset.label_maps['intent'],
            'stance': dataset.label_maps['stance']
        },
        'label_lists': {  # 保留标签列表用于调试，必须排序确保一致性
            'emotion': sorted(dataset.emotion_labels),
            'thinking': sorted(dataset.thinking_labels),
            'intent': sorted(dataset.intent_labels),
            'stance': sorted(dataset.stance_labels)
        }
    }

    # 特征质量检查（保存前）
    print("\n🔍 特征质量检查:")

    # 1. 基本统计
    print(f"  特征形状: {all_features.shape}")
    print(f"  特征范围: [{all_features.min():.6f}, {all_features.max():.6f}]")
    print(f"  特征均值: {all_features.mean():.6f}")
    print(f"  特征标准差: {all_features.std():.6f}")

    # 2. 零方差检查
    feat_flat = all_features.view(all_features.size(0), -1)  # [num_samples, 4*hidden_dim]
    zero_variance_features = (feat_flat.std(dim=0) < 1e-6).sum().item()
    total_features = feat_flat.size(1)
    print(f"  零方差特征: {zero_variance_features}/{total_features} ({zero_variance_features/total_features*100:.2f}%)")

    if zero_variance_features == total_features:
        print("  ❌ 警告: 所有特征都是零方差！数据提取有问题！")
        return False

    # 3. 样本间相似性检查
    if all_features.size(0) >= 10:
        sample_similarities = []
        for i in range(min(20, all_features.size(0))):
            for j in range(i+1, min(20, all_features.size(0))):
                cos_sim = torch.nn.functional.cosine_similarity(
                    all_features[i].flatten().unsqueeze(0),
                    all_features[j].flatten().unsqueeze(0)
                ).item()
                sample_similarities.append(cos_sim)

        import numpy as np
        print(f"  样本间余弦相似度: 均值={np.mean(sample_similarities):.6f}, 标准差={np.std(sample_similarities):.6f}")

        if np.mean(sample_similarities) > 0.95:
            print("  ⚠️ 警告: 样本间相似度过高，可能存在特征提取问题")

    # 4. 维度间差异检查
    dim_similarities = []
    for sample_idx in range(min(10, all_features.size(0))):
        for i in range(4):
            for j in range(i+1, 4):
                cos_sim = torch.nn.functional.cosine_similarity(
                    all_features[sample_idx, i].unsqueeze(0),
                    all_features[sample_idx, j].unsqueeze(0)
                ).item()
                dim_similarities.append(cos_sim)

    print(f"  维度间平均相似度: {np.mean(dim_similarities):.6f}")

    # 5. NaN/Inf检查
    if torch.isnan(all_features).any():
        print("  ❌ 警告: 特征包含NaN值！")
        return False
    if torch.isinf(all_features).any():
        print("  ❌ 警告: 特征包含Inf值！")
        return False

    print("  ✅ 特征质量检查通过！")

    torch.save(output_data, args.output_path)
    print(f"\n特征已保存到: {args.output_path}")

    # 保存后验证
    print("\n🔒 保存后验证:")
    try:
        # 重新加载验证
        loaded_data = torch.load(args.output_path, map_location='cpu')
        print(f"  重新加载成功: {loaded_data['features'].shape}")

        # 验证数据完整性
        if loaded_data['features'].shape != all_features.shape:
            print("  ❌ 错误: 保存后特征形状不匹配！")
            return False

        for key in all_labels:
            if key not in loaded_data['labels']:
                print(f"  ❌ 错误: 缺少标签 {key}！")
                return False

            if not torch.equal(loaded_data['labels'][key], all_labels[key]):
                print(f"  ❌ 错误: 标签 {key} 不匹配！")
                return False

        print("  ✅ 保存验证通过！")

    except Exception as e:
        print(f"  ❌ 保存验证失败: {e}")
        return False

    # 打印统计信息
    print("\n📊 统计信息:")
    for key in all_labels:
        unique, counts = torch.unique(all_labels[key], return_counts=True)
        print(f"\n{key} 分布:")
        for idx, count in zip(unique, counts):
            label_name = getattr(dataset, f"{key}_labels")[idx.item()]
            print(f"  {label_name}: {count.item()}")


if __name__ == "__main__":
    main()