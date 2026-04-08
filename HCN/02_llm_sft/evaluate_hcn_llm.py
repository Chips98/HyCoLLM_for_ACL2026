#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HCN LLM推理评估脚本 (修复版)
修复了Padding方向、正则解析逻辑，并集成了鲁棒的标签匹配功能
"""

import argparse
import torch
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
from tqdm import tqdm
import yaml
from sklearn.metrics import accuracy_score, f1_score, classification_report, hamming_loss
import pandas as pd
import numpy as np

# 添加utils目录到路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
try:
    from prompts import get_prompts
except ImportError:
    # 如果找不到，提供一个简单的回退或报错
    print("警告: 未找到 utils.prompts 模块，请确保路径设置正确")

class RobustParser:
    """鲁棒的响应解析器，移植自 inference/core/response_parser.py 并适配 HCN 格式"""

    def __init__(self, labels_path: str):
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.labels_config = json.load(f)

        self.valid_labels = {
            "emotion": list(self.labels_config["emotion"].keys()),
            "stance": list(self.labels_config["stance"].keys()),
            "thinking": list(self.labels_config["thinking"]["values"].keys()), # 注意：这里用 values
            "intent": list(self.labels_config["intent"].keys())
        }

    def clean_text(self, text: str) -> str:
        """基础文本清洗"""
        text = re.sub(r'</?think>', '', text) # 移除 think 标签
        return text.strip()

    def fuzzy_match(self, label: str, dimension: str) -> str:
        """模糊匹配标签"""
        label = label.strip()
        # 1. 移除干扰词
        label_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', label) # 仅保留中英文数字

        valid_list = self.valid_labels.get(dimension, [])

        # 1. 精确匹配
        if label in valid_list: return label
        if label_clean in valid_list: return label_clean

        # 2. 包含匹配
        for v in valid_list:
            if v in label: return v

        # 3. 编辑距离/模糊匹配 (简单版)
        best_match = None
        max_overlap = 0
        for v in valid_list:
            # 计算重叠字符数
            overlap = len(set(label_clean) & set(v))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = v

        if best_match and max_overlap >= 2: # 至少重叠2个字
            return best_match

        return ""

    def parse(self, text: str) -> Dict[str, str]:
        """解析 LLM 输出"""
        result = {'emotion': '', 'thinking': '', 'intent': '', 'stance': ''}
        text = self.clean_text(text)

        # 定义正则表达式 (修复了 \\n 问题)
        patterns = {
            'emotion': [
                r'<<<EMOTION>>>(.+?)(?=<<<|$|\n)',  # 标准 HCN 格式
                r'情感[:：]\s*(.+?)(?=\n|$)'          # 兼容 Baseline 格式
            ],
            'thinking': [
                r'<<<THINKING>>>(.+?)(?=<<<|$|\n)',
                r'思维[:：]\s*(.+?)(?=\n|$)'
            ],
            'intent': [
                r'<<<INTENT>>>(.+?)(?=<<<|$|\n)',
                r'意图[:：]\s*(.+?)(?=\n|$)'
            ],
            'stance': [
                r'<<<STANCE>>>(.+?)(?=<<<|$|\n)',
                r'立场[:：]\s*(.+?)(?=\n|$)'
            ]
        }

        for key, pat_list in patterns.items():
            extracted_raw = ""
            for pat in pat_list:
                match = re.search(pat, text, re.DOTALL)
                if match:
                    extracted_raw = match.group(1).strip()
                    break

            if extracted_raw:
                # 尝试验证和模糊匹配
                final_label = self.fuzzy_match(extracted_raw, key)
                result[key] = final_label

        return result


class HCNLLMEvaluator:
    """HCN LLM推理评估器"""

    def __init__(self, model_path: str, lora_path: str = None, test_data_path: str = None,
                 output_dir: str = None, labels_path: str = None, device: str = 'auto'):
        """
        初始化评估器

        Args:
            model_path: 基础模型路径
            lora_path: LoRA适配器路径
            test_data_path: 测试数据路径
            output_dir: 输出目录
            labels_path: 标签配置文件路径
            device: 设备选择
        """
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = device

        # 设置路径
        if test_data_path is None:
            test_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        'data', 'processed', 'test.jsonl')
        self.test_data_path = Path(test_data_path)

        if labels_path is None:
            labels_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'data', 'cut', 'labels.json')
        self.labels_path = labels_path

        # 输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = model_path.replace("/", "_")
            self.output_dir = Path("results") / f"hcn_llm_{model_name}_{timestamp}"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载资源
        self.model, self.tokenizer = self._load_model()
        self.prompts = get_prompts(self.labels_path)
        self.parser = RobustParser(self.labels_path)  # 使用鲁棒解析器
        self.test_data = self._load_test_data()
        self.results = []

        self.dimensions = ["emotion", "stance", "thinking", "intent"]

    def _load_model(self):
        """加载模型和tokenizer"""
        print(f"加载基础模型: {self.model_path}")

        # 修复 1: 必须设置 padding_side="left" 用于 decoder-only 模型的生成
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left"
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )

        # 如果有LoRA适配器，加载它
        if self.lora_path and os.path.exists(self.lora_path):
            print(f"加载LoRA适配器: {self.lora_path}")
            model = PeftModel.from_pretrained(model, self.lora_path)

        model.eval()
        return model, tokenizer

    def _load_test_data(self) -> List[Dict[str, Any]]:
        """加载测试数据"""
        if not self.test_data_path.exists():
            raise FileNotFoundError(f"测试文件不存在: {self.test_data_path}")

        data = []
        if self.test_data_path.suffix == '.jsonl':
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        print(f"加载了 {len(data)} 个测试样本")
        return data

    def _empty_output(self):
        """返回空的输出结构"""
        return {k: "" for k in self.dimensions}


    def extract_real_answer(self, response_text: str) -> str:
        """
        从模型输出中提取真实回答内容

        Args:
            response_text: 原始模型输出

        Returns:
            提取后的真实回答内容
        """
        if not response_text:
            return response_text

        # 首先处理Qwen3模型的""分隔符
        if "</think>" in response_text:
            parts = response_text.split("</think>")
            if len(parts) > 1:
                # 取最后一个"</think>"之后的内容
                real_content = parts[-1].strip()
                # 移除开头的换行符
                real_content = real_content.lstrip('\n')
                response_text = real_content
        return response_text

  
    def run_inference(self, batch_size: int = 8):
        print(f"\n开始批量推理，共 {len(self.test_data)} 个样本，批次大小: {batch_size}...")

        # 预处理数据
        parsed_data = []
        for i, item in enumerate(self.test_data):
            processed_item = self._preprocess_item(item, i)
            if processed_item:
                parsed_data.append(processed_item)
            else:
                # 记录解析失败的样本
                failed_res = item.copy()
                failed_res["llm_output"] = self._empty_output()
                self.results.append(failed_res)

        # 批量处理
        for i in tqdm(range(0, len(parsed_data), batch_size), desc="批量推理"):
            batch = parsed_data[i:i + batch_size]
            self._process_batch(batch)

        return self.results

    def _preprocess_item(self, item, index):
        """统一数据格式提取"""
        context_post = ""
        target_post = ""
        cognitive_labels = {}

        # 尝试从不同格式中提取
        if "context_post" in item and "target_post" in item:
            context_post = item.get("context_post", "")
            target_post = item.get("target_post", "")
            cognitive_labels = item.get("cognitive_labels", {})
        elif "messages" in item:
            for msg in item["messages"]:
                if msg["role"] == "user":
                    content = msg["content"]
                    # 简单提取逻辑
                    if "[背景]" in content and "[评论]" in content:
                        parts = content.split("[评论]")
                        context_post = parts[0].replace("[背景]", "").strip()
                        target_post = parts[1].split("[指令]")[0].strip()
                elif msg["role"] == "assistant":
                    # 尝试从GT中提取标签（如果有）
                    gt_parsed = self.parser.parse(msg["content"])
                    # 合并
                    for k, v in gt_parsed.items():
                        if v: cognitive_labels[k] = v

        if target_post:
            return {
                'index': index,
                'item': item,
                'context_post': context_post,
                'target_post': target_post,
                'cognitive_labels': cognitive_labels
            }
        return None

    def _process_batch(self, batch):
        """处理单个批次"""
        batch_prompts = []
        batch_data = []

        for data in batch:
            user_prompt = f"[背景]{data['context_post']}\n[评论]{data['target_post']}\n[指令]请分析该评论的认知状态。"
            messages = [
                {"role": "system", "content": self.prompts.get_system_prompt()},
                {"role": "user", "content": user_prompt}
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_prompts.append(text)
            batch_data.append(data)

        # 批量 Tokenize (Left Padding 已在 init 设置)
        inputs = self.tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 解码并解析
        for j, output in enumerate(outputs):
            # 获取仅生成的 tokens
            generated_tokens = output[inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # 打印原始响应信息
            # print(f"样本 {batch_data[j]['index']}:")
            # print(f"LLM原始输出: {response}")
            response = self.extract_real_answer(response)
            # print(f"提取后输出: {response}")

            # 使用鲁棒解析器
            llm_output = self.parser.parse(response)
            # print(f"解析结果: {llm_output}")

            '''
            LLM原始输出: <think>
            </think>
            <<<EMOTION>>>中性
            <<<THINKING>>>逻辑
            <<<INTENT>>>表达主张
            <<<STANCE>>>不明确
            
            提取后输出: <<<EMOTION>>>中性
            <<<THINKING>>>逻辑
            <<<INTENT>>>表达主张
            <<<STANCE>>>不明确

            解析结果: {'emotion': '中性', 'thinking': '逻辑', 'intent': '表达主张', 'stance': '不明确'}
            '''
            # 保存结果
            original_data = batch_data[j]['item']
            res_item = original_data.copy()
            res_item["llm_output"] = llm_output

            # 确保有 cognitive_labels (如果是从原始数据继承的)
            if "cognitive_labels" not in res_item or not res_item["cognitive_labels"]:
                 res_item["cognitive_labels"] = batch_data[j]['cognitive_labels']

            self.results.append(res_item)

    
    def save_results(self):
        """保存推理结果，结构与Inference/完全一致"""
        # 保存完整结果
        output_file = self.output_dir / "test_data_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        # 保存配置信息
        config = {
            "model_path": self.model_path,
            "lora_path": self.lora_path,
            "test_data_path": str(self.test_data_path),
            "labels_path": self.labels_path,
            "device": self.device,
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(self.results)
        }

        config_file = self.output_dir / "config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)

        # 保存统计信息
        successful_samples = sum(1 for r in self.results if r.get("llm_output", {}).get("emotion"))
        failed_samples = sum(1 for r in self.results if not r.get("llm_output", {}).get("emotion"))

        stats = {
            "total_samples": len(self.results),
            "successful_samples": successful_samples,
            "failed_samples": failed_samples,
            "model_name": self.model_path,
            "dataset": "hcn_test",
            "timestamp": datetime.now().isoformat()
        }

        stats_file = self.output_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到: {self.output_dir}")
        print(f"成功: {stats['successful_samples']}/{stats['total_samples']}")

        return output_file

    def evaluate_and_save_metrics(self):
        """评估结果并保存指标，格式与Inference/evaluate.py完全一致"""
        if not self.results:
            print("没有结果可以评估")
            return

        print("\n开始评估指标...")

        # 维度独立指标
        metrics = {}

        for dim in self.dimensions:
            # 提取真实标签和预测标签
            y_true = []
            y_pred = []

            for item in self.results:
                if "cognitive_labels" in item and "llm_output" in item:
                    # 处理thinking维度的名称差异
                    if dim == "thinking":
                        true_label = item["cognitive_labels"].get("thinking_value", "")
                        if not true_label:  # 如果没有thinking_value，尝试thinking
                            true_label = item["cognitive_labels"].get("thinking", "")
                    else:
                        true_label = item["cognitive_labels"].get(dim, "")

                    pred_label = item["llm_output"].get(dim, "")

                    if true_label and pred_label:
                        y_true.append(true_label)
                        y_pred.append(pred_label)

            if y_true and y_pred:
                # 计算指标
                accuracy = accuracy_score(y_true, y_pred)
                macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

                # 保存详细报告
                report = classification_report(
                    y_true, y_pred,
                    output_dict=True,
                    zero_division=0
                )

                metrics[dim] = {
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "classification_report": report,
                    "y_true": y_true,
                    "y_pred": y_pred
                }

                print(f"{dim}: ACC={accuracy:.4f}, Macro-F1={macro_f1:.4f}")

        # 全景一致性指标
        all_correct = []
        three_correct = []
        two_correct = []
        all_true = []
        all_pred = []

        for item in self.results:
            if "cognitive_labels" in item and "llm_output" in item:
                true_labels = []
                pred_labels = []

                correct_count = 0
                total_count = 0

                for dim in self.dimensions:
                    if dim == "thinking":
                        true_label = item["cognitive_labels"].get("thinking_value", "")
                        if not true_label:  # 如果没有thinking_value，尝试thinking
                            true_label = item["cognitive_labels"].get("thinking", "")
                        pred_label = item["llm_output"].get("thinking", "")
                    else:
                        true_label = item["cognitive_labels"].get(dim, "")
                        pred_label = item["llm_output"].get(dim, "")

                    if true_label and pred_label:
                        true_labels.append(true_label)
                        pred_labels.append(pred_label)
                        total_count += 1
                        if true_label == pred_label:
                            correct_count += 1

                if total_count > 0:
                    all_true.append(true_labels)
                    all_pred.append(pred_labels)

                    if correct_count == 4:  # 4-All (Exact Match)
                        all_correct.append(1)
                    else:
                        all_correct.append(0)

                    if correct_count >= 3:  # 3-All
                        three_correct.append(1)
                    else:
                        three_correct.append(0)

                    if correct_count >= 2:  # 2-All
                        two_correct.append(1)
                    else:
                        two_correct.append(0)

        # 计算指标
        em_acc = np.mean(all_correct) if all_correct else 0
        three_all_acc = np.mean(three_correct) if three_correct else 0
        two_all_acc = np.mean(two_correct) if two_correct else 0

        if all_true and all_pred:
            from sklearn.preprocessing import MultiLabelBinarizer
            mlb = MultiLabelBinarizer()
            y_true_bin = mlb.fit_transform(all_true)
            y_pred_bin = mlb.transform(all_pred)
            hamming = hamming_loss(y_true_bin, y_pred_bin)
            avg_sample_acc = 1 - hamming
        else:
            hamming = 1.0
            avg_sample_acc = 0.0

        metrics["holistic"] = {
            "exact_match": {"acc": em_acc},
            "three_all": {"acc": three_all_acc},
            "two_all": {"acc": two_all_acc},
            "avg_sample_accuracy": avg_sample_acc,
            "hamming_loss": hamming
        }

        # 保存指标到JSON
        save_metrics = {}

        # 维度指标
        for dim in self.dimensions:
            if dim in metrics:
                save_metrics[dim] = {
                    "accuracy": metrics[dim]["accuracy"],
                    "macro_f1": metrics[dim]["macro_f1"]
                }

        # 全景指标
        if "holistic" in metrics:
            save_metrics["holistic"] = metrics["holistic"]

        # 保存为JSON
        metrics_file = self.output_dir / "metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(save_metrics, f, ensure_ascii=False, indent=2)

        # 保存为CSV
        csv_file = self.output_dir / "metrics.csv"
        rows = []

        # 维度指标行
        for dim in self.dimensions:
            if dim in metrics:
                rows.append({
                    "Dimension": dim,
                    "Accuracy": f"{metrics[dim]['accuracy']:.4f}",
                    "Macro-F1": f"{metrics[dim]['macro_f1']:.4f}"
                })

        # 全景指标行
        if "holistic" in metrics:
            holistic = metrics["holistic"]
            rows.extend([
                {"Dimension": "Exact Match (4-All)", "Accuracy": f"{holistic['exact_match']['acc']:.4f}", "Macro-F1": "N/A"},
                {"Dimension": "3-All", "Accuracy": f"{holistic['three_all']['acc']:.4f}", "Macro-F1": "N/A"},
                {"Dimension": "2-All", "Accuracy": f"{holistic['two_all']['acc']:.4f}", "Macro-F1": "N/A"},
                {"Dimension": "Avg Sample Accuracy", "Accuracy": f"{holistic['avg_sample_accuracy']:.4f}", "Macro-F1": "N/A"},
                {"Dimension": "Hamming Loss", "Accuracy": f"{holistic['hamming_loss']:.4f}", "Macro-F1": "N/A"}
            ])

        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False, encoding='utf-8')

        # 生成评估报告
        self._generate_evaluation_report(save_metrics)

        print(f"\n指标已保存到: {metrics_file} 和 {csv_file}")

    def _generate_evaluation_report(self, metrics):
        """生成评估报告"""
        report_lines = []
        report_lines.append("=" * 50)
        report_lines.append("认知维度推理评估报告")
        report_lines.append("=" * 50)
        report_lines.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"结果文件: {self.output_dir / 'test_data_output.json'}")
        report_lines.append(f"样本数量: {len(self.results)}")
        report_lines.append("")

        # 维度指标
        report_lines.append("维度独立指标:")
        report_lines.append("-" * 30)
        report_lines.append(f"\n{'维度':<15} {'准确率':<10} {'Macro-F1':<10}")
        report_lines.append("-" * 30)

        for dim in self.dimensions:
            if dim in metrics:
                acc = metrics[dim]["accuracy"]
                f1 = metrics[dim]["macro_f1"]
                report_lines.append(f"{dim:<15} {acc:<10.4f} {f1:<10.4f}")

        report_lines.append("")

        # 全景一致性指标
        report_lines.append("全景一致性指标:")
        report_lines.append("-" * 30)
        if "holistic" in metrics:
            holistic = metrics["holistic"]
            report_lines.append(f"Exact Match (4-All): {holistic['exact_match']['acc']:.4f} (所有维度全对的比例)")
            report_lines.append(f"3-All: {holistic['three_all']['acc']:.4f} (至少对3个维度的比例)")
            report_lines.append(f"2-All: {holistic['two_all']['acc']:.4f} (至少对2个维度的比例)")
            report_lines.append(f"Avg Sample Accuracy: {holistic['avg_sample_accuracy']:.4f} (平均每个样本对的维度比例)")
            report_lines.append(f"Hamming Loss: {holistic['hamming_loss']:.4f} (越低越好)")

        report_lines.append("")
        report_lines.append("=" * 50)

        # 保存报告
        report_file = self.output_dir / "evaluation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        # 打印报告
        print('\n' + '\n'.join(report_lines))


  
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--labels_path", type=str, help="标签配置文件路径")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    evaluator = HCNLLMEvaluator(
        model_path=args.model_path,
        lora_path=args.lora_path,
        test_data_path=args.test_data_path,
        labels_path=args.labels_path
    )

    evaluator.run_inference(batch_size=args.batch_size)
    evaluator.save_results()
    evaluator.evaluate_and_save_metrics()

if __name__ == "__main__":
    main()