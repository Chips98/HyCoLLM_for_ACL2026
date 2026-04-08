#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HCN模型训练脚本
训练双曲空间+Transformer的认知状态分类模型
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter  # 移除tensorboard依赖
from datetime import datetime
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss

# 导入模型和损失函数
from models.hcn_model import HCNModel
from models.hyperbolic import PoincareBall
from loss import HCNMutliTaskLoss
from dataset_tensor import CognitiveTensorDataset, create_data_loaders, compute_class_weights


# 设置随机种子
def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # 确保结果可重现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RiemannianAdam(optim.Optimizer):
    """适用于双曲空间的Riemannian Adam优化器"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RiemannianAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """执行一个优化步骤"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # 更新移动平均
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # 偏差校正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # 计算自适应学习率
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                # 对于双曲空间中的参数，使用retraction更新
                if hasattr(p, 'hyperbolic') and p.hyperbolic:
                    # 简化的黎曼更新（实际中需要更复杂的处理）
                    p.data.add_(-step_size, grad)
                    # 投影回双曲空间
                    norm = p.data.norm()
                    if norm > 0.99:
                        p.data.mul_(0.99 / norm)
                else:
                    # 标准欧几里得更新
                    p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


def compute_gradient_stats(model):
    """
    计算模型参数的梯度统计信息
    返回包含各种梯度统计指标的字典
    """
    grad_stats = {
        'param_grad_mean': 0.0,
        'param_grad_std': 0.0,
        'param_grad_max': 0.0,
        'param_grad_min': 0.0,
        'zero_grad_ratio': 0.0,
        'large_grad_count': 0
    }

    all_grads = []
    total_params = 0
    zero_grad_params = 0
    large_grad_params = 0

    # 遍历所有参数的梯度
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            all_grads.append(grad.flatten())
            total_params += grad.numel()

            # 统计零梯度参数
            zero_grad_params += (grad == 0).sum().item()

            # 统计大梯度参数（绝对值大于0.01）
            large_grad_params += (grad.abs() > 0.01).sum().item()
        else:
            total_params += param.numel()
            zero_grad_params += param.numel()

    if all_grads:
        # 将所有梯度合并成一个张量
        all_grads_tensor = torch.cat(all_grads)

        # 计算统计信息
        grad_stats['param_grad_mean'] = all_grads_tensor.mean().item()
        grad_stats['param_grad_std'] = all_grads_tensor.std().item()
        grad_stats['param_grad_max'] = all_grads_tensor.max().item()
        grad_stats['param_grad_min'] = all_grads_tensor.min().item()
        grad_stats['zero_grad_ratio'] = zero_grad_params / total_params
        grad_stats['large_grad_count'] = large_grad_params

    return grad_stats


def evaluate_model(model, dataloader, criterion, device, log_file=None, save_predictions=False, save_dir=None):
    """评估模型 - 使用全面评估函数，显示完整指标"""

    # 获取标签映射（如果可能）
    label_maps = None
    if hasattr(dataloader.dataset, 'label_maps'):
        label_maps = dataloader.dataset.label_maps

    # 调用全面评估函数
    comprehensive_metrics = evaluate_comprehensive(
        model,
        dataloader,
        criterion,
        device,
        label_maps=label_maps,
        output_dir=save_dir,
        prefix="val",  # 验证模式
        save_predictions=save_predictions
    )

    # 转换格式以兼容原有代码
    simple_metrics = {'loss': comprehensive_metrics['loss']}
    for task in ['emotion', 'thinking', 'intent', 'stance']:
        if task in comprehensive_metrics:
            simple_metrics[f'{task}_acc'] = comprehensive_metrics[task]['accuracy'] * 100

    # 打印验证结果到日志（保持原有格式）
    if log_file:
        val_log = f"\nEpoch 验证结果:\n"
        val_log += f"验证损失: {comprehensive_metrics['loss']:.4f}\n"
        for task in ['emotion', 'thinking', 'intent', 'stance']:
            acc = comprehensive_metrics[task]['accuracy'] * 100
            val_log += f"{task} 准确率: {acc:.2f}%\n"

        # 添加完整指标到日志
        if 'holistic' in comprehensive_metrics:
            hol = comprehensive_metrics['holistic']
            val_log += f"完整指标 - Exact Match(4-All): {hol['exact_match']:.4f}, "
            val_log += f"3-All: {hol['three_all']:.4f}, "
            val_log += f"2-All: {hol['two_all']:.4f}, "
            val_log += f"Hamming Loss: {hol['hamming_loss']:.4f}\n"

        log_file.write(val_log)
        log_file.flush()

    return simple_metrics


def evaluate_comprehensive(model, dataloader, criterion, device, label_maps=None, output_dir=None, prefix="test", save_predictions=False):
    """
    全面评估函数 - 计算详细指标并可选保存预测结果
    """
    model.eval()
    total_loss = 0.0
    tasks = ['emotion', 'thinking', 'intent', 'stance']

    # 存储用于计算指标的列表
    all_preds = {task: [] for task in tasks}
    all_labels = {task: [] for task in tasks}

    # 存储用于保存JSON的详细列表 (新增)
    detailed_predictions = []
    sample_idx_counter = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Comprehensive Evaluation ({prefix})", leave=False):
            features = batch['features'].to(device)
            labels_batch = {task: batch[task].to(device) for task in tasks}

            outputs, intermediate_features = model(features, return_features=True)
            loss = criterion(outputs, labels_batch, intermediate_features, model.hyperbolic_layer)
            total_loss += loss.item()

            # 批次级预测处理
            batch_preds = {}
            for task in tasks:
                preds = torch.argmax(outputs[task], dim=-1)
                all_preds[task].extend(preds.cpu().numpy())
                all_labels[task].extend(labels_batch[task].cpu().numpy())
                batch_preds[task] = preds.cpu().numpy()

            # 如果需要保存详细预测结果
            if save_predictions:
                batch_size = features.size(0)
                for i in range(batch_size):
                    item_result = {
                        "sample_id": sample_idx_counter,
                        "ground_truth": {t: int(labels_batch[t][i].item()) for t in tasks},
                        "prediction": {t: int(batch_preds[t][i]) for t in tasks}
                    }
                    detailed_predictions.append(item_result)
                    sample_idx_counter += 1

    # --- 1. 基础指标计算 ---
    avg_loss = total_loss / len(dataloader)
    metrics = {'loss': avg_loss}

    # --- 2. 详细指标计算 (Acc, F1) ---
    for task in tasks:
        y_true = np.array(all_labels[task])
        y_pred = np.array(all_preds[task])

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        metrics[task] = {
            'accuracy': acc,
            'macro_f1': f1,
        }

        # 为了兼容原有的日志格式，添加扁平化的key
        metrics[f'{task}_acc'] = acc * 100

    # --- 3. 全景一致性指标 (Holistic) - 4-All等 ---
    # 需要将不同维度的预测对齐到样本级别
    num_samples = len(all_labels['emotion'])

    # 构建样本级矩阵 [N_samples, N_tasks]
    y_true_matrix = np.zeros((num_samples, 4), dtype=int)
    y_pred_matrix = np.zeros((num_samples, 4), dtype=int)

    for i, task in enumerate(tasks):
        y_true_matrix[:, i] = all_labels[task]
        y_pred_matrix[:, i] = all_preds[task]

    # 计算每个样本正确的维度数量
    correct_matrix = (y_true_matrix == y_pred_matrix)  # [N, 4] bool
    correct_counts = correct_matrix.sum(axis=1)  # [N] (0-4)

    exact_match = np.mean(correct_counts == 4)  # 4-All (Exact Match)
    three_all = np.mean(correct_counts >= 3)     # 3-All
    two_all = np.mean(correct_counts >= 2)       # 2-All

    # Hamming Loss (错误标签的比例)
    hamming = 1.0 - np.mean(correct_matrix)

    metrics['holistic'] = {
        'exact_match': exact_match,
        'three_all': three_all,
        'two_all': two_all,
        'hamming_loss': hamming
    }

    # --- 2. 保存报告和数据 ---
    if output_dir:
        # 保存原有格式的报告
        generate_evaluation_report_files(metrics, output_dir, prefix)

        # 新增：保存详细预测 JSON (带时间戳，避免覆盖)
        if save_predictions:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pred_json_path = os.path.join(output_dir, f"{prefix}_predictions_{timestamp}.json")
            with open(pred_json_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_predictions, f, ensure_ascii=False, indent=2)
            print(f"📄 详细预测结果已保存至: {pred_json_path}")

        # 打印控制台信息 (保持原有逻辑，显示完整指标)
        print(f"\n🎯 {prefix.upper()} 评估结果:")
        print(f"  Loss: {avg_loss:.4f}")
        for task in tasks:
            print(f"  {task.upper()}: ACC={metrics[task]['accuracy']:.4f}, Macro-F1={metrics[task]['macro_f1']:.4f}")
        if 'holistic' in metrics:
            hol = metrics['holistic']
            print(f"  Exact Match (4-All): {hol['exact_match']:.4f}")
            print(f"  3-All:               {hol['three_all']:.4f}")
            print(f"  2-All:               {hol['two_all']:.4f}")
            print(f"  Hamming Loss:        {hol['hamming_loss']:.4f}")

    return metrics


def generate_evaluation_report_files(metrics, output_dir, prefix="test"):
    """生成与evaluate.py格式一致的详细评估报告文件"""

    # 1. 保存 JSON 指标
    json_path = os.path.join(output_dir, f"{prefix}_detailed_metrics.json")
    # 过滤掉无法序列化的对象
    json_metrics = {}
    for k, v in metrics.items():
        if k == 'holistic':
            json_metrics[k] = v
        elif isinstance(v, dict) and 'accuracy' in v:
            json_metrics[k] = {
                'accuracy': v['accuracy'],
                'macro_f1': v['macro_f1']
            }
        elif k == 'loss':
            json_metrics[k] = v

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_metrics, f, ensure_ascii=False, indent=2)

    # 2. 保存 CSV 指标 (便于Excel分析)
    csv_path = os.path.join(output_dir, f"{prefix}_detailed_metrics.csv")
    rows = []
    tasks = ['emotion', 'thinking', 'intent', 'stance']

    # 维度指标
    for task in tasks:
        if task in metrics:
            rows.append({
                "Dimension": task.capitalize(),
                "Accuracy": f"{metrics[task]['accuracy']:.4f}",
                "Macro-F1": f"{metrics[task]['macro_f1']:.4f}",
                "Accuracy_%": f"{metrics[task]['accuracy']*100:.2f}%"
            })

    # 全景指标
    if 'holistic' in metrics:
        holistic = metrics['holistic']
        rows.extend([
            {"Dimension": "Exact Match (4-All)", "Accuracy": f"{holistic['exact_match']:.4f}", "Macro-F1": "N/A", "Accuracy_%": f"{holistic['exact_match']*100:.2f}%"},
            {"Dimension": "3-All", "Accuracy": f"{holistic['three_all']:.4f}", "Macro-F1": "N/A", "Accuracy_%": f"{holistic['three_all']*100:.2f}%"},
            {"Dimension": "2-All", "Accuracy": f"{holistic['two_all']:.4f}", "Macro-F1": "N/A", "Accuracy_%": f"{holistic['two_all']*100:.2f}%"},
            {"Dimension": "Hamming Loss", "Accuracy": f"{holistic['hamming_loss']:.4f}", "Macro-F1": "N/A", "Accuracy_%": "N/A"}
        ])

    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8')

    # 3. 生成详细的 TXT 报告 (人类可读)
    txt_path = os.path.join(output_dir, f"{prefix}_comprehensive_report.txt")
    lines = []
    lines.append("=" * 80)
    lines.append(f"HCN模型详细评估报告 ({prefix.upper()})")
    lines.append("=" * 80)
    lines.append(f"📅 评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"🔢 样本数量: {len(all_labels if 'all_labels' in locals() else 'N/A')}")
    lines.append(f"📊 总体Loss: {metrics.get('loss', 0.0):.4f}")
    lines.append("")
    lines.append("🎯 维度独立指标:")
    lines.append("-" * 50)
    lines.append(f"{'认知维度':<15} {'准确率':<12} {'Macro-F1':<12} {'性能评级'}")
    lines.append("-" * 50)

    # 性能评级函数
    def get_performance_grade(acc, f1):
        avg_score = (acc + f1) / 2
        if avg_score >= 0.8:
            return "🟢 优秀"
        elif avg_score >= 0.6:
            return "🟡 良好"
        elif avg_score >= 0.4:
            return "🟠 一般"
        else:
            return "🔴 较差"

    for task in tasks:
        if task in metrics:
            acc = metrics[task]['accuracy']
            f1 = metrics[task]['macro_f1']
            grade = get_performance_grade(acc, f1)
            lines.append(f"{task.capitalize():<15} {acc:<12.4f} {f1:<12.4f} {grade}")

    lines.append("")
    if 'holistic' in metrics:
        hol = metrics['holistic']
        lines.append("🌟 全景一致性指标:")
        lines.append("-" * 50)
        lines.append(f"Exact Match (4-All):  {hol['exact_match']:.4f} ({hol['exact_match']*100:.2f}%)")
        lines.append(f"3-All及以上:         {hol['three_all']:.4f} ({hol['three_all']*100:.2f}%)")
        lines.append(f"2-All及以上:         {hol['two_all']:.4f} ({hol['two_all']*100:.2f}%)")
        lines.append(f"Hamming Loss:        {hol['hamming_loss']:.4f}")
        lines.append("")
        lines.append("💡 指标说明:")
        lines.append("• Exact Match (4-All): 四个认知维度全部预测正确")
        lines.append("• 3-All及以上: 至少三个维度预测正确")
        lines.append("• 2-All及以上: 至少两个维度预测正确")
        lines.append("• Hamming Loss: 错误预测的平均比例 (越低越好)")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"📄 详细报告已保存至: {txt_path}")


def load_model_for_evaluation(checkpoint_path, device):
    """从检查点加载模型用于评估"""
    print(f"🔄 正在加载模型检查点: {checkpoint_path}")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 尝试从检查点中推断模型配置
    model_config = {}

    # 方法1: 如果检查点中保存了配置
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    # 方法2: 从模型状态字典推断
    else:
        # 从第一层推断input_dim
        first_param = next(iter(checkpoint['model_state_dict'].values()))
        if len(first_param.shape) >= 2:
            model_config['input_dim'] = first_param.shape[1]

    # 如果还是没有，使用默认配置
    if 'input_dim' not in model_config:
        model_config.update({
            'input_dim': 4096,  # 默认值，可以根据实际情况调整
            'hidden_dim': 512,
            'n_layers': 4,
            'n_heads': 8,
            'dropout': 0.1,
            'curvature': 1.0,
            'learnable_curvature': False
        })

    print(f"🔧 检测到模型配置: {model_config}")

    # 创建模型
    model = HCNModel(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config.get('hidden_dim', 512),
        n_layers=model_config.get('n_layers', 4),
        n_heads=model_config.get('n_heads', 8),
        dropout=model_config.get('dropout', 0.1),
        curvature=model_config.get('curvature', 1.0),
        learnable_curvature=model_config.get('learnable_curvature', False),
        hyperbolic_scale=model_config.get('hyperbolic_scale', 0.05),
        use_hyperbolic=model_config.get('use_hyperbolic', True),
        use_transformer=model_config.get('use_transformer', True)
    ).to(device)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✅ 模型加载成功!")
    return model, model_config


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs, global_progress, log_file):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    # 训练状态信息
    train_info = {
        'current_loss': 0.0,
        'avg_loss': 0.0,
        'detailed_losses': {}
    }

    # 使用普通的数据加载器，不创建额外的进度条
    for batch_idx, batch in enumerate(dataloader):
        features = batch['features'].to(device)
        labels = {task: batch[task].to(device) for task in ['emotion', 'thinking', 'intent', 'stance']}

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播，获取中间特征用于损失计算
        outputs, intermediate_features = model(features, return_features=True)

        # 计算损失，使用模型生成的中间特征而不是原始输入特征
        loss = criterion(outputs, labels, intermediate_features, model.hyperbolic_layer)
        total_loss += loss.item()

        # 获取详细的损失信息
        current_loss = loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        train_info['current_loss'] = current_loss
        train_info['avg_loss'] = avg_loss

        if hasattr(criterion, 'last_losses'):
            detailed_losses = criterion.last_losses
            train_info['detailed_losses'] = detailed_losses

        # 更新全局进度条
        global_progress.update(1)

        # 更新进度条信息
        postfix_dict = {
            'Epoch': f'{epoch+1}/{total_epochs}',
            'Loss': f'{current_loss:.3f}',
            'Avg': f'{avg_loss:.3f}'
        }

        if hasattr(criterion, 'last_losses'):
            detailed_losses = criterion.last_losses
            postfix_dict.update({
                'Task': f'{detailed_losses.get("task_loss", 0):.3f}'
            })

        global_progress.set_postfix(postfix_dict)

        # 记录到日志文件（但不打印到控制台，避免干扰进度条）
        global_step = epoch * num_batches + batch_idx
        if log_file:
            if batch_idx == 0 or batch_idx % 10 == 0:  # 每10个批次记录一次
                if hasattr(criterion, 'last_losses'):
                    detailed_losses = criterion.last_losses
                    detailed_log = (f"Epoch {epoch+1}, Step {global_step}, Batch {batch_idx+1}/{num_batches}\n"
                                  f"  总损失: {detailed_losses['total']:.4f}\n"
                                  f"  任务损失: {detailed_losses['task_loss']:.4f}\n"
                                  f"  双曲损失: {detailed_losses['hyper_loss']:.4f}\n"
                                  f"  对比损失: {detailed_losses['contrastive_loss']:.4f}\n"
                                  f"  各任务损失 - 情感: {detailed_losses['emotion']:.4f}, "
                                  f"思维: {detailed_losses['thinking']:.4f}, "
                                  f"意图: {detailed_losses['intent']:.4f}, "
                                  f"立场: {detailed_losses['stance']:.4f}\n")
                else:
                    detailed_log = f"Epoch {epoch+1}, Step {global_step}, Batch Loss: {current_loss:.4f}\n"

                log_file.write(detailed_log + "\n")
                log_file.flush()

        # 反向传播
        loss.backward()

        # --- 梯度检查和打印 ---
        # 梯度裁剪 (clip_grad_norm_ 会返回总范数)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 计算梯度统计信息
        grad_stats = compute_gradient_stats(model)

        # 打印梯度信息（每10个批次打印一次）
        # if batch_idx % 10 == 0:
        #     grad_log = (f"\n📊 梯度统计 - Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}:\n"
        #                f"  总梯度范数: {grad_norm:.6f}\n"
        #                f"  参数梯度均值: {grad_stats['param_grad_mean']:.6f}\n"
        #                f"  参数梯度标准差: {grad_stats['param_grad_std']:.6f}\n"
        #                f"  参数梯度最大值: {grad_stats['param_grad_max']:.6f}\n"
        #                f"  参数梯度最小值: {grad_stats['param_grad_min']:.6f}\n"
        #                f"  零梯度参数比例: {grad_stats['zero_grad_ratio']:.2%}\n"
        #                f"  大梯度参数数量(>0.01): {grad_stats['large_grad_count']}\n"
        #                f"  当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

        #     print(grad_log)

        #     if log_file:
        #         log_file.write(grad_log + "\n")
        #         log_file.flush()

        # 如果梯度本身已经是 NaN/Inf，跳过这一步更新
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"⚠️ Warning: 梯度爆炸检测到 (norm={grad_norm}). Skipping step.")
            if log_file:
                log_file.write(f"警告: 梯度爆炸检测到 (norm={grad_norm})，跳过更新\n")
                log_file.flush()
            optimizer.zero_grad() # 清除坏梯度
        else:
            # 更新参数
            optimizer.step()

    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, epoch, metrics_or_loss, path):
    """保存检查点 - 支持保存完整指标或仅损失"""

    # 构建模型配置（包含消融实验参数）
    model_config = {
        'input_dim': model.input_dim,
        'hidden_dim': model.hidden_dim,
        'use_hyperbolic': getattr(model, 'use_hyperbolic', True),
        'use_transformer': getattr(model, 'use_transformer', True)
    }

    # 判断是完整的指标字典还是仅仅损失值
    if isinstance(metrics_or_loss, dict):
        # 完整的指标字典
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': metrics_or_loss['loss'],
            'metrics': metrics_or_loss,  # 保存完整指标
            'model_config': model_config  # 保存模型配置
        }
    else:
        # 仅仅是损失值
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': metrics_or_loss,
            'model_config': model_config  # 保存模型配置
        }

    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description="HCN模型训练 (集成详细评估版)")

    # 模式选择
    parser.add_argument("--eval_only", action="store_true",
                       help="仅评估模式：加载已训练模型并在测试集上生成详细报告")

    # 数据参数
    parser.add_argument("--train_path", type=str, help="训练特征路径 (训练模式必需)")
    parser.add_argument("--val_path", type=str, help="验证特征路径")
    parser.add_argument("--test_path", type=str, help="测试特征路径")

    # 模型加载参数 (评估模式)
    parser.add_argument("--model_path", type=str, help="模型检查点路径 (评估模式必需)")

    # 模型参数
    parser.add_argument("--input_dim", type=int, default=768, help="输入特征维度")
    parser.add_argument("--hidden_dim", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--n_layers", type=int, default=4, help="Transformer层数")
    parser.add_argument("--n_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    parser.add_argument("--curvature", type=float, default=1.0, help="双曲空间曲率")
    parser.add_argument("--learnable_curvature", action="store_true", help="是否学习曲率参数")
    parser.add_argument("--hyperbolic_scale", type=float, default=0.05, help="双曲缩放因子（建议小于0.1）")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 损失函数参数
    parser.add_argument("--task_weights", nargs=4, type=float, default=[1.0, 1.0, 1.0, 1.0],
                        help="四个任务的权重")
    parser.add_argument("--use_uncertainty_weighting", action="store_true",
                        help="使用不确定性加权")
    parser.add_argument("--lambda_hyper", type=float, default=0.1, help="双曲空间正则化权重")
    parser.add_argument("--lambda_contrastive", type=float, default=0.1, help="对比学习权重")
    parser.add_argument("--use_class_weights", action="store_true",
                        help="是否使用类别权重来处理类别不平衡问题")

    # 消融实验参数
    parser.add_argument("--disable_hyperbolic", action="store_true",
                       help="消融实验：禁用双曲空间（使用欧氏空间）")
    parser.add_argument("--disable_transformer", action="store_true",
                       help="消融实验：禁用跨维度Transformer交互")

    # 保存和日志
    parser.add_argument("--output_dir", type=str, default="./checkpoints/hcn_model", help="输出目录")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志目录")
    parser.add_argument("--save_every", type=int, default=50, help="每多少轮保存一次")
    parser.add_argument("--eval_every", type=int, default=10, help="每多少轮评估一次")

    # 设备
    parser.add_argument("--device", type=str, default="auto", help="设备选择 (auto, cpu, cuda)")
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=[0, 1], help="使用的GPU ID列表 (兼容参数，建议使用CUDA_VISIBLE_DEVICES)")

    args = parser.parse_args()

    # 欧氏模式下自动禁用双曲正则化Loss
    if args.disable_hyperbolic:
        print("⚠️ 欧氏模式已启用：自动禁用双曲正则化Loss")
        args.lambda_hyper = 0.0

    # 检查CUDA_VISIBLE_DEVICES环境变量
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        print(f"CUDA_VISIBLE_DEVICES={cuda_visible}")

    # 设置设备 - 适配CUDA_VISIBLE_DEVICES环境变量
    if args.device == "auto":
        if torch.cuda.is_available():
            # 使用CUDA_VISIBLE_DEVICES环境变量时，第一个可见GPU就是cuda:0
            device = torch.device("cuda:0")
            print(f"使用设备: {device}")
        else:
            device = torch.device("cpu")
            print("使用设备: cpu (CUDA不可用)")
    else:
        device = torch.device(args.device)
        print(f"使用设备: {device}")

    # 显示GPU信息
    if device.type == 'cuda':
        print(f"可见GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"默认使用GPU {device.index} 进行训练")

    # 参数验证
    if args.eval_only:
        if not args.model_path:
            raise ValueError("评估模式需要提供 --model_path 参数")
        if not args.test_path:
            raise ValueError("评估模式需要提供 --test_path 参数")
    else:
        if not args.train_path:
            raise ValueError("训练模式需要提供 --train_path 参数")

    # 设置随机种子
    set_seed(args.seed)

    # 评估模式处理
    if args.eval_only:
        print("🔍 运行评估模式...")

        # 输出目录设置（基于模型路径）
        model_dir = os.path.dirname(args.model_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = os.path.join(model_dir, f"evaluation_results_{timestamp}")
        os.makedirs(eval_dir, exist_ok=True)

        # 加载测试数据
        from dataset_tensor import CognitiveTensorDataset

        if args.train_path:
            # ... 您之前的逻辑 ...
            print(f"📉 正在加载训练集统计量: {args.train_path}")
            # 注意：这里只需临时加载，不需要保存数据集对象，只要 mean/std
            train_ds_temp = CognitiveTensorDataset(args.train_path, normalize=True)
            train_mean = train_ds_temp.norm_mean
            train_std = train_ds_temp.norm_std
        else:
            print("⚠️ 未提供训练集路径，使用测试集自身统计量")
            train_mean = None
            train_std = None


        test_dataset = CognitiveTensorDataset(
            args.test_path, 
            normalize=True, 
            mean=train_mean,  # 👈 传入训练集均值
            std=train_std     # 👈 传入训练集方差
        )

        # # 1. 先加载训练集获取统计量
        # if args.train_path:
        #     print(f"📉 正在加载训练集统计量: {args.train_path}")
        #     train_ds_temp = CognitiveTensorDataset(args.train_path, normalize=True)
        #     train_mean = train_ds_temp.norm_mean
        #     train_std = train_ds_temp.norm_std
        # else:
        #     print("⚠️ 未提供训练集路径，使用测试集自身统计量")
        #     train_mean = None
        #     train_std = None

        # test_dataset = CognitiveTensorDataset(args.test_path, normalize=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # 获取标签映射
        label_maps = test_dataset.label_maps

        # 加载模型
        model, model_config = load_model_for_evaluation(args.model_path, device)

        # 创建损失函数（用于计算测试集损失）
        if args.use_class_weights:
            class_weights = compute_class_weights(args.test_path)
        else:
            class_weights = None

        criterion = HCNMutliTaskLoss(
            task_weights=args.task_weights,
            use_uncertainty_weighting=args.use_uncertainty_weighting,
            lambda_hyper=args.lambda_hyper,
            lambda_contrastive=args.lambda_contrastive
        )

        if class_weights:
            criterion.set_class_weights(class_weights)
            print("已设置评估类别权重")

        # 运行评估
        print("📊 开始评估模型...")
        test_metrics = evaluate_comprehensive(
            model, test_loader, criterion, device, label_maps,
            output_dir=eval_dir, prefix="test"
        )

        print(f"\n✅ 评估完成!")
        print(f"  结果保存在: {eval_dir}")
        print(f"  测试集Exact Match (4-All): {test_metrics['holistic']['exact_match']:.4f}")

        return

    # 训练模式继续...
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, f"model_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # 创建日志文件（保存在模型目录下）
    log_file_path = os.path.join(save_dir, "training.log")
    log_file = open(log_file_path, "w", encoding="utf-8")
    log_file.write(f"HCN模型训练开始 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"参数配置: {json.dumps(vars(args), indent=2, ensure_ascii=False)}\n\n")
    log_file.flush()

    # 保存参数
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # 创建数据加载器
    print("加载数据...")
    train_loader, val_loader, test_loader = create_data_loaders(
        args.train_path,
        args.val_path,
        args.test_path,
        batch_size=args.batch_size,
        shuffle=True
    )

    print(f"训练集: {len(train_loader.dataset)} 样本")
    if val_loader:
        print(f"验证集: {len(val_loader.dataset)} 样本")
    if test_loader:
        print(f"测试集: {len(test_loader.dataset)} 样本")

    # 计算类别权重并检测输入维度
    if args.use_class_weights:
        print("计算类别权重...")
        class_weights = compute_class_weights(args.train_path)
    else:
        print("不使用类别权重")
        class_weights = None

    # 检测实际的输入维度
    train_data = torch.load(args.train_path, map_location='cpu', weights_only=False)
    actual_input_dim = train_data['features'].shape[-1]  # 获取最后一个维度
    print(f"检测到输入维度: {actual_input_dim}")

    # 覆盖命令行参数中的input_dim
    if args.input_dim != actual_input_dim:
        print(f"覆盖命令行input_dim: {args.input_dim} -> {actual_input_dim}")
        args.input_dim = actual_input_dim

    # 创建模型
    print("创建模型...")
    model = HCNModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        curvature=args.curvature,
        learnable_curvature=args.learnable_curvature,
        hyperbolic_scale=args.hyperbolic_scale,
        use_hyperbolic=not args.disable_hyperbolic,
        use_transformer=not args.disable_transformer
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 显示消融实验设置
    print(f"🧪 消融实验设置:")
    print(f"  双曲空间: {'✅ 启用' if not args.disable_hyperbolic else '❌ 禁用 (欧氏模式)'}")
    print(f"  Transformer交互: {'✅ 启用' if not args.disable_transformer else '❌ 禁用 (各维度独立)'}")
    print(f"  双曲正则化权重: {args.lambda_hyper:.4f}")

    # 创建损失函数
    criterion = HCNMutliTaskLoss(
        task_weights=args.task_weights,
        use_uncertainty_weighting=args.use_uncertainty_weighting,
        lambda_hyper=args.lambda_hyper,
        lambda_contrastive=args.lambda_contrastive
    )
    

    # 设置类别权重（如果启用）
    if class_weights:
        criterion.set_class_weights(class_weights)
        print("已设置类别权重")
    else:
        print("未设置类别权重")

    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5
    )

    # 训练循环 - 无早停，保存最佳和最终模型
    best_val_loss = float('inf')
    best_avg_acc = 0.0  # 🏆 新增：跟踪最佳平均准确率
    total_batches = len(train_loader) * args.epochs  # 总批次数量

    print("开始训练...")
    print(f"训练计划: {args.epochs} 个epoch，共 {total_batches} 个批次")
    print("将保存: 1. 最佳Loss模型 2. 最佳准确率模型 3. 最终epoch模型")

    # 创建全局进度条（横跨所有epoch）
    global_progress = tqdm(
        total=total_batches,
        desc="Training Progress",
        ncols=240,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    for epoch in range(args.epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs, global_progress, log_file)

        # 在progress bar完成后显示epoch总结
        print(f"\nEpoch {epoch+1}/{args.epochs} 完成 - 训练损失: {train_loss:.4f}")
        if log_file:
            log_file.write(f"Epoch {epoch+1}/{args.epochs} 训练损失: {train_loss:.4f}\n")
            log_file.flush()

        # 评估
        if val_loader and epoch % args.eval_every == 0:
            # 创建验证结果保存目录
            val_output_dir = os.path.join(save_dir, f"val_results_epoch_{epoch+1}")
            os.makedirs(val_output_dir, exist_ok=True)

            # 获取完整指标
            label_maps = val_loader.dataset.label_maps if hasattr(val_loader.dataset, 'label_maps') else None
            comprehensive_metrics = evaluate_comprehensive(
                model, val_loader, criterion, device,
                label_maps=label_maps,
                output_dir=val_output_dir,
                prefix="val",
                save_predictions=True
            )

            # 转换格式以兼容原有代码
            val_metrics = {'loss': comprehensive_metrics['loss']}
            for task in ['emotion', 'thinking', 'intent', 'stance']:
                if task in comprehensive_metrics:
                    val_metrics[f'{task}_acc'] = comprehensive_metrics[task]['accuracy'] * 100

            # 打印完整验证结果到控制台
            print(f"\nEpoch {epoch+1}/{args.epochs} 验证结果:")
            print(f"验证损失: {val_metrics['loss']:.4f}")
            for task in ['emotion', 'thinking', 'intent', 'stance']:
                print(f"{task} 准确率: {val_metrics[f'{task}_acc']:.2f}%")

            # 显示完整指标
            if 'holistic' in comprehensive_metrics:
                hol = comprehensive_metrics['holistic']
                print(f"完整指标 - Exact Match(4-All): {hol['exact_match']:.4f}, "
                      f"3-All: {hol['three_all']:.4f}, "
                      f"2-All: {hol['two_all']:.4f}, "
                      f"Hamming Loss: {hol['hamming_loss']:.4f}")

            # 记录到日志文件
            if log_file:
                val_log = f"\nEpoch {epoch+1}/{args.epochs} 验证结果:\n"
                val_log += f"验证损失: {val_metrics['loss']:.4f}\n"
                for task in ['emotion', 'thinking', 'intent', 'stance']:
                    val_log += f"{task} 准确率: {val_metrics[f'{task}_acc']:.2f}%\n"
                if 'holistic' in comprehensive_metrics:
                    hol = comprehensive_metrics['holistic']
                    val_log += f"完整指标 - Exact Match(4-All): {hol['exact_match']:.4f}, "
                    val_log += f"3-All: {hol['three_all']:.4f}, "
                    val_log += f"2-All: {hol['two_all']:.4f}, "
                    val_log += f"Hamming Loss: {hol['hamming_loss']:.4f}\n"
                val_log += f"验证结果已保存至: {val_output_dir}\n"
                log_file.write(val_log + "\n")
                log_file.flush()

            # 计算平均准确率和其他指标用于模型选择
            avg_acc = (val_metrics['emotion_acc'] + val_metrics['thinking_acc'] +
                      val_metrics['intent_acc'] + val_metrics['stance_acc']) / 4.0

            # 1. 策略A：保存Loss最低的模型 (原有策略)
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_checkpoint(model, optimizer, epoch, val_metrics['loss'],
                              os.path.join(save_dir, "best_loss_model.pt"))
                print("  💾 [Loss] 最佳Loss模型已保存")

            # 2. 策略B：保存准确率最高的模型 (新策略 - 你需要的)
            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                save_checkpoint(model, optimizer, epoch, val_metrics,
                              os.path.join(save_dir, "best_acc_model.pt"))
                print(f"  🏆 [ACC] 最佳准确率模型已保存 (Avg Acc: {best_avg_acc:.4f})")

                # 🎯 集成功能：立即在测试集上生成详细报告（仅在发现更好准确率模型时）
                if test_loader:
                    print("📊 [Auto-Eval] 正在测试集上生成最终预测 JSON...")

                    # 确保输出目录存在
                    eval_output_dir = os.path.join(save_dir, "best_acc_eval_results")
                    os.makedirs(eval_output_dir, exist_ok=True)

                    # 调用修改后的评估函数，开启 save_predictions=True
                    test_comprehensive_metrics = evaluate_comprehensive(
                        model,
                        test_loader,
                        criterion,
                        device,
                        label_maps=train_loader.dataset.label_maps,
                        output_dir=eval_output_dir,
                        prefix="final_test",     # 文件名前缀
                        save_predictions=True    # 👈 关键：开启JSON保存
                    )

                    # 记录到日志
                    test_log = f"最佳模型测试集 Exact Match: {test_comprehensive_metrics['holistic']['exact_match']:.4f}\n"
                    test_log += f"测试集详细指标已保存至: {eval_output_dir}\n"
                    if log_file:
                        log_file.write(test_log)
                        log_file.flush()

            # 显示当前epoch的结果
            print(f"  📊 Epoch {epoch+1} 结果: Loss={val_metrics['loss']:.4f}, Avg Acc={avg_acc:.4f}")
            print(f"     - 情感: {val_metrics['emotion_acc']:.2f}%")
            print(f"     - 思维: {val_metrics['thinking_acc']:.2f}%")
            print(f"     - 意图: {val_metrics['intent_acc']:.2f}%")
            print(f"     - 立场: {val_metrics['stance_acc']:.2f}%")
            print(f"     - 最佳Loss: {best_val_loss:.4f}, 最佳Acc: {best_avg_acc:.4f}")

        # 更新学习率
        scheduler.step()
        lr_log = f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}"
        print(lr_log)
        if log_file:
            log_file.write(lr_log + "\n")
            log_file.flush()

        # 定期保存检查点
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)

    # 保存最终模型
    final_model_path = os.path.join(save_dir, "final_model.pt")
    if args.epochs > 0:
        save_checkpoint(model, optimizer, args.epochs-1, train_loss, final_model_path)
        print(f"💾 最终模型已保存: {final_model_path}")
    else:
        print("📝 跳过保存最终模型 (训练轮数为0)")

    # 关闭全局进度条
    global_progress.close()

    # 汇训练结果
    print(f"\n🎉 训练完成！")
    print(f"训练了 {args.epochs} 个epoch")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳平均准确率: {best_avg_acc:.4f}")
    print(f"模型保存位置: {save_dir}")
    print(f"  - 最佳损失模型: best_loss_model.pt (基于最低Loss)")
    print(f"  - 最佳准确率模型: best_acc_model.pt (基于最高准确率) 🏆")
    print(f"  - 最终模型: final_model.pt (训练结束模型)")

    # 如果需要测试，使用最佳准确率模型
    if test_loader and val_loader:
        print("\n📊 使用最佳准确率模型进行最终测试...")
        best_acc_model_path = os.path.join(save_dir, "best_acc_model.pt")
        if os.path.exists(best_acc_model_path):
            checkpoint = torch.load(best_acc_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ 已加载最佳准确率模型")
        else:
            # 如果没有最佳准确率模型，使用最佳Loss模型
            best_loss_model_path = os.path.join(save_dir, "best_loss_model.pt")
            checkpoint = torch.load(best_loss_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("⚠️ 未找到最佳准确率模型，使用最佳Loss模型")

    if test_loader:
        print("\n🔍 运行最终测试评估...")
        final_test_metrics = evaluate_comprehensive(
            model, test_loader, criterion, device,
            label_maps=train_loader.dataset.label_maps,
            output_dir=save_dir,
            prefix="final_test"
        )

        print("\n🎯 最终测试结果:")
        print(f"📊 测试集Loss: {final_test_metrics['loss']:.4f}")
        for task in ['emotion', 'thinking', 'intent', 'stance']:
            print(f"📈 {task.upper()}准确率: {final_test_metrics[task]['accuracy']*100:.2f}%")
        if 'holistic' in final_test_metrics:
            hol = final_test_metrics['holistic']
            print(f"🌟 Exact Match (4-All): {hol['exact_match']*100:.2f}%")
            print(f"🔢 3-All及以上: {hol['three_all']*100:.2f}%")
            print(f"🔢 2-All及以上: {hol['two_all']*100:.2f}%")

    print("\n训练完成！")

    # 关闭日志文件
    if log_file:
        completion_log = f"HCN模型训练完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        log_file.write(completion_log)
        log_file.close()
        print(f"训练日志已保存到: {log_file_path}")

    # 自动生成轻简化训练可视化（保存在模型目录下）
    print("\n🎨 正在生成轻简化训练过程可视化图表...")
    try:
        from simple_visualize import create_simple_visualization
        plot_dir = os.path.join(save_dir, "plots")
        create_simple_visualization(log_file_path, plot_dir)
        print(f"✅ 轻简化可视化图表已保存到: {plot_dir}")
    except Exception as e:
        print(f"⚠️  轻简化可视化生成失败: {e}")
        print("可以手动运行: python simple_visualize.py --log_file", log_file_path)

    # 显示完整保存路径和文件列表
    print(f"\n📁 所有文件已保存在: {save_dir}")
    print("📂 目录结构:")
    print("  ├── best_model.pt        # 最佳验证损失模型")
    print("  ├── final_model.pt       # 最终训练模型")
    print("  ├── training.log         # 训练日志")
    print("  ├── args.json           # 训练参数")
    print("  ├── checkpoint_epoch_*.pt # 定期检查点")
    print("  └── plots/               # 训练可视化图表")
    print("      ├── training_curves.png")
    print("      ├── training_curves.pdf")
    print("      └── main_curves.png")


if __name__ == "__main__":
    import math  # 确保math模块被导入
    main()