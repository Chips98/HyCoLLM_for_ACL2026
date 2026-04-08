#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析嵌入向量文件并生成可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import os
import sys

# 设置中文字体和绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 统一的绘图参数设置
PLOT_PARAMS = {
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8
}

# 应用绘图参数
plt.rcParams.update(PLOT_PARAMS)

def analyze_embeddings(embeddings_path):
    """分析嵌入向量文件"""
    print(f"正在加载嵌入向量文件: {embeddings_path}")

    # 加载嵌入数据
    data = torch.load(embeddings_path, map_location='cpu')

    print(f"数据键: {data.keys()}")
    print(f"样本数量: {data['num_samples']}")
    print(f"特征维度: {data['feature_dim']}")
    print(f"特征形状: {data['features'].shape}")

    # 分离四个维度的特征
    features = data['features']  # [num_samples, 4, feature_dim]
    labels = data['labels']
    label_maps = data['label_maps']

    # 四个维度的名称
    dimension_names = ['emotion', 'thinking', 'intent', 'stance']

    print("\n=== 基本统计分析 ===")

    # 为每个维度生成统计信息
    for i, dim_name in enumerate(dimension_names):
        dim_features = features[:, i, :]  # [num_samples, feature_dim]

        print(f"\n{dim_name.upper()} 维度:")
        print(f"  特征形状: {dim_features.shape}")
        print(f"  均值: {dim_features.mean().item():.6f}")
        print(f"  标准差: {dim_features.std().item():.6f}")
        print(f"  最小值: {dim_features.min().item():.6f}")
        print(f"  最大值: {dim_features.max().item():.6f}")
        print(f"  范围: {dim_features.max().item() - dim_features.min().item():.6f}")

        # 检查零方差特征
        feat_flat = dim_features.view(dim_features.size(0), -1)
        zero_var_count = (feat_flat.std(dim=0) < 1e-6).sum().item()
        total_features = feat_flat.size(1)
        print(f"  零方差特征: {zero_var_count}/{total_features} ({zero_var_count/total_features*100:.2f}%)")

    print("\n=== 维度间相似性分析 ===")

    # 计算维度间的相似性
    dimension_similarities = np.zeros((4, 4))

    for i in range(4):
        for j in range(4):
            # 计算所有样本中维度i和维度j的平均余弦相似度
            similarities = []
            for sample_idx in range(min(1000, features.size(0))):  # 采样1000个样本以提高效率
                feat_i = features[sample_idx, i, :]
                feat_j = features[sample_idx, j, :]
                cos_sim = torch.nn.functional.cosine_similarity(
                    feat_i.unsqueeze(0), feat_j.unsqueeze(0)
                ).item()
                similarities.append(cos_sim)

            dimension_similarities[i, j] = np.mean(similarities)

    print("维度间平均余弦相似度矩阵:")
    for i, dim_name in enumerate(dimension_names):
        row = "  ".join([f"{dimension_similarities[i, j]:.3f}" for j in range(4)])
        print(f"  {dim_name[:8].upper():<8}: {row}")

    return data, dimension_similarities

def visualize_embeddings(data, dimension_similarities, output_dir):
    """生成嵌入向量的可视化"""

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    features = data['features']  # [num_samples, 4, feature_dim]
    labels = data['labels']
    label_maps = data['label_maps']

    dimension_names = ['emotion', 'thinking', 'intent', 'stance']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # 红色、青色、蓝色、绿色

    print(f"\n=== 开始生成可视化图表，保存到: {output_dir} ===")

    # 1. 维度间相似性热力图
    plt.figure(figsize=PLOT_PARAMS['figure.figsize'])
    sns.heatmap(dimension_similarities,
                xticklabels=[name.upper() for name in dimension_names],
                yticklabels=[name.upper() for name in dimension_names],
                annot=True, cmap='coolwarm', center=0.5,
                square=True, linewidths=1, linecolor='white',
                fmt='.3f', cbar_kws={'label': '余弦相似度'})
    plt.title('四个认知维度间的平均余弦相似度', fontsize=PLOT_PARAMS['axes.titlesize'], pad=20)
    plt.xlabel('认知维度', fontsize=PLOT_PARAMS['axes.labelsize'])
    plt.ylabel('认知维度', fontsize=PLOT_PARAMS['axes.labelsize'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dimension_similarity_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 生成维度相似性热力图")

    # 2. 特征分布直方图（每个维度）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (dim_name, color) in enumerate(zip(dimension_names, colors)):
        dim_features = features[:, i, :].flatten().numpy()

        axes[i].hist(dim_features, bins=50, alpha=0.7, color=color, density=True)
        axes[i].axvline(dim_features.mean(), color='black', linestyle='--', linewidth=2,
                       label=f'均值: {dim_features.mean():.3f}')
        axes[i].set_title(f'{dim_name.upper()} 维度特征分布', fontsize=PLOT_PARAMS['axes.titlesize'])
        axes[i].set_xlabel('特征值', fontsize=PLOT_PARAMS['axes.labelsize'])
        axes[i].set_ylabel('密度', fontsize=PLOT_PARAMS['axes.labelsize'])
        axes[i].legend(fontsize=PLOT_PARAMS['legend.fontsize'])
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('四个认知维度的特征分布', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 生成特征分布直方图")

    # 3. PCA降维可视化
    print("正在进行PCA降维...")

    # 对每个维度的特征进行PCA降维到2D
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (dim_name, color) in enumerate(zip(dimension_names, colors)):
        dim_features = features[:, i, :].numpy()  # [num_samples, feature_dim]

        # 如果特征维度太高，先用PCA降到50维
        if dim_features.shape[1] > 50:
            pca_pre = PCA(n_components=50)
            dim_features = pca_pre.fit_transform(dim_features)
            print(f"{dim_name} PCA(50) 解释方差比: {pca_pre.explained_variance_ratio_.sum():.3f}")

        # 再降到2D进行可视化
        pca = PCA(n_components=2)
        dim_features_2d = pca.fit_transform(dim_features)

        # 按标签着色
        unique_labels = np.unique(labels[dim_name])
        label_names = label_maps[dim_name]

        scatter = axes[i].scatter(dim_features_2d[:, 0], dim_features_2d[:, 1],
                                 c=labels[dim_name], cmap='tab10', alpha=0.6, s=20)

        axes[i].set_title(f'{dim_name.upper()} 维度 - PCA可视化', fontsize=PLOT_PARAMS['axes.titlesize'])
        axes[i].set_xlabel(f'PC1 (解释方差: {pca.explained_variance_ratio_[0]:.1%})',
                          fontsize=PLOT_PARAMS['axes.labelsize'])
        axes[i].set_ylabel(f'PC2 (解释方差: {pca.explained_variance_ratio_[1]:.1%})',
                          fontsize=PLOT_PARAMS['axes.labelsize'])
        axes[i].grid(True, alpha=0.3)

        # 添加图例
        legend1 = axes[i].legend(handles=scatter.legend_elements()[0],
                                labels=[label_names[idx] for idx in unique_labels],
                                title=dim_name.upper(), loc='best', fontsize=10)
        axes[i].add_artist(legend1)

    plt.suptitle('四个认知维度的PCA降维可视化', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 生成PCA可视化")

    # 4. t-SNE降维可视化（如果样本数量不太大）
    if features.shape[0] <= 5000:
        print("正在进行t-SNE降维...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for i, (dim_name, color) in enumerate(zip(dimension_names, colors)):
            dim_features = features[:, i, :].numpy()

            # 如果特征维度太高，先用PCA降到50维
            if dim_features.shape[1] > 50:
                pca = PCA(n_components=min(50, dim_features.shape[1] // 10))
                dim_features = pca.fit_transform(dim_features)

            # t-SNE降维
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, dim_features.shape[0] // 4))
            dim_features_2d = tsne.fit_transform(dim_features)

            # 按标签着色
            unique_labels = np.unique(labels[dim_name])
            label_names = label_maps[dim_name]

            scatter = axes[i].scatter(dim_features_2d[:, 0], dim_features_2d[:, 1],
                                     c=labels[dim_name], cmap='tab10', alpha=0.6, s=20)

            axes[i].set_title(f'{dim_name.upper()} 维度 - t-SNE可视化', fontsize=PLOT_PARAMS['axes.titlesize'])
            axes[i].set_xlabel('t-SNE 1', fontsize=PLOT_PARAMS['axes.labelsize'])
            axes[i].set_ylabel('t-SNE 2', fontsize=PLOT_PARAMS['axes.labelsize'])
            axes[i].grid(True, alpha=0.3)

            # 添加图例
            legend1 = axes[i].legend(handles=scatter.legend_elements()[0],
                                    labels=[label_names[idx] for idx in unique_labels],
                                    title=dim_name.upper(), loc='best', fontsize=10)
            axes[i].add_artist(legend1)

        plt.suptitle('四个认知维度的t-SNE降维可视化', fontsize=18, y=1.02)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tsne_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 生成t-SNE可视化")
    else:
        print(f"样本数量过多 ({features.shape[0]})，跳过t-SNE可视化")

    # 5. 标签分布统计
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, dim_name in enumerate(dimension_names):
        label_counts = {}
        label_names = label_maps[dim_name]

        for label_id in labels[dim_name]:
            label_name = label_names[label_id.item()]
            label_counts[label_name] = label_counts.get(label_name, 0) + 1

        # 绘制条形图
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        labels_list, counts_list = zip(*sorted_labels)

        bars = axes[i].bar(range(len(labels_list)), counts_list, color=colors[i], alpha=0.7)
        axes[i].set_title(f'{dim_name.upper()} 标签分布', fontsize=PLOT_PARAMS['axes.titlesize'])
        axes[i].set_xlabel('标签', fontsize=PLOT_PARAMS['axes.labelsize'])
        axes[i].set_ylabel('数量', fontsize=PLOT_PARAMS['axes.labelsize'])
        axes[i].set_xticks(range(len(labels_list)))
        axes[i].set_xticklabels(labels_list, rotation=45, ha='right')
        axes[i].grid(True, alpha=0.3, axis='y')

        # 在条形图上添加数值
        for bar, count in zip(bars, counts_list):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_list)*0.01,
                        str(count), ha='center', va='bottom', fontsize=PLOT_PARAMS['font.size']-2)

    plt.suptitle('四个认知维度的标签分布', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/label_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 生成标签分布图")

    print(f"\n✅ 所有可视化图表已保存到: {output_dir}")
    print("生成的文件:")
    print("  - dimension_similarity_heatmap.png: 维度相似性热力图")
    print("  - feature_distributions.png: 特征分布直方图")
    print("  - pca_visualization.png: PCA降维可视化")
    if features.shape[0] <= 5000:
        print("  - tsne_visualization.png: t-SNE降维可视化")
    print("  - label_distribution.png: 标签分布统计")

def main():
    # 文件路径
    embeddings_path = "../data/processed/cut/cut_train_embeddings.pt"
    output_dir = "../HCN/03_feature_extraction/embedding_analysis"

    if not os.path.exists(embeddings_path):
        print(f"❌ 嵌入文件不存在: {embeddings_path}")
        return

    # 分析嵌入向量
    data, dimension_similarities = analyze_embeddings(embeddings_path)

    # 生成可视化
    visualize_embeddings(data, dimension_similarities, output_dir)

    print(f"\n📊 分析完成！")
    print(f"请查看 {output_dir} 目录中的可视化图表")

if __name__ == "__main__":
    main()