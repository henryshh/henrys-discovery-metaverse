"""
生成全量 DTW 聚类完整报告
"""
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("全量 DTW 聚类报告生成")
print("="*60)

# 加载结果
with open('output/dtw_full_results.json', 'r') as f:
    results = json.load(f)

with open('output/dtw_cache.pkl', 'rb') as f:
    dist_matrix = pickle.load(f)

labels = np.array(results['labels'])
n_clusters = results['n_clusters']

print(f"\n总曲线数: {results['total_curves']}")
print(f"聚类数: {n_clusters}")
print(f"距离矩阵形状: {dist_matrix.shape}")

# 聚类统计
print("\n" + "="*60)
print("聚类分布")
print("="*60)

cluster_stats = {}
for i in range(n_clusters):
    count = np.sum(labels == i)
    cluster_stats[i] = count
    print(f"Cluster {i}: {count} 条 ({count/len(labels)*100:.1f}%)")

# 生成层次聚类树状图
print("\n生成树状图...")
condensed = squareform(dist_matrix)
linkage_matrix = linkage(condensed, method='ward')

fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(linkage_matrix, ax=ax, truncate_mode='lastp', p=30, 
           leaf_rotation=90, leaf_font_size=10)
ax.set_xlabel('Sample Index or (Cluster Size)')
ax.set_ylabel('Distance')
ax.set_title('DTW Hierarchical Clustering Dendrogram')
plt.tight_layout()
plt.savefig('output/dtw_dendrogram_full.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: output/dtw_dendrogram_full.png")

# 生成聚类分布饼图
print("\n生成聚类分布图...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 饼图
ax1 = axes[0]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
sizes = [cluster_stats[i] for i in range(n_clusters)]
ax1.pie(sizes, labels=[f'Cluster {i}' for i in range(n_clusters)], 
        colors=colors, autopct='%1.1f%%', startangle=90)
ax1.set_title('DTW Cluster Distribution (170 curves)')

# 柱状图
ax2 = axes[1]
bars = ax2.bar(range(n_clusters), sizes, color=colors)
ax2.set_xlabel('Cluster')
ax2.set_ylabel('Count')
ax2.set_title('Cluster Size Distribution')
ax2.set_xticks(range(n_clusters))
for i, (bar, size) in enumerate(zip(bars, sizes)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             str(size), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('output/dtw_cluster_distribution_full.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: output/dtw_cluster_distribution_full.png")

# 生成完整报告
print("\n生成报告文件...")
report = []
report.append("# 全量 DTW 聚类分析报告")
report.append(f"\n**数据规模**: {results['total_curves']} 条拧紧曲线")
report.append(f"**聚类数**: {n_clusters}")
report.append(f"**分析方法**: DTW + 层次聚类 (Ward)")
report.append(f"**生成时间**: 2026-03-11")

report.append("\n## 聚类分布")
for i in range(n_clusters):
    count = cluster_stats[i]
    report.append(f"- **Cluster {i}**: {count} 条 ({count/len(labels)*100:.1f}%)")

report.append("\n## 与采样结果对比")
report.append("| 聚类 | 采样(50条) | 全量(170条) | 变化 |")
report.append("|------|-----------|------------|------|")
report.append("| Cluster 0 | 8 (16%) | 60 (35.3%) | 显著增加 |")
report.append("| Cluster 1 | 10 (20%) | 20 (11.8%) | 比例下降 |")
report.append("| Cluster 2 | 11 (22%) | 10 (5.9%) | 显著下降 |")
report.append("| Cluster 3 | 2 (4%) | 8 (4.7%) | 基本持平 |")
report.append("| Cluster 4 | 19 (38%) | 72 (42.4%) | 占比最高 |")

report.append("\n## 关键发现")
report.append("1. **Cluster 4** 是主要模式 (42.4%)，可能是标准拧紧流程")
report.append("2. **Cluster 0** 占比第二 (35.3%)，可能是高扭矩模式")
report.append("3. **Cluster 1** 占 11.8%，可能是中等扭矩模式")
report.append("4. **Cluster 2** 仅占 5.9%，特殊模式")
report.append("5. **Cluster 3** 最少 (4.7%)，异常/短曲线模式")

report.append("\n## 可视化文件")
report.append("- `dtw_dendrogram_full.png` - 层次聚类树状图")
report.append("- `dtw_cluster_distribution_full.png` - 聚类分布图")
report.append("- `dtw_full_results.json` - 聚类结果数据")
report.append("- `dtw_cache.pkl` - DTW 距离矩阵缓存 (231KB)")

report_text = '\n'.join(report)
with open('全量DTW聚类报告.md', 'w', encoding='utf-8') as f:
    f.write(report_text)

print("  已保存: 全量DTW聚类报告.md")
print("\n" + "="*60)
print("报告生成完成!")
print("="*60)
