import json
import numpy as np

# 加载聚类结果
with open('output/dtw_full_results.json', 'r') as f:
    results = json.load(f)

print('='*60)
print('DTW 聚类结果分析')
print('='*60)
print(f"总曲线数: {results['total_curves']}")
print(f"聚类数: {results['n_clusters']}")
print()

labels = np.array(results['labels'])

print('聚类分布:')
for i in range(results['n_clusters']):
    count = np.sum(labels == i)
    percentage = count / len(labels) * 100
    print(f'  Cluster {i}: {count:3d} 条 ({percentage:5.1f}%)')

print()
print('='*60)
print('注意: 当前聚类描述是基于DTW算法结果的假设')
print('实际特征需要进一步分析每条聚类的曲线形状')
print('='*60)
