#!/usr/bin/env python3
"""Test data loader with both V1 and V2 formats"""
import sys
sys.path.insert(0, 'src')

from src.data_loader import TighteningDataLoader

# 测试 V2 格式 (Anord02.json)
print('=== Testing V2 format (Anord02.json) ===')
loader_v2 = TighteningDataLoader('API/Anord02.json')
loader_v2.load()
print(f'First record keys: {list(loader_v2.data[0].keys())}')
print(f'First resultNumber: {loader_v2.data[0]["model"]["resultNumber"]}')
print()

# 测试 V1 格式 (Anord.json)
print('=== Testing V1 format (Anord.json) ===')
loader_v1 = TighteningDataLoader('API/Anord.json')
loader_v1.load()
print(f'First record keys: {list(loader_v1.data[0].keys())}')
print(f'First resultNumber: {loader_v1.data[0]["model"]["resultNumber"]}')
print()

# 测试曲线提取
print('=== Testing curve extraction (V2) ===')
loader_v2.extract_curves(target_length=500)
curves, labels, metadata = loader_v2.get_data()
print(f'Curves shape: {curves.shape}')
print(f'Labels shape: {labels.shape}')
print(f'Sample metadata: {metadata[0]}')
