"""
拧紧曲线数据加载模块
Data Loader for Tightening Curve Analysis
"""

import json
import time
from datetime import datetime
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d


class TighteningDataLoader:
    """拧紧曲线数据加载器"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.data = None
        self.curves = []
        self.labels = []
        self.metadata = []
        
    def load(self) -> 'TighteningDataLoader':
        """加载整个 JSON 文件"""
        self.data = list(self.load_stream())
        return self

    def load_stream(self, chunk_size=65536):
        """流式加载 JSON 数据，支持极大数据文件 (Generator 模式)"""
        print(f"Streaming data from {self.data_path}...")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            # 首先判断是 V2 (Array) 还是 V1 (Concatenated)
            first_char = f.read(1)
            while first_char and first_char in ' \t\n\r':
                first_char = f.read(1)
            
            f.seek(0)
            if first_char == '[':
                # V2: JSON Array
                yield from self._stream_v2_array(f, chunk_size)
            else:
                # V1: Concatenated Objects
                yield from self._stream_v1_concatenated(f, chunk_size)

    def _stream_v1_concatenated(self, f, chunk_size):
        """流式解析拼接的 JSON 对象"""
        decoder = json.JSONDecoder()
        buffer = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            
            idx = 0
            while idx < len(buffer):
                # 跳过空白
                while idx < len(buffer) and buffer[idx] in ' \t\n\r':
                    idx += 1
                if idx >= len(buffer):
                    break
                
                try:
                    obj, end = decoder.raw_decode(buffer, idx)
                    if isinstance(obj, dict) and 'model' in obj:
                        yield obj
                    idx = end
                except json.JSONDecodeError:
                    # 如果 buffer 里的数据不够一个完整对象，跳出内循环读取更多 chunk
                    break
            
            buffer = buffer[idx:]

    def _stream_v2_array(self, f, chunk_size):
        """流式解析 JSON 数组 (简单实现，仅针对一级数组内的对象)"""
        # 注意：这里为了性能和复杂度平衡，采用分段识别模式或者使用 ijson (如果环境允许)
        # 此处我们通过 decoder 处理数组内的每一项
        decoder = json.JSONDecoder()
        buffer = ""
        
        # 跳过开头的 '['
        char = f.read(1)
        while char and char != '[':
            char = f.read(1)
            
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            
            idx = 0
            while idx < len(buffer):
                # 跳过空白, 逗号, 和数组结束符
                while idx < len(buffer) and buffer[idx] in ' \t\n\r,[]':
                    idx += 1
                if idx >= len(buffer):
                    break
                
                try:
                    obj, end = decoder.raw_decode(buffer, idx)
                    if isinstance(obj, dict):
                        yield obj
                    idx = end
                except json.JSONDecodeError:
                    break
            
            buffer = buffer[idx:]
    
    def process_record(self, record: Dict, target_length: Optional[int] = 500) -> List[Tuple[np.ndarray, int, Dict]]:
        """处理单个记录，提取曲线、标签和元数据"""
        extracted = []
        model = record.get('model', {})
        results = model.get('results', [])
        
        if not results:
            return []
            
        # 1. 基础元数据
        report = model.get('report', 'OK')
        label = 1 if report == 'OK' else 0
        result_number = model.get('resultNumber')
        vin = model.get('vin')
        
        # 提取并解析时间戳 (ISO 或 Unix)
        raw_date = model.get('executionDate') or model.get('executionTime')
        timestamp = None
        if raw_date:
            try:
                if isinstance(raw_date, (int, float)):
                    timestamp = float(raw_date)
                else:
                    dt = datetime.fromisoformat(raw_date.replace('Z', '+00:00'))
                    timestamp = dt.timestamp()
            except:
                pass
        
        # 2. 语义层级元数据 (Tool -> Pset)
        pset_info = model.get('pSet', {})
        pset_number = pset_info.get('number')
        
        for result in results:
            tool_info = result.get('tool', {})
            tool_serial = tool_info.get('serialNumber')
            
            curves = result.get('curves', [])
            steps = result.get('steps', [{}])
            step_data = steps[0].get('data', {}) if steps else {}
            step_config = steps[0].get('configuration', {}) if steps else {}
            
            for curve in curves:
                point_duration = curve.get('pointDuration', 0)
                curve_points = curve.get('data', {}).get('points', [])
                
                if not curve_points:
                    continue
                
                # 3. 提取 4 通道数据 (同时计算时间轴)
                torque_seq = [p.get('torque', 0) for p in curve_points]
                angle_seq = [p.get('angle', 0) for p in curve_points]
                current_seq = [p.get('current', 0) for p in curve_points]
                marker_seq = [p.get('marker', 0) for p in curve_points]
                point_indices = [p.get('pointIndex', i) for i, p in enumerate(curve_points)]
                
                # 4. 归一化 (如果需要)
                if target_length is not None:
                    torque_seq = self._normalize_length(torque_seq, target_length)
                    angle_seq = self._normalize_length(angle_seq, target_length)
                    current_seq = self._normalize_length(current_seq, target_length)
                    marker_seq = self._normalize_length(marker_seq, target_length)
                else:
                    torque_seq = np.array(torque_seq, dtype=np.float32)
                    angle_seq = np.array(angle_seq, dtype=np.float32)
                    current_seq = np.array(current_seq, dtype=np.float32)
                    marker_seq = np.array(marker_seq, dtype=np.float32)
                
                # 5. SOTA 数据硬化：中值滤波去噪
                # 对扭矩和电流进行 5 点中值滤波，消除电磁干扰和采样脉冲
                torque_seq = median_filter(torque_seq, size=5)
                current_seq = median_filter(current_seq, size=5)
                
                curve_array = np.stack([torque_seq, angle_seq, current_seq, marker_seq], axis=-1)
                
                # 5. 组合元数据
                meta = {
                    'resultNumber': result_number,
                    'vin': vin,
                    'report': report,
                    'toolSerialNumber': tool_serial,
                    'pSetNumber': pset_number,
                    'pointDuration': point_duration,
                    'pointIndices': point_indices,
                    'pointCount': len(torque_seq),
                    'config': step_config,
                    'finalTorque': step_data.get('finalTorque') if step_data.get('finalTorque') is not None else float(torque_seq[-1]),
                    'finalAngle': step_data.get('finalAngle') if step_data.get('finalAngle') is not None else float(angle_seq[-1]),
                    'timestamp': timestamp or time.time() # 兜底使用当前时间
                }
                
                extracted.append((curve_array, label, meta))
        
        return extracted
    
    def extract_curves(self, target_length: Optional[int] = 500) -> 'TighteningDataLoader':
        """(Legacy) 批量提取曲线，已迁移至 process_record"""
        self.curves = []
        self.labels = []
        self.metadata = []
        
        for record in self.data:
            results = self.process_record(record, target_length)
            for c, l, m in results:
                self.curves.append(c)
                self.labels.append(l)
                self.metadata.append(m)
        return self
    
    def _normalize_length(self, seq: List[float], target_length: int) -> np.ndarray:
        """标准化序列长度"""
        seq = np.array(seq, dtype=np.float32)
        if len(seq) >= target_length:
            return seq[:target_length]
        else:
            # 填充到目标长度
            padded = np.zeros(target_length, dtype=np.float32)
            padded[:len(seq)] = seq
            return padded
    
    def get_data(self) -> Tuple[Any, np.ndarray, List[Dict]]:
        """获取数据集 (如果长度一致返回 ndarray，否则返回 list)"""
        try:
            return np.array(self.curves), np.array(self.labels), self.metadata
        except ValueError:
            # 长度不一致时返回 list
            return self.curves, np.array(self.labels), self.metadata
    
    def get_curves_by_label(self, label: int) -> List[np.ndarray]:
        """按标签获取曲线"""
        return [c for c, l in zip(self.curves, self.labels) if l == label]


    def align_curves_by_angle(self, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        将所有提取的曲线对齐至统一的角度网格 (Torque-Angle Coordination)
        """
        if not self.curves:
            return np.array([]), np.array([]), []
        
        aligned_curves = []
        
        # 1. 确定统一的角度范围 (使用所有曲线角度的最大公约范围或简单取 0 到 max)
        # 工业界通常采用固定步长，这里我们插值到 200 点
        max_angles = [c[-1, 1] for c in self.curves]
        common_max_angle = np.mean(max_angles) # 或者取 max，或者取固定值
        angle_grid = np.linspace(0, common_max_angle, n_points)
        
        for curve in self.curves:
            torque = curve[:, 0]
            angle = curve[:, 1]
            
            # 使用 linear 插值将 torque 对映到 angle_grid
            f = interp1d(angle, torque, kind='linear', fill_value="extrapolate")
            torque_aligned = f(angle_grid)
            aligned_curves.append(torque_aligned)
            
        return np.array(aligned_curves), angle_grid, self.metadata


if __name__ == "__main__":
    # 测试数据加载
    loader = load_example_data("../API/example data.json")
    curves, labels, metadata = loader.get_data()
    print(f"\nData shape: {curves.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample curve shape: {curves[0].shape}")
    print(f"Channels: torque, angle, current, marker")
