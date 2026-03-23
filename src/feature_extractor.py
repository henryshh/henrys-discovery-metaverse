"""
拧紧曲线特征提取模块
Feature Extractor for Tightening Curve Analysis
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
from scipy.signal import find_peaks


class TighteningFeatureExtractor:
    """
    拧紧曲线特征提取器
    
    提取的特征包括：
    1. 统计特征（均值、方差、偏度、峰度等）
    2. 时域特征（最大值、最小值、斜率等）
    3. 曲线形态特征（贴合点、屈服点等）
    4. 频域特征（FFT 变换）
    """
    
    def __init__(self):
        self.feature_names = []
        
    def extract_all_features(self, curves: np.ndarray) -> np.ndarray:
        """
        提取所有特征
        
        Args:
            curves: 曲线数据 [n_samples, n_points, n_channels]
                    channels: [torque, angle, current, marker]
        
        Returns:
            features: 特征矩阵 [n_samples, n_features]
        """
        all_features = []
        
        for i, curve in enumerate(curves):
            features = self.extract_curve_features(curve)
            all_features.append(features)
            
        return np.array(all_features)
    
    def extract_curve_features(self, curve: np.ndarray) -> List[float]:
        """
        提取单条曲线的特征 (含 SOTA 动力学特征)
        """
        torque = curve[:, 0]
        angle = curve[:, 1]
        current = curve[:, 2]
        marker = curve[:, 3]
        
        features = []
        self.feature_names = []
        
        # 0. 基础梯度计算 (dT/da)
        # 避免除以零，并对角度进行平滑处理
        da = np.diff(angle)
        dt = np.diff(torque)
        # 过滤过小的角度增量以减少噪声
        valid_diff = da > 0.001
        gradients = np.zeros_like(da)
        gradients[valid_diff] = dt[valid_diff] / da[valid_diff]
        # 使用滑动平均平滑梯度以识别趋势
        smooth_gradients = np.convolve(gradients, np.ones(5)/5, mode='same')
        
        # ========== 1. 统计特征 ==========
        features.extend([
            np.mean(torque),
            np.std(torque),
            np.max(torque),
            stats.skew(torque),
        ])
        self.feature_names.extend(['torque_mean', 'torque_std', 'torque_max', 'torque_skew'])
        
        # ========== 2. 动力学专家特征 (SOTA) ==========
        # 贴合点 (Snug Point): 梯度持续超过阈值的起始点
        snug_idx = self._detect_snug_point_advanced(torque, angle, smooth_gradients)
        features.append(float(snug_idx))
        self.feature_names.append('snug_point_idx')
        
        # 屈服点 (Yield Point): 进入线性段后，梯度下降 > 20% 的位置
        yield_idx, elastic_stiffness = self._detect_yield_point(torque, angle, smooth_gradients, snug_idx)
        features.extend([float(yield_idx), float(elastic_stiffness)])
        self.feature_names.extend(['yield_point_idx', 'elastic_stiffness'])
        
        # 弹性区间角度范围 (Elastic Reach)
        elastic_reach = angle[yield_idx] - angle[snug_idx] if yield_idx > snug_idx > 0 else 0
        features.append(float(elastic_reach))
        self.feature_names.append('elastic_reach_deg')

        # ========== 3. 最终状态 ==========
        non_zero_idx = np.where(torque > 0.1)[0]
        final_torque = torque[non_zero_idx[-1]] if len(non_zero_idx) > 0 else 0
        final_angle = angle[non_zero_idx[-1]] if len(non_zero_idx) > 0 else 0
        features.extend([float(final_torque), float(final_angle)])
        self.feature_names.extend(['final_torque', 'final_angle'])
        
        # ========== 4. 电流与辅助特征 ==========
        features.extend([
            np.mean(current),
            np.max(current),
        ])
        self.feature_names.extend(['current_mean', 'current_max'])
        
        return features

    def _detect_snug_point_advanced(self, torque, angle, gradients, threshold_nm_deg=0.5) -> int:
        """
        SOTA 贴合点检测：
        识别扭矩-角度梯度持续达到稳定增长的起始位置（克服间隙期）
        """
        # 寻找梯度连续 N 个点超过阈值的地方
        window = 10
        for i in range(len(gradients) - window):
            if np.all(gradients[i:i+window] > threshold_nm_deg) and torque[i] > 0.5:
                return i
        return 0

    def _detect_yield_point(self, torque, angle, gradients, snug_idx) -> Tuple[int, float]:
        """
        SOTA 屈服点检测 (梯度下降法)：
        1. 在贴合点之后寻找线性段（最大梯度点）
        2. 识别梯度从峰值下降（通常设为 20%）的点，判定为屈服起始
        """
        if snug_idx >= len(gradients) - 20:
            return len(torque) - 1, 0.0
            
        # 寻找贴合后的最大梯度（弹性阶段）
        elastic_zone_gradients = gradients[snug_idx:]
        max_grad = np.max(elastic_zone_gradients)
        max_grad_idx = np.argmax(elastic_zone_gradients) + snug_idx
        
        # 从最大梯度点向后找，梯度下降到 80% 的位置
        yield_threshold = max_grad * 0.8
        for i in range(max_grad_idx, len(gradients)):
            if gradients[i] < yield_threshold:
                return i, float(max_grad)
                
        return len(torque) - 1, float(max_grad)

    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return self.feature_names

if __name__ == "__main__":
    extractor = TighteningFeatureExtractor()
    print("SOTA Feature Extractor Loaded.")
