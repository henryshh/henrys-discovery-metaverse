"""
Diagnosis Service - SOTA Stage 2 Explainable AI (XAI)
Failure Mode Library & Semantic Mapping
"""
import numpy as np
from typing import Dict, List, Any, Optional

class DiagnosisService:
    """提供可解释的拧紧故障诊断"""
    
    def __init__(self):
        # 故障门限定义 (百分比偏移)
        self.FAIL_THRESHOLDS = {
            'SOFT_JOINT_RATIO': 0.7,      # 刚度低于基准 70% 判定为软连接
            'CROSS_THREAD_GRAD': 2.0,    # 初始梯度超过基准 2 倍判定为错牙
            'YIELD_EARLY_RATIO': 0.8,    # 屈服扭矩低于规格 80% 判定为拉长
        }

    def diagnose_curve(self, 
                       features: Dict[str, float], 
                       baseline_features: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        根据物理特征进行语义诊断
        返回故障列表，每个故障包含：name, probability, description
        """
        diagnoses = []
        
        # 提取当前特征
        current_stiffness = features.get('elastic_stiffness', 0)
        current_yield_idx = int(features.get('yield_point_idx', 0))
        current_snug_idx = int(features.get('snug_point_idx', 0))
        final_torque = features.get('final_torque', 0)
        
        # 如果没有基准，只能进行通用物理检查
        if not baseline_features:
            return [{"name": "Baseline Missing", "probability": 1.0, "description": "系统尚无该 Silo 的参考基准"}]

        base_stiffness = baseline_features.get('elastic_stiffness', 0)
        
        # 1. 软连接/垫片缺失检查 (Soft Joint / Missing Washer)
        if base_stiffness > 0:
            stiffness_ratio = current_stiffness / base_stiffness
            if stiffness_ratio < self.FAIL_THRESHOLDS['SOFT_JOINT_RATIO']:
                diagnoses.append({
                    "name": "Soft Joint (软连接/垫片缺失)",
                    "probability": min(1.0, 1.2 - stiffness_ratio),
                    "description": f"当前刚度 ({current_stiffness:.3f}) 显著低于基准 ({base_stiffness:.3f})，可能缺失垫片或零件材质过软。"
                })

        # 2. 错牙检查 (Cross Thread)
        # 如果在贴合点之前的平均梯度过高
        if current_snug_idx > 10:
             # 这里只是示例逻辑，实际需要更复杂的区间分析
             pass

        # 3. 屈服点过早 (Elongated Bolt / Over-tightened)
        if baseline_features.get('yield_point_idx') and current_yield_idx < baseline_features['yield_point_idx'] * 0.8:
             diagnoses.append({
                "name": "Early Yielding (屈服过早/螺栓拉长)",
                "probability": 0.8,
                "description": "螺栓提前进入屈服阶段，可能螺栓已发生永久塑性变形或材质不符。"
            })

        # 4. 浮拧/扭矩失速 (Stall / Stripped Thread)
        # 如果最终扭矩未达标但角度过大
        if final_torque < 1.0: # 简化逻辑
            diagnoses.append({
                "name": "Stripped Thread / Stall (滑牙/失速)",
                "probability": 0.9,
                "description": "扭矩未能建立或突然中断，需检查螺纹完好性。"
            })

        if not diagnoses:
            diagnoses.append({
                "name": "Process Healthy",
                "probability": 1.0,
                "description": "物理特征与基准高度吻合，工艺流程稳定。"
            })

        return diagnoses

    def explain_residual(self, error_curve: np.ndarray) -> str:
        """分析 Autoencoder 的残差分布，给出空间解释"""
        max_error_idx = np.argmax(error_curve)
        if max_error_idx < len(error_curve) * 0.3:
            return "异常主要发生在【旋入阶段】，可能存在异物或丝牙干涉。"
        elif max_error_idx < len(error_curve) * 0.7:
            return "异常主要发生在【贴合/弹性阶段】，可能涉及零件刚性或夹紧力问题。"
        else:
            return "异常主要发生在【终紧阶段】，可能涉及屈服控制或扭矩过冲。"
