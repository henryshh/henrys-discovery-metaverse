import time
import collections
from typing import Tuple, Optional, Dict, List

class SynergyService:
    """负责将视觉轨迹数据与拧紧结果在时空上进行关联"""
    def __init__(self, buffer_sec: int = 600):
        # 使用 deque 存储轨迹：(timestamp, x, y)
        self.track_buffer = collections.deque(maxlen=buffer_sec * 30) # 假设 30fps，存储 10 分钟
        self.active_session_id: Optional[str] = None

    def push_coordinate(self, x: int, y: int, timestamp: float = None):
        """记录当前时刻的工具坐标"""
        ts = timestamp or time.time()
        self.track_buffer.append((ts, x, y))

    def get_coordinate_at(self, target_ts: float, tolerance_ms: int = 500) -> Optional[Tuple[int, int]]:
        """回溯查找特定时间点附近的工具坐标"""
        if not self.track_buffer:
            return None
        
        # 寻找最接近 target_ts 的记录
        best_match = None
        min_diff = float('inf')
        
        # 由于 track_buffer 是按时间顺序排列的，可以优化查询（这里先用简单线性查找或二分）
        tolerance_sec = tolerance_ms / 1000.0
        
        for ts, x, y in reversed(self.track_buffer):
            diff = abs(ts - target_ts)
            if diff < min_diff:
                min_diff = diff
                best_match = (x, y)
            
            # 如果时间差已经开始增大，且已经超出容差，说明已经过了最佳匹配区间
            if diff > tolerance_sec and ts < target_ts:
                break
                
        if min_diff <= tolerance_sec:
            return best_match
        return None

    def clear_buffer(self):
        self.track_buffer.clear()

    def map_to_slot(self, x: int, y: int, slot_map: Dict[str, Tuple[int, int, int]]) -> Optional[str]:
        """将像素坐标映射到工件槽位 ID。slot_map: {slot_id: (center_x, center_y, radius)}"""
        for slot_id, (sx, sy, r) in slot_map.items():
            dist = ((x - sx)**2 + (y - sy)**2)**0.5
            if dist <= r:
                return slot_id
        return None
