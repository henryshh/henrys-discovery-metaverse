"""
Dataset Service - Business logic for datasets with JSON column_names
"""
import uuid
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

from src.core.database import DatabaseManager
from src.models.dataset import Dataset, DatasetRepository, CurveData
from .ai_baseline_service import AIBaselineService
import numpy as np
from scipy.interpolate import interp1d


class DatasetService:
    """Service layer for dataset operations."""
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self.ai_baseline = AIBaselineService()
        self.train_threshold = 50 # 每 50 条 OK 数据重训一次
    
    def create_dataset(
        self,
        project_id: str,
        name: str,
        column_names: List[str],
        file_path: Optional[str] = None
    ) -> str:
        """
        Create a new dataset.
        
        Args:
            project_id: Parent project ID
            name: Dataset name
            column_names: List of column names (will be JSON serialized)
            file_path: Optional file path reference
        
        Returns:
            Dataset ID
        """
        dataset_id = str(uuid.uuid4())
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            # ✅ 关键：JSON序列化
            cursor.execute(
                """
                INSERT INTO datasets (id, project_id, name, file_path, column_names)
                VALUES (?, ?, ?, ?, ?)
                """,
                (dataset_id, project_id, name, file_path, json.dumps(column_names))
            )
            conn.commit()
        
        return dataset_id
    
    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a dataset by ID.
        
        Returns:
            Dataset dictionary with JSON-deserialized column_names
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
            row = cursor.fetchone()
            
            if row:
                return {
                    "id": row["id"],
                    "project_id": row["project_id"],
                    "name": row["name"],
                    "file_path": row["file_path"],
                    # ✅ 关键：JSON反序列化
                    "column_names": json.loads(row["column_names"]) if row["column_names"] else [],
                    "created_at": row["created_at"]
                }
        return None
    
    def get_dataset_with_model(self, dataset_id: str) -> Optional[Dataset]:
        """
        Get a dataset as a Dataset model.
        
        Returns:
            Dataset model with deserialized column_names
        """
        data = self.get_dataset(dataset_id)
        if data:
            return Dataset(
                id=data["id"],
                project_id=data["project_id"],
                name=data["name"],
                file_path=data.get("file_path"),
                column_names=data["column_names"],
                created_at=data["created_at"]
            )
        return None
    
    def list_datasets(self, project_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List datasets for a project."""
        datasets = []
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM datasets 
                WHERE project_id = ? 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
                """,
                (project_id, limit, offset)
            )
            rows = cursor.fetchall()
            
            for row in rows:
                datasets.append({
                    "id": row["id"],
                    "project_id": row["project_id"],
                    "name": row["name"],
                    "file_path": row["file_path"],
                    # ✅ 关键：JSON反序列化
                    "column_names": json.loads(row["column_names"]) if row["column_names"] else [],
                    "created_at": row["created_at"]
                })
        
        return datasets
    
    def update_dataset(
        self,
        dataset_id: str,
        name: Optional[str] = None,
        column_names: Optional[List[str]] = None,
        file_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Update a dataset."""
        updates = []
        params = []
        
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        
        if file_path is not None:
            updates.append("file_path = ?")
            params.append(file_path)
        
        if column_names is not None:
            updates.append("column_names = ?")
            # ✅ 关键：JSON序列化
            params.append(json.dumps(column_names))
        
        if not updates:
            return self.get_dataset(dataset_id)
        
        params.append(dataset_id)
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE datasets SET {', '.join(updates)} WHERE id = ?",
                params
            )
            conn.commit()
        
        return self.get_dataset(dataset_id)
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            conn.commit()
            return cursor.rowcount > 0

    def import_dataset_from_file(self, project_id: str, file_path: str, name: str = None):
        """Import dataset from file for compatibility."""
        if name is None:
            import os
            name = os.path.basename(file_path)
        # 假设列名可以从文件推断或使用默认值
        return self.create_dataset(project_id, name, ["id", "value"], file_path)
    
    def import_from_json(self, project_id: str, json_file: str, name_prefix: str = None):
        """流式导入 JSON 档案，自动按 Tool -> Pset 层级进行语义分区，并执行一致性检查"""
        from src.data_loader import TighteningDataLoader
        import hashlib
        
        if name_prefix is None:
            name_prefix = os.path.basename(json_file)
            
        loader = TighteningDataLoader(json_file)
        
        # 跟踪此导入会话中的统计数据
        # {silo_id: {"added": 0, "anomalies": 0, "details": []}}
        sync_report = {}
        
        for record in loader.load_stream():
            results = loader.process_record(record, target_length=None)
            
            for curve_array, label, meta in results:
                tool_serial = meta.get('toolSerialNumber', 'UnknownTool')
                pset_num = meta.get('pSetNumber', 0)
                config = meta.get('config', {})
                
                # 1. 语义一致性检查 (Config Hash)
                config_str = json.dumps(config, sort_keys=True)
                config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
                silo_key = (tool_serial, pset_num, config_hash)
                
                dataset_id = self._get_or_create_silo_dataset(
                    project_id, tool_serial, pset_num, config_hash, config, name_prefix
                )
                
                if dataset_id not in sync_report:
                    sync_report[dataset_id] = {"name": f"{tool_serial}_Pset{pset_num}", "added": 0, "anomalies": 0, "geotagged": 0}
                
                # 1.5 空间坐标关联 (Stage 3 Synergy)
                ts = meta.get('timestamp')
                coord = self.synergy.get_coordinate_at(ts) if hasattr(self, 'synergy') else None
                if coord:
                    meta['coord_x'], meta['coord_y'] = coord
                    sync_report[dataset_id]["geotagged"] += 1
                
                # 2. 统计一致性检查 (Data Drift Detection)
                # 检查此曲线相对于该 Silo 已有数据的偏离度
                is_anomaly = self._check_consistency(project_id, dataset_id, meta)
                
                # 3. AI 形状一致性校验 (Strategy A)
                ai_score = self.ai_baseline.calculate_anomaly_score(dataset_id, curve_array[:, 0])
                meta['shapeAnomalyScore'] = ai_score
                
                if is_anomaly or ai_score > 0.5: # 设定一个初步阈值
                    sync_report[dataset_id]["anomalies"] += 1
                
                # 4. 稳健入库
                if not is_anomaly:
                    self._insert_curve_to_project_db(project_id, dataset_id, curve_array, meta)
                    sync_report[dataset_id]["added"] += 1
                else:
                    sync_report[dataset_id]["anomalies"] += 1
                    # 记录并跳过严重长度异常，或者仅记录
                    if meta.get('length_mismatch'):
                         print(f"⚠️ Length mismatch for {meta.get('resultNumber')}: Expected {meta.get('expected_len')}, got {meta.get('pointCount')}")
                
                # 5. 触发增量重训 (训练基准)
                if sync_report[dataset_id]["added"] % self.train_threshold == 0:
                    self._trigger_refinement(project_id, dataset_id)
                
        return sync_report

    def _trigger_refinement(self, project_id, dataset_id):
        """触发该 Silo 的形状基准进化"""
        project_db = self.db.get_project_db(project_id)
        cursor = project_db.cursor()
        
        # 仅使用标注为 OK 的最新曲线进行微调
        cursor.execute(
            "SELECT data FROM curves WHERE dataset_id=? AND json_extract(data, '$.report')='OK' ORDER BY id DESC LIMIT 200",
            (dataset_id,)
        )
        ok_curves = []
        for row in cursor.fetchall():
            c_data = json.loads(row[0])
            ok_curves.append(np.array(c_data["torque"]))
        
        project_db.close()
        
        if len(ok_curves) >= 10:
            print(f"🚀 Refining AI Baseline for Silo {dataset_id}...")
            self.ai_baseline.train_baseline(dataset_id, ok_curves, epochs=30)

    def _check_consistency(self, project_id, dataset_id, meta):
        """检查新增数据是否与 Silo 基准一致 (长度 & 3-Sigma 准则)"""
        new_val = meta.get('finalTorque')
        new_len = meta.get('pointCount') or 0 # 假设 meta 中有长度信息，或者从数组获取
        
        project_db = self.db.get_project_db(project_id)
        cursor = project_db.cursor()
        
        # 1. 检查长度一致性 (硬件决定，同组必须相等)
        cursor.execute(
            "SELECT json_extract(data, '$.pointCount') FROM curves WHERE dataset_id=? LIMIT 1", 
            (dataset_id,)
        )
        row = cursor.fetchone()
        if row and row[0] is not None:
            expected_len = row[0]
            if new_len != expected_len:
                meta['length_mismatch'] = True
                meta['expected_len'] = expected_len
                project_db.close()
                return True # 长度不匹配是严重的硬件/配置异常
        
        # 2. 统计数值一致性 (3-Sigma)
        cursor.execute(
            "SELECT json_extract(data, '$.finalTorque') FROM curves WHERE dataset_id=?", 
            (dataset_id,)
        )
        existing_vals = [row[0] for row in cursor.fetchall() if row[0] is not None]
        project_db.close()
        
        if len(existing_vals) < 10:
            return False
            
        mean = sum(existing_vals) / len(existing_vals)
        std = (sum((v - mean)**2 for v in existing_vals) / len(existing_vals))**0.5
        
        if std == 0: return False
        
        z_score = abs(new_val - mean) / std
        return z_score > 3.0

    def _get_or_create_silo_dataset(self, project_id, tool_serial, pset_num, config_hash, config, prefix):
        """获取或创建符合语义层级的数据集容器"""
        name = f"{prefix}_{tool_serial}_Pset{pset_num}_{config_hash}"
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            # 检查是否存在
            cursor.execute(
                "SELECT id FROM datasets WHERE project_id=? AND name=?", 
                (project_id, name)
            )
            row = cursor.fetchone()
            if row:
                return row[0]
            
            # 创建新数据集 (存储 Pset 和 Tool 信息在 column_names 字段作为扩展元数据)
            dataset_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO datasets (id, project_id, name, column_names)
                VALUES (?, ?, ?, ?)
                """,
                (dataset_id, project_id, name, json.dumps({
                    "tool": tool_serial,
                    "pset": pset_num,
                    "config_hash": config_hash,
                    "config": config
                }))
            )
            conn.commit()
            return dataset_id

    def _insert_curve_to_project_db(self, project_id, dataset_id, curve_array, meta):
        """将单条曲线插入项目特定数据库"""
        project_db = self.db.get_project_db(project_id)
        cursor = project_db.cursor()
        
        # 构造存储结构 (4-channel support)
        curve_data = {
            "resultNumber": meta.get('resultNumber'),
            "vin": meta.get('vin'),
            "report": meta.get('report'),
            "toolSerialNumber": meta.get('toolSerialNumber'),
            "pSetNumber": meta.get('pSetNumber'),
            "pointDuration": meta.get('pointDuration'),
            "pointIndices": meta.get('pointIndices'),
            "torque": curve_array[:, 0].tolist(),
            "angle": curve_array[:, 1].tolist() if curve_array.shape[1] > 1 else [],
            "current": curve_array[:, 2].tolist() if curve_array.shape[1] > 2 else [],
            "marker": curve_array[:, 3].tolist() if curve_array.shape[1] > 3 else []
        }
        
        curve_id = str(meta.get('resultNumber')) if meta.get('resultNumber') else str(uuid.uuid4())
        
        cursor.execute(
            """
            INSERT OR REPLACE INTO curves (id, dataset_id, project_id, name, data)
            VALUES (?, ?, ?, ?, ?)
            """,
            (curve_id, dataset_id, project_id, f"Curve_{curve_id}", json.dumps(curve_data))
        )
        project_db.commit()
        project_db.close()

    def get_aligned_curves(self, project_id: str, dataset_id: str, n_points: int = 200, limit: int = 100) -> Dict[str, Any]:
        """
        获取对齐至标准角度坐标系的曲线集 (Torque-Angle Coordination)
        """
        project_db = self.db.get_project_db(project_id)
        cursor = project_db.cursor()
        
        cursor.execute(
            "SELECT data FROM curves WHERE dataset_id=? LIMIT ?", 
            (dataset_id, limit)
        )
        rows = cursor.fetchall()
        
        all_curves = []
        all_metadata = []
        
        for row in rows:
            c_data = json.loads(row[0])
            torque = np.array(c_data.get("torque", []))
            angle = np.array(c_data.get("angle", []))
            
            if len(torque) > 0 and len(angle) > 0:
                # 角度归一化/对齐逻辑：
                # 1. 寻找该数据集的最大角度范围（或者使用 0 到 max(angle)）
                # 为了简单起见，我们暂且插值到 [0, max(angle)] 的 n_points 个点
                # 实际工业标准通常是对齐到固定角度增量
                x_new = np.linspace(min(angle), max(angle), n_points)
                f = interp1d(angle, torque, kind='linear', fill_value="extrapolate")
                torque_aligned = f(x_new)
                
                all_curves.append(torque_aligned)
                all_metadata.append(c_data)
        
        project_db.close()
        
        return {
            "curves": np.array(all_curves),
            "angle_grid": x_new if 'x_new' in locals() else [],
            "metadata": all_metadata
        }

    def get_dataset_statistics(self, dataset_id: str):
        """Get dataset statistics for compatibility."""
        dataset = self.get_dataset(dataset_id)
        if dataset:
            return {
                "record_count": 0,  # 实际应用中应从数据源获取
                "column_count": len(dataset["column_names"]),
                "name": dataset["name"]
            }
        return None


# Global instance
_dataset_service: Optional[DatasetService] = None
_dataset_repo: Optional[DatasetRepository] = None


def get_dataset_service(db: Optional[DatabaseManager] = None) -> DatasetService:
    """Get or create the global dataset service instance."""
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService(db)
    return _dataset_service


def clear_dataset_service():
    """Clear dataset service for testing."""
    global _dataset_service
    _dataset_service = None


def get_dataset_repo() -> Optional[DatasetRepository]:
    """Get the global dataset repository."""
    global _dataset_repo
    return _dataset_repo


def clear_dataset_repo():
    """Clear dataset repository for testing."""
    global _dataset_repo
    _dataset_repo = None
