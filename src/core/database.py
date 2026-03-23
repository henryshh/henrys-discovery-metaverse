"""
Database Manager - SQLite with connection pooling and indexes
"""
import os
import sqlite3
import queue
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any


class DatabaseManager:
    """SQLite database manager with connection pool and indexes."""
    
    def __init__(self, db_path: str = "data/projects.db", pool_size: int = 10, data_dir: str = None):
        # 如果提供了data_dir，则使用默认数据库路径在data_dir下
        if data_dir is not None:
            db_path = os.path.join(data_dir, "projects.db")
        self.db_path = db_path
        self.data_dir = Path(db_path).parent  # 使用Path对象以支持旧版API
        # 确保数据目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._pool = queue.Queue(maxsize=pool_size)
        self._init_database()
    
    def _init_database(self):
        """Initialize database with tables and indexes."""
        conn = sqlite3.connect(self.db_path)
        # 启用 WAL 模式以解决并发问题 - 必须在创建表之前设置
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        cursor = conn.cursor()
        
        # Create projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                meta_json TEXT,  -- Flexible metadata for vision, metaverse, etc.
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create datasets table with JSON column for column_names
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                name TEXT NOT NULL,
                file_path TEXT,
                column_names TEXT,  -- JSON序列化存储
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        """)
        
        # ✅ 关键：创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_datasets_project_id ON datasets(project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_projects_created_at ON projects(created_at)")
        
        # ✅ CRITICAL: Migration - Ensure meta_json column exists in main DB
        try:
            cursor.execute("ALTER TABLE projects ADD COLUMN meta_json TEXT")
        except sqlite3.OperationalError:
            pass # Column already exists
            
        conn.commit()
        conn.close()
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        try:
            conn = self._pool.get_nowait()
        except queue.Empty:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            # 启用 WAL 模式以解决并发问题
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            try:
                self._pool.put_nowait(conn)
            except queue.Full:
                conn.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a query and return results as list of dicts."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an update/insert and return affected row count."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount
    
    def get_project_db(self, project_id: str):
        """Get a database connection for a specific project."""
        # 在项目特定位置创建数据库
        project_db_path = os.path.join(self.data_dir, project_id + ".db")
        # 确保目录存在
        Path(project_db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(project_db_path, check_same_thread=False)
        # 启用 WAL 模式以解决并发问题
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        
        # 初始化表结构
        cursor = conn.cursor()
        
        # Create projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                meta_json TEXT,  -- Flexible metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create datasets table with JSON column for column_names
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                name TEXT NOT NULL,
                file_path TEXT,
                column_names TEXT,  -- JSON序列化存储
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        """)
        
        # Create curves table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS curves (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                name TEXT NOT NULL,
                data TEXT,  -- JSON serialized
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id),
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        """)
        
        # Create models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                name TEXT NOT NULL,
                config TEXT,  -- JSON serialized
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        """)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_datasets_project_id ON datasets(project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_projects_created_at ON projects(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_curves_dataset_id ON curves(dataset_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_curves_composite ON curves(dataset_id, created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_project_id ON models(project_id)")
        
        # ✅ Migration - Ensure meta_json exists in project-specific DB
        try:
            cursor.execute("ALTER TABLE projects ADD COLUMN meta_json TEXT")
        except sqlite3.OperationalError:
            pass
            
        conn.commit()
        return conn

    def save_project(self, project_id: str, name: str, description: str = None):
        """Save a project to the database."""
        # 使用项目特定的数据库
        conn = self.get_project_db(project_id)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO projects (id, name, description) VALUES (?, ?, ?)",
            (project_id, name, description)
        )
        conn.commit()
        conn.close()

    def close(self):
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except queue.Empty:
                break


        # Global instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager(db_path: str = "data/projects.db") -> DatabaseManager:
    """Get or create the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(db_path)
    return _db_manager


def reset_db_manager():
    """Reset the global database manager (for testing)."""
    global _db_manager
    if _db_manager is not None:
        _db_manager.close()
        _db_manager = None
