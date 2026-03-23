"""
Test database operations
"""
import os
import sys
import pytest
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.database import DatabaseManager, reset_db_manager


class TestDatabaseManager:
    """Tests for DatabaseManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.db = DatabaseManager(self.temp_db)
    
    def teardown_method(self):
        """Clean up after tests."""
        self.db.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)
    
    def test_create_project(self):
        """Test creating a project."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO projects (id, name, description) VALUES (?, ?, ?)",
                ("proj-1", "Test Project", "Test Description")
            )
            conn.commit()
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects WHERE id = ?", ("proj-1",))
            row = cursor.fetchone()
        
        assert row is not None
        assert row["name"] == "Test Project"
        assert row["description"] == "Test Description"
    
    def test_create_dataset(self):
        """Test creating a dataset."""
        import json
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO datasets (id, project_id, name, column_names)
                VALUES (?, ?, ?, ?)
                """,
                ("ds-1", "proj-1", "Test Dataset", json.dumps(["col1", "col2"]))
            )
            conn.commit()
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets WHERE id = ?", ("ds-1",))
            row = cursor.fetchone()
        
        assert row is not None
        assert row["name"] == "Test Dataset"
        column_names = json.loads(row["column_names"])
        assert column_names == ["col1", "col2"]
    
    def test_dataset_project_relationship(self):
        """Test dataset-project foreign key relationship."""
        import json
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            # Create projects
            cursor.execute("INSERT INTO projects (id, name) VALUES (?, ?)", ("proj-1", "Project 1"))
            cursor.execute("INSERT INTO projects (id, name) VALUES (?, ?)", ("proj-2", "Project 2"))
            
            # Create datasets
            cursor.execute(
                "INSERT INTO datasets (id, project_id, name, column_names) VALUES (?, ?, ?, ?)",
                ("ds-1", "proj-1", "Dataset 1", json.dumps(["a", "b"]))
            )
            cursor.execute(
                "INSERT INTO datasets (id, project_id, name, column_names) VALUES (?, ?, ?, ?)",
                ("ds-2", "proj-1", "Dataset 2", json.dumps(["c", "d"]))
            )
            cursor.execute(
                "INSERT INTO datasets (id, project_id, name, column_names) VALUES (?, ?, ?, ?)",
                ("ds-3", "proj-2", "Dataset 3", json.dumps(["e", "f"]))
            )
            conn.commit()
        
        # Verify project-1 has 2 datasets
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM datasets WHERE project_id = ?", ("proj-1",))
            row = cursor.fetchone()
            assert row["count"] == 2
        
        # Verify project-2 has 1 dataset
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM datasets WHERE project_id = ?", ("proj-2",))
            row = cursor.fetchone()
            assert row["count"] == 1
    
    def test_columns_exist(self):
        """Test that required columns exist."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(projects)")
            columns = [row[1] for row in cursor.fetchall()]
            assert "id" in columns
            assert "name" in columns
            assert "description" in columns
            assert "created_at" in columns
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(datasets)")
            columns = [row[1] for row in cursor.fetchall()]
            assert "id" in columns
            assert "project_id" in columns
            assert "name" in columns
            assert "file_path" in columns
            assert "column_names" in columns
            assert "created_at" in columns
    
    def test_indexes_exist(self):
        """Test that required indexes exist."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
            indexes = [row[0] for row in cursor.fetchall()]
        
        assert "idx_datasets_project_id" in indexes
        assert "idx_datasets_created_at" in indexes
        assert "idx_projects_created_at" in indexes


class TestDatabaseConnectionPool:
    """Tests for database connection pooling."""
    
    def test_connection_pool_works(self):
        """Test that connections can be acquired from the pool."""
        db = DatabaseManager(tempfile.mktemp(suffix='.db'))
        
        # Acquire and use connections
        with db.get_connection() as conn1:
            with db.get_connection() as conn2:
                pass
        
        # Verify connections were returned to pool and usable
        with db.get_connection() as conn3:
            cursor = conn3.cursor()
            cursor.execute("SELECT 1")
            assert cursor.fetchone()[0] == 1
        
        db.close()
        os.remove(db.db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
