"""
Project Service - Business logic for projects
"""
import uuid
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from src.core.database import DatabaseManager
from src.models.project import Project


class ProjectService:
    """Service layer for project operations."""
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
    
    def create_project(self, name: str, description: Optional[str] = None, project_id: str = None) -> Project:
        """Create a new project."""
        if project_id is None:
            project_id = str(uuid.uuid4())
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO projects (id, name, description, meta_json) VALUES (?, ?, ?, ?)",
                (project_id, name, description, "{}")
            )
            conn.commit()
        
        return Project(
            id=project_id,
            name=name,
            description=description,
            metadata={},
            created_at=datetime.now()
        )
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
                row = cursor.fetchone()
                
                if row:
                    meta = json.loads(row["meta_json"]) if "meta_json" in row and row["meta_json"] else {}
                    return Project(
                        id=row["id"],
                        name=row["name"],
                        description=row["description"],
                        metadata=meta,
                        created_at=row["created_at"] if "created_at" in row else datetime.now()
                    )
        except Exception:
            pass
        return None
    
    def list_projects(self, limit: int = 100, offset: int = 0) -> List[Project]:
        """List projects with pagination."""
        projects = []
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM projects ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
            rows = cursor.fetchall()
            
            for row in rows:
                meta = json.loads(row["meta_json"]) if "meta_json" in row and row["meta_json"] else {}
                projects.append(Project(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    metadata=meta,
                    created_at=row["created_at"],
                    datasets=[]
                ))
        
        return projects
    
    def update_project(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Optional[Project]:
        """Update a project."""
        updates = []
        params = []
        
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        
        if not updates:
            return self.get_project(project_id)
        
        params.append(project_id)
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE projects SET {', '.join(updates)} WHERE id = ?",
                params
            )
            conn.commit()
        
        return self.get_project(project_id)

    def update_project_metadata(self, project_id: str, category: str, data: Dict[str, Any]) -> Optional[Project]:
        """Update a specific category of project metadata (e.g. 'vision', 'metaverse')."""
        project = self.get_project(project_id)
        if not project:
            return None
        
        # Merge new metadata
        new_meta = project.metadata.copy()
        new_meta[category] = data
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE projects SET meta_json = ? WHERE id = ?",
                (json.dumps(new_meta), project_id)
            )
            conn.commit()
            
            # Record in project-specific DB too for redundancy/local access
            try:
                with self.db.get_project_db(project_id) as p_conn:
                    p_cursor = p_conn.cursor()
                    p_cursor.execute(
                        "UPDATE projects SET meta_json = ? WHERE id = ?",
                        (json.dumps(new_meta), project_id)
                    )
                    p_conn.commit()
                    p_conn.close()
            except Exception:
                pass # Main DB is the truth
                
        return self.get_project(project_id)
    
    def delete_project(self, project_id: str) -> bool:
        """Delete a project (cascades to datasets)."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            conn.commit()
            return cursor.rowcount > 0


# Global instance
_project_service: Optional[ProjectService] = None


def get_project_service(db: Optional[DatabaseManager] = None) -> ProjectService:
    """Get or create the global project service instance."""
    global _project_service
    if _project_service is None:
        _project_service = ProjectService(db)
    return _project_service


def clear_project_service():
    """Clear project service for testing."""
    global _project_service
    _project_service = None
