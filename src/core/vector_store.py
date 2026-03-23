"""
Vector Store - FAISS with conditional import
"""
import os
from typing import Optional, List, Dict, Any


class VectorStore:
    """Vector store using FAISS with conditional import."""
    
    def __init__(self, index_path: str = "data/vectors", project_id: str = None, dimension: int = None, data_dir: str = None):
        # 如果提供了project_id，则使用项目特定的路径
        if project_id is not None:
            index_path = os.path.join(index_path, project_id)
        # 如果提供了data_dir，则基于data_dir构建路径
        if data_dir is not None and project_id is not None:
            index_path = os.path.join(data_dir, "vectors", project_id)
        self.index_path = index_path
        self.project_id = project_id
        self.dimension = dimension
        self._faiss = None
        self._index = None
        self._id_map: Dict[int, str] = {}
        
        # Conditionally import FAISS
        try:
            import faiss
            self._faiss = faiss
            os.makedirs(index_path, exist_ok=True)
        except ImportError:
            print("Warning: faiss-cpu not installed. Vector search disabled.")
            print("Install with: pip install faiss-cpu")
    
    def is_available(self) -> bool:
        """Check if FAISS is available."""
        return self._faiss is not None
    
    def create_index(self, dimension: int, index_type: str = "IndexFlatIP"):
        """Create a new FAISS index."""
        if not self._faiss:
            raise RuntimeError("FAISS is not installed")
        
        if index_type == "IndexFlatIP":
            self._index = self._faiss.index_inner_product(dimension)
        elif index_type == "IndexFlatL2":
            self._index = self._faiss.index_l2(dimension)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self._id_map = {}
    
    def add_vectors(self, vectors: List[List[float]], ids: List[str]):
        """Add vectors to the index."""
        if not self._index:
            raise RuntimeError("Index not created")
        
        import numpy as np
        np_vectors = np.array(vectors, dtype=np.float32)
        
        # Get next index for IDs
        next_id = self._index.ntotal
        faiss_ids = np.array([next_id + i for i in range(len(vectors))], dtype=np.int64)
        
        self._index.add(np_vectors)
        
        # Map FAISS IDs to string IDs
        for i, vector_id in enumerate(ids):
            self._id_map[next_id + i] = vector_id
    
    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if not self._index:
            raise RuntimeError("Index not created")
        
        import numpy as np
        np_query = np.array([query_vector], dtype=np.float32)
        
        distances, indices = self._index.search(np_query, k)
        
        results = []
        for i in range(len(indices[0])):
            faiss_id = indices[0][i]
            if faiss_id >= 0:
                vector_id = self._id_map.get(faiss_id, str(faiss_id))
                results.append({
                    "id": vector_id,
                    "distance": float(distances[0][i])
                })
        
        return results
    
    def save(self, filename: str = "index.faiss"):
        """Save the index to disk."""
        if not self._index:
            raise RuntimeError("Index not created")
        
        import numpy as np
        faiss_path = os.path.join(self.index_path, filename)
        try:
            self._faiss.write_index(self._index, faiss_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save index: {str(e)}")
        
        # Save ID map
        import json
        map_path = os.path.join(self.index_path, "id_map.json")
        try:
            with open(map_path, 'w') as f:
                # Ensure the dict keys are strings for JSON serialization
                serializable_id_map = {str(k): v for k, v in self._id_map.items()}
                json.dump(serializable_id_map, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save ID map: {str(e)}")
    
    def load(self, filename: str = "index.faiss"):
        """Load the index from disk."""
        if not self._faiss:
            raise RuntimeError("FAISS is not installed")
        
        import numpy as np
        faiss_path = os.path.join(self.index_path, filename)
        if os.path.exists(faiss_path):
            try:
                self._index = self._faiss.read_index(faiss_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load index: {str(e)}")
            
            # Load ID map
            import json
            map_path = os.path.join(self.index_path, "id_map.json")
            if os.path.exists(map_path):
                try:
                    with open(map_path, 'r') as f:
                        self._id_map = json.load(f)
                        # Convert string keys back to integers if needed
                        self._id_map = {int(k): v for k, v in self._id_map.items()}
                except Exception as e:
                    raise RuntimeError(f"Failed to load ID map: {str(e)}")
