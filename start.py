#!/usr/bin/env python3
"""
Tightening AI - Main Entry Point
"""
import os
import sys

def main():
    """Main entry point."""
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    print("Henry's Discovery Metaverse - Starting...")
    
    # Import and initialize core components
    from src.core.database import DatabaseManager, get_db_manager
    
    # Create database
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'projects.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    db = DatabaseManager(db_path)
    print(f"Database initialized: {db_path}")
    
    # Check for FAISS availability
    try:
        from src.core.vector_store import VectorStore
        vs = VectorStore()
        print(f"Vector Store: {'Available' if vs.is_available() else 'Not installed (optional)'}")
    except Exception as e:
        print(f"Vector Store: Not available - {e}")
    
    # Check for diskcache availability
    try:
        from src.core.cache import CacheManager, get_cache_manager
        cache = CacheManager()
        print(f"Cache: {'Available' if cache.is_available() else 'Not installed (optional)'}")
    except Exception as e:
        print(f"Cache: Not available - {e}")
    
    print("\nHenry's Discovery Metaverse is ready!")
    print("\nAvailable modules:")
    print("  - src.core.database : DatabaseManager")
    print("  - src.core.storage : StorageManager")
    print("  - src.core.vector_store : VectorStore")
    print("  - src.core.cache : CacheManager")
    print("  - src.services.project_service : ProjectService")
    print("  - src.services.dataset_service : DatasetService")
    print("  - src.models.project : Project model")
    print("  - src.models.dataset : Dataset model")
    
    return db

if __name__ == "__main__":
    db = main()
    
    # Keep running for interactive use
    try:
        import code
        code.InteractiveConsole(locals=globals()).interact(
            banner="\nTightening AI Interactive Console\nType 'exit()' to quit\n",
            exitmsg="\nGoodbye!"
        )
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        db.close()
