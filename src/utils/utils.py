"""
Utility functions for the application
"""
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional


def validate_json_file(file_path: str) -> bool:
    """Validate if a file is valid JSON."""
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, FileNotFoundError):
        return False


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal."""
    # Remove any path components
    filename = filename.replace("/", "").replace("\\", "")
    # Remove any null bytes
    filename = filename.replace("\0", "")
    return filename


def ensure_directory(path: str):
    """Ensure a directory exists, create if not."""
    Path(path).mkdir(parents=True, exist_ok=True)


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    import time
    import random
    timestamp = str(int(time.time() * 1000))[-10:]
    random_part = str(random.randint(10000, 99999))
    return f"{prefix}{timestamp}{random_part}" if prefix else f"{timestamp}{random_part}"


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def merge_dicts(base: Dict, updates: Dict) -> Dict:
    """Recursively merge two dictionaries."""
    result = base.copy()
    for key, value in updates.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split a list into chunks."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_file_summary(file_path: str) -> Optional[Dict[str, Any]]:
    """Get a summary of a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        summary = {
            "type": type(data).__name__,
            "keys": list(data.keys()) if isinstance(data, dict) else len(data) if isinstance(data, list) else None
        }
        
        if isinstance(data, dict):
            summary["sample_keys"] = list(data.keys())[:5]
        elif isinstance(data, list) and len(data) > 0:
            summary["sample_item_type"] = type(data[0]).__name__
        
        return summary
    except Exception:
        return None
