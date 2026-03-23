# Henry's Discovery Metaverse

A Python-based data management and analysis framework with SQLite database, file storage, and optional ML capabilities.

## Features

- **SQLite Database** - With connection pooling and indexes
- **File Storage** - Local file storage with 100MB limit
- **Vector Store** - Optional FAISS integration for similarity search
- **Caching** - Optional diskcache integration
- **Project Management** - CRUD operations for projects
- **Dataset Management** - JSON-serialized column metadata

## Quick Start

### Installation

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Running

```bash
# Start the application
python start.py

# Or run tests
pytest tests/ -v
```

## Project Structure

```
tightening-ai/
├── start.py              # Main entry point
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── src/
│   ├── __init__.py
│   ├── core/            # Core services
│   │   ├── __init__.py
│   │   ├── database.py  # SQLite + pooling + indexes
│   │   ├── vector_store.py  # FAISS (optional)
│   │   ├── storage.py       # Local file storage (100MB limit)
│   │   └── cache.py         # diskcache (optional)
│   ├── models/          # Data models
│   │   ├── __init__.py
│   │   ├── project.py
│   │   ├── dataset.py     # JSON-serialized column_names
│   │   └── model.py
│   ├── services/        # Business logic
│   │   ├── __init__.py
│   │   ├── project_service.py
│   │   └── dataset_service.py
│   └── utils/           # Utilities
│       └── __init__.py
├── tests/               # Test suite
│   ├── __init__.py
│   ├── test_database.py
│   ├── test_storage.py
│   ├── test_services.py
│   └── conftest.py
└── data/                # Data directory
    ├── projects/        # Project files
    ├── vectors/         # Vector indices
    └── uploads/         # User uploads
```

## Core Components

### Database

```python
from src.core.database import DatabaseManager

db = DatabaseManager()
with db.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM projects")
    rows = cursor.fetchall()
```

### Storage

```python
from src.core.storage import StorageManager

storage = StorageManager()
# Saves file with size validation (max 100MB)
file_path = storage.save_file("project-id", "file.txt", "output.txt")
```

### Dataset Service

```python
from src.services.dataset_service import DatasetService

service = DatasetService()
dataset_id = service.create_dataset(
    project_id="proj-123",
    name="My Dataset",
    column_names=["id", "name", "value"]
)
```

## Testing

Run the full test suite:

```bash
pytest tests/ -v
```

Run specific test files:

```bash
pytest tests/test_database.py -v
pytest tests/test_storage.py -v
pytest tests/test_services.py -v
```

## Requirements

### Core (Required)
- Python 3.8+
- SQLite (built-in)

### Optional
- FAISS (`faiss-cpu`) - For vector search
- diskcache (`diskcache`) - For caching
- pytest - For testing

Install optional dependencies:

```bash
pip install faiss-cpu diskcache pytest
```

## License

MIT License
