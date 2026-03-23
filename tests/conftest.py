"""
Test configuration and fixtures
"""
import os
import sys
import tempfile
import shutil
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.database import DatabaseManager, reset_db_manager
from src.models.project import Project
from src.models.dataset import Dataset, CurveData, DatasetConfig


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    db_path = tempfile.mktemp(suffix='.db')
    db = DatabaseManager(db_path)
    yield db
    db.close()
    if os.path.exists(db_path):
        os.remove(db_path)


def cleanup_temp_files():
    """清理临时文件"""
    import glob
    for temp_path in glob.glob(tempfile.gettempdir() + "/tmp*"):
        try:
            if os.path.isdir(temp_path):
                shutil.rmtree(temp_path, ignore_errors=True)
            else:
                os.remove(temp_path)
        except (OSError, PermissionError):
            # 忽略权限错误
            pass


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            # 清理时忽略权限错误
            pass


@pytest.fixture
def clean_workspace():
    """Create a clean workspace for testing."""
    import tempfile
    import os
    temp_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    yield temp_dir
    os.chdir(original_cwd)
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except PermissionError:
        # 清理时忽略权限错误
        pass


@pytest.fixture(autouse=True)
def cleanup():
    yield
    # 清理全局数据库管理器
    reset_db_manager()
    # 清理时忽略权限错误
    try:
        cleanup_temp_files()
    except PermissionError:
        pass


@pytest.fixture
def sample_project():
    """Create a sample project for testing."""
    return Project(
        id="test-project-123",
        name="Test Project",
        description="A test project"
    )


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return Dataset(
        id="test-dataset-456",
        project_id="test-project-123",
        name="Test Dataset",
        column_names=["id", "name", "value"]
    )


@pytest.fixture
def test_data_dir():
    """Get the test data directory."""
    return os.path.join(os.path.dirname(__file__), 'data')


def pytest_configure():
    """Pytest configuration."""
    pytest.test_start_time = None
