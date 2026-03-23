"""
Web Utilities for Streamlit integration with backend services.
"""
import streamlit as st
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in path
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.database import get_db_manager, DatabaseManager
from src.services.project_service import get_project_service, ProjectService
from src.services.dataset_service import get_dataset_service, DatasetService
from src.services.synergy_service import SynergyService

@st.cache_resource
def get_services():
    """Initialize and cache backend services."""
    db_manager = get_db_manager()
    synergy_service = SynergyService()
    project_service = get_project_service(db_manager)
    dataset_service = get_dataset_service(db_manager)
    
    # Inject synergy service into dataset service
    dataset_service.synergy = synergy_service
    
    return {
        "db": db_manager,
        "project": project_service,
        "dataset": dataset_service,
        "synergy": synergy_service
    }

def init_app_state():
    """Initialize session state variables if they don't exist."""
    # Note: We now prioritize DB over session_state for core data
    if 'current_project_id' not in st.session_state:
        st.session_state.current_project_id = None
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "dashboard"
    if 'clustering_results' not in st.session_state:
        st.session_state.clustering_results = {}

def get_current_project():
    """Get the currently selected project object from DB."""
    if not st.session_state.current_project_id:
        return None
    services = get_services()
    return services["project"].get_project(st.session_state.current_project_id)
