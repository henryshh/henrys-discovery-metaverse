import streamlit as st
from src.web.web_utils import get_services, get_current_project

def render_project_tab():
    services = get_services()
    project_service = services["project"]
    
    st.markdown("#### 📁 Project Management")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        with st.container(border=True):
            st.markdown("##### ➕ Create New Project")
            name = st.text_input("Project Name")
            desc = st.text_area("Description")
            if st.button("Create Project", type="primary", use_container_width=True):
                if name:
                    project_service.create_project(name, desc)
                    st.success(f"✅ Project '{name}' created successfully!")
                    st.rerun()
    
    with col2:
        with st.container(border=True):
            st.markdown("##### 📋 Active Projects")
            projects = project_service.list_projects()
            if not projects:
                st.info("No projects created yet.")
            for proj in projects:
                with st.container(border=True):
                    cols = st.columns([3, 1])
                    with cols[0]:
                        st.subheader(proj.name)
                        st.caption(f"Created: {str(proj.created_at)[:10]} | {proj.description or ''}")
                    with cols[1]:
                        if st.session_state.current_project_id == proj.id:
                            st.button("Active", disabled=True, key=f"sel_{proj.id}")
                        else:
                            if st.button("Select", key=f"sel_{proj.id}", type="primary"):
                                st.session_state.current_project_id = proj.id
                                st.rerun()
