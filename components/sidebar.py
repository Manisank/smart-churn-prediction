import streamlit as st

def render_sidebar():
    """Renders the custom sidebar with branding and logout."""
    st.sidebar.title("💳 SWOMII AI")
    st.sidebar.markdown("---")
    
    # Custom Sidebar Navigation is handled by st.navigation in app.py,
    # but you can add global filters or status indicators here.
    st.sidebar.success("Model Status: Online")
    st.sidebar.info("System Version: 1.0.4-SaaS")
    
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.auth = False
        st.rerun()