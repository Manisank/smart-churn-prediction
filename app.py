# import streamlit as st
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
import sys
import os



# # Page configuration
# st.set_page_config(
#     page_title="Smart Churn Analytics",
#     page_icon="🦉",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Load custom CSS
# def load_css():
#     css_file = "assets/style.css"
#     if os.path.exists(css_file):
#         with open(css_file) as f:
#             st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# load_css()

# # Sidebar Navigation
# st.sidebar.title("🦉 Smart Churn Prediction")
# st.sidebar.markdown("### Customer Churn Intelligence Suite")
# st.sidebar.markdown("---")

# page = st.sidebar.radio(
#     "Navigate to:",
#     ["🏠 Home", "🔮 Churn Prediction", "📊 Historical Analytics"],
#     index=0
# )

# st.sidebar.markdown("---")
# st.sidebar.info(
#     "**Powered by:**\n"
#     "- XGBoost ML Model\n"
#     "- SHAP Explainability\n"
#     "- Gemini AI Strategy"
# )

# # Home Page
# if page == "🏠 Home":
#     st.title("🦉 Welcome to SWOMII AI")
#     st.markdown("### *Smart Wise Owl for Monitoring Insights & Intelligence*")
    
#     st.markdown("---")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#         ## 🎯 What We Do
        
#         SWOMII AI is your **intelligent companion** for predicting and preventing customer churn in E-commerce.
        
#         ### Key Features:
#         - **🔮 Real-time Predictions**: Score customers instantly with XGBoost ML
#         - **🧠 Explainable AI**: Understand *why* customers might leave with SHAP
#         - **🦉 Smart Strategies**: Get AI-powered retention tactics from Gemini
#         - **📊 Deep Analytics**: Explore historical patterns and trends
#         """)
    
#     with col2:
#         st.markdown("""
#         ## 🚀 Quick Start Guide
        
#         ### For Predictions:
#         1. Click **"🔮 Churn Prediction"** in the sidebar
#         2. Enter customer details
#         3. Get instant risk scores + AI strategies
        
#         ### For Analysis:
#         1. Click **"📊 Historical Analytics"**
#         2. Filter by category, city, or tenure
#         3. Discover hidden churn patterns
        
#         ---
        
#         ### 💡 Pro Tip
#         Use SHAP Force Plots to see exactly which factors drive each prediction!
#         """)
    
#     st.markdown("---")
    
#     # Metrics Overview
#     st.markdown("## 📈 Why Churn Prediction Matters")
#     metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
#     with metric_col1:
#         st.metric("Customer Acquisition Cost", "$200-500", delta=None)
    
#     with metric_col2:
#         st.metric("Retention Cost", "$50-100", delta="-75%", delta_color="normal")
    
#     with metric_col3:
#         st.metric("Avg. Churn Impact", "15-25%", delta="Revenue Loss")
    
#     with metric_col4:
#         st.metric("Early Detection ROI", "5-10x", delta="+500%", delta_color="normal")
    
#     st.markdown("---")
#     st.success("✅ **Ready to start?** Choose a page from the sidebar!")

# # Churn Prediction Page
# elif page == "🔮 Churn Prediction":
#     from pages.prediction import show_prediction_page
#     show_prediction_page()

# # Historical Analytics Page
# elif page == "📊 Historical Analytics":
#     from pages.historical import show_historical_page
#     show_historical_page()

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SWOMII AI - Churn Analytics",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_file = "assets/style.css"
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Sidebar Navigation with Clean Design
st.sidebar.markdown("""
<div style='text-align: center; padding: 1.5rem 0 2rem 0;'>
    <h1 style='color: #ffffff; font-size: 2.2rem; margin: 0; font-weight: 700;'>
        🦉 SWOMII AI
    </h1>
    <p style='color: #94a3b8; font-size: 0.85rem; margin: 0.5rem 0 0 0; letter-spacing: 0.5px;'>
        Churn Intelligence Platform
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔮 Churn Prediction", "📊 Analytics"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Tech Stack - Compact Version
st.sidebar.markdown("""
<div style='background: rgba(99, 102, 241, 0.1); 
            padding: 1rem; 
            border-radius: 10px;
            border-left: 3px solid #6366f1;'>
    <p style='color: #cbd5e1; font-size: 0.7rem; margin: 0; text-transform: uppercase; letter-spacing: 1px; font-weight: 700;'>
        POWERED BY
    </p>
    <p style='color: #e2e8f0; font-size: 0.85rem; margin: 0.5rem 0 0 0; line-height: 1.8;'>
        ✨ XGBoost ML<br>
        🧠 SHAP AI<br>
        🤖 Gemini 2.0
    </p>
</div>
""", unsafe_allow_html=True)

# Footer
st.sidebar.markdown("""
<div style='position: absolute; bottom: 1rem; left: 0; right: 0; text-align: center;'>
    <p style='color: #64748b; font-size: 0.7rem; margin: 0;'>
        © 2026 SWOMII AI<br>
        <span style='color: #475569;'>Version 1.0</span>
    </p>
</div>
""", unsafe_allow_html=True)

# Home Page
if page == "🏠 Home":
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0 1rem 0;'>
        <h1 style='font-size: 3rem; color: #1e293b; margin: 0; font-weight: 700;'>
            🦉 Welcome to SWOMII AI
        </h1>
        <p style='font-size: 1.2rem; color: #64748b; margin: 1rem 0 0 0;'>
            Smart Wise Owl for Monitoring Insights & Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 🎯 What We Do
        
        SWOMII AI helps **e-commerce businesses predict and prevent customer churn** using advanced machine learning and AI.
        
        ### Key Features:
        - **🔮 Real-time Predictions** - Score customers instantly with XGBoost ML
        - **🧠 Explainable AI** - Understand *why* customers might leave with SHAP
        - **🤖 Smart Strategies** - Get AI-powered retention tactics from Gemini
        - **📊 Deep Analytics** - Explore historical patterns and trends
        """)
    
    with col2:
        st.markdown("""
        ## 🚀 Quick Start Guide
        
        ### Step 1: Predict Churn
        Click **"🔮 Churn Prediction"** in the sidebar
        
        ### Step 2: Enter Customer Data
        Fill in customer attributes (tenure, satisfaction, etc.)
        
        ### Step 3: Get Insights
        - Churn probability score
        - SHAP visualizations explaining the prediction
        - AI-generated retention strategies
        
        ### Step 4: Analyze Trends
        Explore **"📊 Analytics"** for historical churn patterns
        
        ---
        
        ### 💡 Pro Tip
        SHAP plots show exactly which factors drive each prediction!
        """)
    
    st.markdown("---")
    
    # Metrics Overview
    st.markdown("## 📈 Why Churn Prevention Matters")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric(
            "Acquisition Cost", 
            "$200-500",
            help="Average cost to acquire a new customer"
        )
    
    with metric_col2:
        st.metric(
            "Retention Cost", 
            "$50-100", 
            delta="-75%", 
            delta_color="normal",
            help="Cost to retain an existing customer (75% cheaper)"
        )
    
    with metric_col3:
        st.metric(
            "Avg. Churn Impact", 
            "15-25%", 
            delta="Revenue Loss",
            delta_color="inverse",
            help="Typical revenue lost to customer churn"
        )
    
    with metric_col4:
        st.metric(
            "Early Detection ROI", 
            "5-10x", 
            delta="+500%", 
            delta_color="normal",
            help="Return on investment with AI-powered retention"
        )
    
    st.markdown("---")
    st.success("✅ **Ready to start?** Select a page from the sidebar to begin!")

# Churn Prediction Page
elif page == "🔮 Churn Prediction":
    from pages.prediction import show_prediction_page
    show_prediction_page()

# Analytics Page
elif page == "📊 Analytics":
    from pages.historical import show_historical_page
    show_historical_page()
