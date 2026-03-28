# import streamlit as st
# import os
# import sys
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Force the app root into the python path to resolve module issues
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)

# # Page configuration
# st.set_page_config(
#     page_title="Customer Churn Intelligence",
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

# # Sidebar Navigation with Clean Design
# st.sidebar.markdown("""
# <div style='text-align: center; padding: 1.5rem 0 2rem 0;'>
#     <h1 style='color: #ffffff; font-size: 1.8rem; margin: 0; font-weight: 700;'>
#         🦉 Smart Churn Prediction
#     </h1>
#     <p style='color: #94a3b8; font-size: 0.85rem; margin: 0.5rem 0 0 0; letter-spacing: 0.5px;'>
#         Customer Intelligence Platform
#     </p>
# </div>
# """, unsafe_allow_html=True)

# st.sidebar.markdown("---")

# # Navigation
# page = st.sidebar.radio(
#     "Navigate",
#     ["🏠 Home", "🔮 Churn Prediction", "📊 Analytics"],
#     label_visibility="collapsed"
# )

# st.sidebar.markdown("---")

# # Tech Stack - Compact Version
# st.sidebar.markdown("""
# <div style='background: rgba(99, 102, 241, 0.1); 
#             padding: 1rem; 
#             border-radius: 10px;
#             border-left: 3px solid #6366f1;'>
#     <p style='color: #cbd5e1; font-size: 0.7rem; margin: 0; text-transform: uppercase; letter-spacing: 1px; font-weight: 700;'>
#         POWERED BY
#     </p>
#     <p style='color: #e2e8f0; font-size: 0.85rem; margin: 0.5rem 0 0 0; line-height: 1.8;'>
#         ✨ XGBoost ML<br>
#         🧠 SHAP AI<br>
#         🤖 Gemini 2.0
#     </p>
# </div>
# """, unsafe_allow_html=True)

# # Footer
# st.sidebar.markdown("""
# <div style='position: absolute; bottom: 1rem; left: 0; right: 0; text-align: center;'>
#     <p style='color: #64748b; font-size: 0.7rem; margin: 0;'>
#         © 2026 Customer Churn Intelligence<br>
#         <span style='color: #475569;'>Version 1.0</span>
#     </p>
# </div>
# """, unsafe_allow_html=True)

# # Home Page
# if page == "🏠 Home":
#     st.markdown("""
#     <div style='text-align: center; padding: 2rem 0 1rem 0;'>
#         <h1 style='font-size: 3rem; color: #1e293b; margin: 0; font-weight: 700;'>
#             🦉 Welcome to Customer Churn Intelligence
#         </h1>
#         <p style='font-size: 1.2rem; color: #64748b; margin: 1rem 0 0 0;'>
#             Advanced Monitoring Insights & Predictive Intelligence
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("---")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#         ## 🎯 What We Do
        
#         Our platform helps **e-commerce businesses predict and prevent customer churn** using advanced machine learning and AI.
        
#         ### Key Features:
#         - **🔮 Real-time Predictions** - Score customers instantly with XGBoost ML
#         - **🧠 Explainable AI** - Understand *why* customers might leave with SHAP
#         - **🤖 Smart Strategies** - Get AI-powered retention tactics from Gemini
#         - **📊 Deep Analytics** - Explore historical patterns and trends
#         """)
    
#     with col2:
#         st.markdown("""
#         ## 🚀 Quick Start Guide
        
#         ### Step 1: Predict Churn
#         Click **"🔮 Churn Prediction"** in the sidebar
        
#         ### Step 2: Enter Customer Data
#         Fill in customer attributes (tenure, satisfaction, etc.)
        
#         ### Step 3: Get Insights
#         - Churn probability score
#         - SHAP visualizations explaining the prediction
#         - AI-generated retention strategies
        
#         ### Step 4: Analyze Trends
#         Explore **"📊 Analytics"** for historical churn patterns
        
#         ---
        
#         ### 💡 Pro Tip
#         SHAP plots show exactly which factors drive each prediction!
#         """)
    
#     st.markdown("---")
    
#     # Metrics Overview
#     st.markdown("## 📈 Why Churn Prevention Matters")
#     metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
#     with metric_col1:
#         st.metric(
#             "Acquisition Cost", 
#             "$200-500",
#             help="Average cost to acquire a new customer"
#         )
    
#     with metric_col2:
#         st.metric(
#             "Retention Cost", 
#             "$50-100", 
#             delta="-75%", 
#             delta_color="normal",
#             help="Cost to retain an existing customer (75% cheaper)"
#         )
    
#     with metric_col3:
#         st.metric(
#             "Avg. Churn Impact", 
#             "15-25%", 
#             delta="Revenue Loss",
#             delta_color="inverse",
#             help="Typical revenue lost to customer churn"
#         )
    
#     with metric_col4:
#         st.metric(
#             "Early Detection ROI", 
#             "5-10x", 
#             delta="+500%", 
#             delta_color="normal",
#             help="Return on investment with AI-powered retention"
#         )
    
#     st.markdown("---")
#     st.success("✅ **Ready to start?** Select a page from the sidebar to begin!")

# # Churn Prediction Page
# elif page == "🔮 Churn Prediction":
#     from pages.prediction import show_prediction_page
#     show_prediction_page()

# # Analytics Page
# elif page == "📊 Analytics":
#     from pages.historical import show_historical_page
#     show_historical_page()
# import streamlit as st
# import os
# import sys
# import json
# import matplotlib.pyplot as plt
# from dotenv import load_dotenv

# load_dotenv()

# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)

# # Page configuration
# st.set_page_config(
#     page_title="Customer Churn Intelligence",
#     page_icon="🦉",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ✅ FIXED: added encoding='utf-8'
# def load_css():
#     css_file = "assets/style.css"
#     if os.path.exists(css_file):
#         with open(css_file, encoding='utf-8') as f:
#             st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# load_css()

# # Dark theme for matplotlib / SHAP charts
# # In app.py — replace your existing plt.rcParams block with this
# plt.rcParams.update({
#     # Backgrounds
#     'figure.facecolor':     '#0a1628',
#     'axes.facecolor':       '#0a1628',
#     'savefig.facecolor':    '#0a1628',
#     'figure.edgecolor':     '#0a1628',
#     'axes.edgecolor':       '#1e3a5f',

#     # Text & labels — THIS was missing, causing white/invisible labels
#     'text.color':           '#e2e8f0',
#     'axes.labelcolor':      '#94a3b8',
#     'xtick.color':          '#94a3b8',
#     'ytick.color':          '#e2e8f0',   # feature names on y-axis
#     'xtick.labelsize':      10,
#     'ytick.labelsize':      10,

#     # Grid
#     'grid.color':           '#1e3a5f',
#     'grid.alpha':           0.5,

#     # Title
#     'axes.titlecolor':      '#e2e8f0',
#     'axes.titlesize':       13,
#     'axes.titleweight':     'bold',

#     # Legend
#     'legend.facecolor':     '#0d1f3c',
#     'legend.edgecolor':     '#1e3a5f',
#     'legend.labelcolor':    '#e2e8f0',
#     'legend.fontsize':      9,

#     # Bar / plot colors
#     'axes.prop_cycle': plt.cycler(color=[
#         '#6366f1', '#10b981', '#f59e0b',
#         '#ef4444', '#0ea5e9', '#8b5cf6'
#     ]),

#     # Font
#     'font.family':          'sans-serif',
#     'font.size':            10,
# })


# # ─── AUTH HELPERS ────────────────────────────────────────────────
# def load_users():
#     try:
#         with open("users.json", "r", encoding='utf-8') as f:
#             return json.load(f)
#     except Exception:
#         return {"admin": "admin123"}

# def save_users(users):
#     with open("users.json", "w", encoding='utf-8') as f:
#         json.dump(users, f, indent=2)

# def check_login(username, password):
#     users = load_users()
#     return users.get(username) == password

# def register_user(username, password):
#     users = load_users()
#     if username in users:
#         return False, "❌ Username already exists. Try a different one."
#     if len(password) < 6:
#         return False, "⚠️ Password must be at least 6 characters."
#     users[username] = password
#     save_users(users)
#     return True, "✅ Account created successfully!"

# def logout():
#     st.session_state.logged_in = False
#     st.session_state.username  = ""
#     st.rerun()


# # ─── LOGIN / SIGNUP PAGE ─────────────────────────────────────────
# def show_auth_page():
#     # Hide sidebar on auth screen
#     st.markdown("""
#     <style>
#         [data-testid="stSidebar"]        { display: none !important; }
#         [data-testid="collapsedControl"] { display: none !important; }

#         .auth-card {
#             max-width: 460px;
#             margin: 4rem auto 0 auto;
#             background: linear-gradient(135deg, #0a1628, #0d1f3c);
#             border: 1px solid #1e3a5f;
#             border-radius: 20px;
#             padding: 2.8rem 2.5rem 2rem 2.5rem;
#             box-shadow: 0 20px 60px rgba(0,0,0,0.6),
#                         inset 0 1px 0 rgba(99,102,241,0.1);
#         }
#         .auth-icon  { text-align:center; font-size:3.2rem; line-height:1; margin-bottom:0.4rem; }
#         .auth-title {
#             text-align:center; color:#e2e8f0;
#             font-size:1.75rem; font-weight:800;
#             margin:0; letter-spacing:-0.5px;
#         }
#         .auth-sub {
#             text-align:center; color:#334155;
#             font-size:0.75rem; margin:0.3rem 0 0 0;
#             text-transform:uppercase; letter-spacing:1.2px;
#         }
#         .auth-hint {
#             text-align:center; color:#334155;
#             font-size:0.75rem; margin-top:1.2rem; line-height:1.9;
#         }
#         .auth-hint b { color:#475569; }

#         /* Style the tab bar inside auth card */
#         .stTabs [data-baseweb="tab-list"] {
#             margin-top: 1.5rem !important;
#         }
#     </style>
#     """, unsafe_allow_html=True)

#     col1, col2, col3 = st.columns([1, 1.6, 1])
#     with col2:
#         # Branding header
#         st.markdown("""
#         <div class='auth-card'>
#             <div class='auth-icon'>🦉</div>
#             <h1 class='auth-title'>Smart Churn</h1>
#             <p class='auth-sub'>Customer Intelligence Platform</p>
#         </div>
#         """, unsafe_allow_html=True)

#         st.markdown("<br>", unsafe_allow_html=True)

#         # ── Tabs: Sign In / Sign Up ──────────────────────────
#         tab_login, tab_signup = st.tabs(["🔐 Sign In", "✨ Sign Up"])

#         # ── SIGN IN ──────────────────────────────────────────
#         with tab_login:
#             with st.form("login_form"):
#                 username = st.text_input(
#                     "Username", placeholder="Enter your username",
#                     key="login_user"
#                 )
#                 password = st.text_input(
#                     "Password", type="password",
#                     placeholder="Enter your password",
#                     key="login_pass"
#                 )
#                 st.markdown("<br>", unsafe_allow_html=True)
#                 submitted = st.form_submit_button(
#                     "🔐 Sign In", use_container_width=True
#                 )

#             if submitted:
#                 if not username or not password:
#                     st.warning("⚠️ Please enter both username and password.")
#                 elif check_login(username, password):
#                     st.session_state.logged_in = True
#                     st.session_state.username  = username
#                     st.rerun()
#                 else:
#                     st.error("❌ Invalid username or password. Please try again.")

#             st.markdown("""
#             <p class='auth-hint'>
#                 Demo credentials<br>
#                 <b>Username:</b> demo &nbsp;|&nbsp; <b>Password:</b> demo123
#             </p>
#             """, unsafe_allow_html=True)

#         # ── SIGN UP ──────────────────────────────────────────
#         with tab_signup:
#             with st.form("signup_form"):
#                 new_username = st.text_input(
#                     "Choose Username", placeholder="Pick a unique username",
#                     key="signup_user"
#                 )
#                 new_password = st.text_input(
#                     "Choose Password", type="password",
#                     placeholder="Min. 6 characters",
#                     key="signup_pass"
#                 )
#                 confirm_password = st.text_input(
#                     "Confirm Password", type="password",
#                     placeholder="Repeat your password",
#                     key="signup_confirm"
#                 )
#                 st.markdown("<br>", unsafe_allow_html=True)
#                 signup_submitted = st.form_submit_button(
#                     "✨ Create Account", use_container_width=True
#                 )

#             if signup_submitted:
#                 if not new_username or not new_password or not confirm_password:
#                     st.warning("⚠️ Please fill in all fields.")
#                 elif len(new_username) < 3:
#                     st.warning("⚠️ Username must be at least 3 characters.")
#                 elif new_password != confirm_password:
#                     st.error("❌ Passwords do not match.")
#                 else:
#                     success, msg = register_user(new_username, new_password)
#                     if success:
#                         st.success(f"✅ Account created! Please **Sign In** as **{new_username}**")
#                         st.balloons()
#                     else:
#                         st.error(msg)

#             st.markdown("""
#             <p class='auth-hint'>
#                 Already have an account? Switch to <b>Sign In</b> tab above.
#             </p>
#             """, unsafe_allow_html=True)


# # ─── SESSION STATE INIT ──────────────────────────────────────────
# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False
# if "username" not in st.session_state:
#     st.session_state.username  = ""

# if not st.session_state.logged_in:
#     show_auth_page()
#     st.stop()


# # ─── SIDEBAR ─────────────────────────────────────────────────────
# st.sidebar.markdown(f"""
# <div style='text-align:center; padding:2rem 1rem 1.5rem 1rem;'>
#     <div style='font-size:3rem; line-height:1;'>🦉</div>
#     <h2 style='color:#e2e8f0; font-size:1.25rem; font-weight:800;
#                margin:0.6rem 0 0.2rem 0; letter-spacing:-0.3px;'>
#         Smart Churn
#     </h2>
#     <p style='color:#334155; font-size:0.72rem; margin:0;
#               text-transform:uppercase; letter-spacing:1.2px;'>
#         Intelligence Platform
#     </p>
# </div>
# """, unsafe_allow_html=True)

# st.sidebar.markdown("---")

# # Logged-in user badge
# st.sidebar.markdown(f"""
# <div style='background:rgba(99,102,241,0.06); border:1px solid #1e3a5f;
#             border-radius:10px; padding:0.7rem 1rem; margin-bottom:1rem;'>
#     <div style='display:flex; align-items:center; gap:0.7rem;'>
#         <div style='background:linear-gradient(135deg,#6366f1,#4f46e5);
#                     border-radius:50%; width:32px; height:32px; flex-shrink:0;
#                     display:flex; align-items:center; justify-content:center;
#                     font-size:0.9rem;'>👤</div>
#         <div>
#             <p style='color:#475569; font-size:0.62rem; margin:0;
#                       text-transform:uppercase; letter-spacing:0.8px;'>SIGNED IN AS</p>
#             <p style='color:#e2e8f0; font-size:0.88rem; margin:0; font-weight:600;'>
#                 {st.session_state.username}
#             </p>
#         </div>
#     </div>
# </div>
# """, unsafe_allow_html=True)

# # Navigation
# page = st.sidebar.radio(
#     "Navigate",
#     ["🏠 Home", "🔮 Churn Prediction", "📊 Analytics"],
#     label_visibility="collapsed"
# )

# st.sidebar.markdown("---")

# # Tech stack badge
# st.sidebar.markdown("""
# <div style='background:rgba(99,102,241,0.06); border:1px solid #1e3a5f;
#             border-left:3px solid #6366f1; border-radius:0 10px 10px 0;
#             padding:0.9rem 1rem; margin-bottom:0.8rem;'>
#     <p style='color:#334155; font-size:0.62rem; margin:0 0 0.6rem 0;
#               text-transform:uppercase; letter-spacing:1.2px; font-weight:700;'>
#         POWERED BY
#     </p>
#     <div style='display:flex; flex-direction:column; gap:0.45rem;'>
#         <span style='color:#94a3b8; font-size:0.8rem;'>🌲 &nbsp;Random Forest ML</span>
#         <span style='color:#94a3b8; font-size:0.8rem;'>🧠 &nbsp;SHAP Explainability</span>
#         <span style='color:#94a3b8; font-size:0.8rem;'>🤖 &nbsp;Gemini 2.0 Flash</span>
#     </div>
# </div>
# """, unsafe_allow_html=True)

# # Logout
# st.sidebar.markdown("<br>", unsafe_allow_html=True)
# if st.sidebar.button("🚪 Logout", use_container_width=True):
#     logout()

# # Copyright
# st.sidebar.markdown("""
# <div style='text-align:center; padding:1.5rem 0 0.5rem 0;
#             border-top:1px solid #0f2040; margin-top:1rem;'>
#     <p style='color:#1e3a5f; font-size:0.68rem; margin:0; line-height:1.8;'>
#         © 2026 Smart Churn Intelligence<br>
#         <span style='color:#162032;'>Version 1.0 · All rights reserved</span>
#     </p>
# </div>
# """, unsafe_allow_html=True)


# # ─── HOME PAGE ───────────────────────────────────────────────────
# if page == "🏠 Home":

#     # Hero
#     st.markdown(f"""
#     <div style='background:linear-gradient(135deg,#0a1628 0%,#0d1f3c 60%,#0a1628 100%);
#                 border:1px solid #1e3a5f; border-radius:20px;
#                 padding:3.5rem 2rem; text-align:center; margin-bottom:2rem;
#                 box-shadow:0 8px 32px rgba(0,0,0,0.4),
#                            inset 0 1px 0 rgba(99,102,241,0.15);'>
#         <div style='font-size:4rem; margin-bottom:0.8rem; line-height:1;'>🦉</div>
#         <h1 style='color:#e2e8f0; font-size:2.6rem; font-weight:800;
#                    margin:0; letter-spacing:-1px; line-height:1.2;'>
#             Customer Churn Intelligence
#         </h1>
#         <p style='color:#475569; font-size:1.05rem; margin:0.8rem 0 0.4rem 0;'>
#             Predict · Understand · Retain — Powered by AI
#         </p>
#         <p style='color:#6366f1; font-size:0.9rem; margin:0; font-weight:500;'>
#             Welcome back, <b style='color:#818cf8;'>{st.session_state.username}</b> 👋
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

#     # Metrics
#     m1, m2, m3, m4 = st.columns(4)
#     with m1:
#         st.metric("Acquisition Cost",    "$200–500", help="Avg cost to acquire a new customer")
#     with m2:
#         st.metric("Retention Cost",      "$50–100",  delta="-75%", help="75% cheaper than acquisition")
#     with m3:
#         st.metric("Avg Churn Impact",    "15–25%",   delta="Revenue Loss", delta_color="inverse")
#     with m4:
#         st.metric("Early Detection ROI", "5–10×",    delta="+500%")

#     st.markdown("---")

#     col1, col2 = st.columns(2, gap="large")

#     with col1:
#         st.markdown("""
#         <div style='background:linear-gradient(135deg,#0a1628,#0d1f3c);
#                     border:1px solid #1e3a5f; border-radius:14px; padding:1.8rem;'>
#             <h3 style='color:#e2e8f0; margin:0 0 1.2rem 0; font-size:1.05rem; font-weight:700;'>
#                 🎯 What This Platform Does
#             </h3>
#             <div style='display:flex; flex-direction:column; gap:1rem;'>
#                 <div style='display:flex; gap:0.8rem; align-items:flex-start;'>
#                     <span style='background:rgba(99,102,241,0.15); border-radius:8px;
#                                  padding:0.35rem 0.55rem; font-size:1rem; flex-shrink:0;'>🌲</span>
#                     <div>
#                         <p style='color:#e2e8f0; font-size:0.87rem; font-weight:600; margin:0;'>Random Forest ML</p>
#                         <p style='color:#475569; font-size:0.77rem; margin:0.1rem 0 0 0;'>Score customers instantly with 94% test accuracy</p>
#                     </div>
#                 </div>
#                 <div style='display:flex; gap:0.8rem; align-items:flex-start;'>
#                     <span style='background:rgba(99,102,241,0.15); border-radius:8px;
#                                  padding:0.35rem 0.55rem; font-size:1rem; flex-shrink:0;'>🧠</span>
#                     <div>
#                         <p style='color:#e2e8f0; font-size:0.87rem; font-weight:600; margin:0;'>SHAP Explainability</p>
#                         <p style='color:#475569; font-size:0.77rem; margin:0.1rem 0 0 0;'>Understand exactly why a customer might leave</p>
#                     </div>
#                 </div>
#                 <div style='display:flex; gap:0.8rem; align-items:flex-start;'>
#                     <span style='background:rgba(99,102,241,0.15); border-radius:8px;
#                                  padding:0.35rem 0.55rem; font-size:1rem; flex-shrink:0;'>🤖</span>
#                     <div>
#                         <p style='color:#e2e8f0; font-size:0.87rem; font-weight:600; margin:0;'>Gemini AI Strategies</p>
#                         <p style='color:#475569; font-size:0.77rem; margin:0.1rem 0 0 0;'>Personalised retention actions per customer</p>
#                     </div>
#                 </div>
#                 <div style='display:flex; gap:0.8rem; align-items:flex-start;'>
#                     <span style='background:rgba(99,102,241,0.15); border-radius:8px;
#                                  padding:0.35rem 0.55rem; font-size:1rem; flex-shrink:0;'>📊</span>
#                     <div>
#                         <p style='color:#e2e8f0; font-size:0.87rem; font-weight:600; margin:0;'>Analytics Dashboard</p>
#                         <p style='color:#475569; font-size:0.77rem; margin:0.1rem 0 0 0;'>Explore historical churn patterns and trends</p>
#                     </div>
#                 </div>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)

#     with col2:
#         st.markdown("""
#         <div style='background:linear-gradient(135deg,#0a1628,#0d1f3c);
#                     border:1px solid #1e3a5f; border-radius:14px; padding:1.8rem;'>
#             <h3 style='color:#e2e8f0; margin:0 0 1.2rem 0; font-size:1.05rem; font-weight:700;'>
#                 🚀 How to Use
#             </h3>
#             <div style='display:flex; flex-direction:column; gap:0.9rem;'>
#                 <div style='display:flex; gap:0.8rem; align-items:flex-start;'>
#                     <span style='background:linear-gradient(135deg,#6366f1,#4f46e5);
#                                  border-radius:50%; width:26px; height:26px; flex-shrink:0;
#                                  display:flex; align-items:center; justify-content:center;
#                                  color:#fff; font-size:0.72rem; font-weight:700; margin-top:2px;'>1</span>
#                     <p style='color:#94a3b8; font-size:0.84rem; margin:0; padding-top:3px;'>
#                         Click <b style='color:#818cf8;'>🔮 Churn Prediction</b> in the sidebar
#                     </p>
#                 </div>
#                 <div style='display:flex; gap:0.8rem; align-items:flex-start;'>
#                     <span style='background:linear-gradient(135deg,#6366f1,#4f46e5);
#                                  border-radius:50%; width:26px; height:26px; flex-shrink:0;
#                                  display:flex; align-items:center; justify-content:center;
#                                  color:#fff; font-size:0.72rem; font-weight:700; margin-top:2px;'>2</span>
#                     <p style='color:#94a3b8; font-size:0.84rem; margin:0; padding-top:3px;'>
#                         Fill in the customer attributes form
#                     </p>
#                 </div>
#                 <div style='display:flex; gap:0.8rem; align-items:flex-start;'>
#                     <span style='background:linear-gradient(135deg,#6366f1,#4f46e5);
#                                  border-radius:50%; width:26px; height:26px; flex-shrink:0;
#                                  display:flex; align-items:center; justify-content:center;
#                                  color:#fff; font-size:0.72rem; font-weight:700; margin-top:2px;'>3</span>
#                     <p style='color:#94a3b8; font-size:0.84rem; margin:0; padding-top:3px;'>
#                         Hit <b style='color:#818cf8;'>Predict Churn Risk</b> to get the score
#                     </p>
#                 </div>
#                 <div style='display:flex; gap:0.8rem; align-items:flex-start;'>
#                     <span style='background:linear-gradient(135deg,#6366f1,#4f46e5);
#                                  border-radius:50%; width:26px; height:26px; flex-shrink:0;
#                                  display:flex; align-items:center; justify-content:center;
#                                  color:#fff; font-size:0.72rem; font-weight:700; margin-top:2px;'>4</span>
#                     <p style='color:#94a3b8; font-size:0.84rem; margin:0; padding-top:3px;'>
#                         Review <b style='color:#818cf8;'>SHAP charts</b> — see which features drive the prediction
#                     </p>
#                 </div>
#                 <div style='display:flex; gap:0.8rem; align-items:flex-start;'>
#                     <span style='background:linear-gradient(135deg,#6366f1,#4f46e5);
#                                  border-radius:50%; width:26px; height:26px; flex-shrink:0;
#                                  display:flex; align-items:center; justify-content:center;
#                                  color:#fff; font-size:0.72rem; font-weight:700; margin-top:2px;'>5</span>
#                     <p style='color:#94a3b8; font-size:0.84rem; margin:0; padding-top:3px;'>
#                         Get <b style='color:#818cf8;'>Gemini AI</b> retention strategy & download report
#                     </p>
#                 </div>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)
#     st.success("✅ **Ready to start?** Click **🔮 Churn Prediction** in the sidebar!")


# # ─── PREDICTION PAGE ─────────────────────────────────────────────
# elif page == "🔮 Churn Prediction":
#     from pages.prediction import show_prediction_page
#     show_prediction_page()


# # ─── ANALYTICS PAGE ──────────────────────────────────────────────
# elif page == "📊 Analytics":
#     from pages.historical import show_historical_page
#     show_historical_page()
import streamlit as st
import os
import sys
import json
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ─── INITIALIZATION ──────────────────────────────────────────────
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    css_file = "assets/style.css"
    if os.path.exists(css_file):
        with open(css_file, encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ─── PLOT THEME CONFIGURATION ────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0a1628',
    'axes.facecolor':    '#0a1628',
    'savefig.facecolor': '#0a1628',
    'figure.edgecolor':  '#0a1628',
    'axes.edgecolor':    '#1e3a5f',
    'text.color':        '#e2e8f0',
    'axes.labelcolor':   '#94a3b8',
    'xtick.color':       '#94a3b8',
    'ytick.color':       '#e2e8f0',
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'grid.color':        '#1e3a5f',
    'grid.alpha':        0.5,
    'axes.titlecolor':   '#e2e8f0',
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'legend.facecolor':  '#0d1f3c',
    'legend.edgecolor':  '#1e3a5f',
    'legend.labelcolor': '#e2e8f0',
    'legend.fontsize':   9,
    'axes.prop_cycle':   plt.cycler(color=[
        '#6366f1', '#10b981', '#f59e0b',
        '#ef4444', '#0ea5e9', '#8b5cf6'
    ]),
    'font.family':       'sans-serif',
    'font.size':         10,
})

# ─── AUTH HELPERS ────────────────────────────────────────────────
def load_users():
    try:
        with open("users.json", "r", encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {"admin": "admin123"}

def save_users(users):
    with open("users.json", "w", encoding='utf-8') as f:
        json.dump(users, f, indent=2)

def check_login(username, password):
    users = load_users()
    return users.get(username) == password

def register_user(username, password):
    users = load_users()
    if username in users:
        return False, "❌ Username already exists. Try a different one."
    if len(password) < 6:
        return False, "⚠️ Password must be at least 6 characters."
    users[username] = password
    save_users(users)
    return True, "✅ Account created successfully!"

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.current_page = "Home"
    st.rerun()

# ─── AUTH PAGE ───────────────────────────────────────────────────
def show_auth_page():
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
        .auth-card {
            max-width: 460px;
            margin: 4rem auto 0 auto;
            background: linear-gradient(135deg, #0a1628, #0d1f3c);
            border: 1px solid #1e3a5f;
            border-radius: 20px;
            padding: 2.8rem 2.5rem 2rem 2.5rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.6), inset 0 1px 0 rgba(99,102,241,0.1);
        }
        .auth-icon { text-align:center; font-size:3.2rem; line-height:1; margin-bottom:0.4rem; }
        .auth-title { text-align:center; color:#e2e8f0; font-size:1.75rem; font-weight:800; margin:0; letter-spacing:-0.5px; }
        .auth-sub { text-align:center; color:#334155; font-size:0.75rem; margin:0.3rem 0 0 0; text-transform:uppercase; letter-spacing:1.2px; }
        .auth-hint { text-align:center; color:#334155; font-size:0.75rem; margin-top:1.2rem; line-height:1.9; }
        .auth-hint b { color:#475569; }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.6, 1])
    with col2:
        st.markdown("""
        <div class='auth-card'>
            <div class='auth-icon'>🦉</div>
            <h1 class='auth-title'>Smart Churn</h1>
            <p class='auth-sub'>Customer Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        tab_login, tab_signup = st.tabs(["🔐 Sign In", "✨ Sign Up"])

        with tab_login:
            with st.form("login_form"):
                u = st.text_input("Username", placeholder="Enter your username", key="login_user")
                p = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.form_submit_button("🔐 Sign In", use_container_width=True):
                    if not u or not p: st.warning("⚠️ Please enter both.")
                    elif check_login(u, p):
                        st.session_state.logged_in, st.session_state.username = True, u
                        st.rerun()
                    else: st.error("❌ Invalid username or password.")

        with tab_signup:
            with st.form("signup_form"):
                new_u = st.text_input("Choose Username", key="signup_user")
                new_p = st.text_input("Choose Password", type="password", key="signup_pass")
                conf_p = st.text_input("Confirm Password", type="password", key="signup_confirm")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.form_submit_button("✨ Create Account", use_container_width=True):
                    if new_p != conf_p: st.error("❌ Passwords do not match.")
                    else:
                        success, msg = register_user(new_u, new_p)
                        if success: st.success(f"✅ Account created! Sign in as {new_u}")
                        else: st.error(msg)

# ─── SESSION STATE ───────────────────────────────────────────────
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = ""
if "current_page" not in st.session_state: st.session_state.current_page = "Home"

if not st.session_state.logged_in:
    show_auth_page()
    st.stop()

# ─── SIDEBAR ─────────────────────────────────────────────────────
st.sidebar.markdown(f"""
<div style='text-align:center; padding:2rem 1rem 1.5rem 1rem;'>
    <div style='font-size:3rem; line-height:1;'>🦉</div>
    <h2 style='color:#e2e8f0; font-size:1.25rem; font-weight:800; margin:0.6rem 0 0.2rem 0; letter-spacing:-0.3px;'>Smart Churn</h2>
    <p style='color:#334155; font-size:0.72rem; margin:0; text-transform:uppercase; letter-spacing:1.2px;'>Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# User badge
st.sidebar.markdown(f"""
<div style='background:rgba(99,102,241,0.06); border:1px solid #1e3a5f; border-radius:10px; padding:0.7rem 1rem; margin-bottom:1.2rem;'>
    <div style='display:flex; align-items:center; gap:0.7rem;'>
        <div style='background:linear-gradient(135deg,#6366f1,#4f46e5); border-radius:50%; width:32px; height:32px; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-size:0.9rem;'>👤</div>
        <div>
            <p style='color:#475569; font-size:0.62rem; margin:0; text-transform:uppercase; letter-spacing:0.8px;'>SIGNED IN AS</p>
            <p style='color:#e2e8f0; font-size:0.88rem; margin:0; font-weight:600;'>{st.session_state.username}</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
# ─── NAVIGATION (FIXED TEXT VISIBILITY) ─────────────────────────

st.sidebar.markdown("""
<style>
/* 1. Reset the radio button container */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
}

/* 2. Style the boxes */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background-color: rgba(255, 255, 255, 0.05) !important; 
    border: 1px solid #1e3a5f !important;
    border-radius: 12px !important;
    padding: 0.7rem 1rem !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
}

/* 3. THE FIX: Force the Page Name Text to be Visible and White */
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: #ffffff !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    visibility: visible !important;
    display: block !important;
    opacity: 1 !important;
    margin: 0 !important;
}

/* 4. Hide the default radio circle */
section[data-testid="stSidebar"] div[role="radiogroup"] [data-baseweb="radio"] div:first-child {
    display: none !important;
}

/* 5. Selected state */
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    border-color: #818cf8 !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
}
</style>
""", unsafe_allow_html=True)

page_labels = ["🏠 Home", "🔮 Churn Prediction", "📊 Analytics"]
page_keys   = ["Home", "Prediction", "Analytics"]

# Check if current page is valid
if st.session_state.current_page not in page_keys:
    st.session_state.current_page = "Home"

current_idx = page_keys.index(st.session_state.current_page)

# Radio Navigation
selected = st.sidebar.radio(
    "Navigate", 
    page_labels, 
    index=current_idx, 
    label_visibility="collapsed"
)

# Update session state
st.session_state.current_page = page_keys[page_labels.index(selected)]
# # ─── NAVIGATION (FIXED TEXT VISIBILITY) ─────────────────────────

# st.sidebar.markdown("""
# <style>
# /* 1. Base label style - ensure it's a block that can contain text */
# section[data-testid="stSidebar"] div[role="radiogroup"] label {
#     background-color: rgba(255, 255, 255, 0.08) !important; 
#     border: 1px solid #2a4a7a !important;
#     border-radius: 12px !important;
#     padding: 0.7rem 1rem !important;
#     margin-bottom: 8px !important;
#     display: flex !important;
#     align-items: center !important;
#     width: 100% !important;
#     cursor: pointer !important;
# }

# /* 2. CRITICAL: Force the text container to be visible after a click */
# section[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {
#     display: block !important;
#     visibility: visible !important;
#     opacity: 1 !important;
# }

# /* 3. Force the actual page name text to be white */
# section[data-testid="stSidebar"] div[role="radiogroup"] label p {
#     color: #ffffff !important;
#     font-size: 0.95rem !important;
#     font-weight: 600 !important;
#     opacity: 1 !important;
#     visibility: visible !important;
#     margin: 0 !important;
#     display: block !important;
# }

# /* 4. Selected state gradient */
# section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
#     background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
#     border-color: #818cf8 !important;
#     box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
# }

# /* 5. Hide the default ugly radio circle */
# section[data-testid="stSidebar"] div[role="radiogroup"] [data-baseweb="radio"] div:first-child {
#     display: none !important;
# }
# </style>
# """, unsafe_allow_html=True)
# page_labels = ["🏠 Home", "🔮 Churn Prediction", "📊 Analytics"]
# page_keys   = ["Home", "Prediction", "Analytics"]
# current_idx = page_keys.index(st.session_state.current_page) if st.session_state.current_page in page_keys else 0

# selected = st.sidebar.radio("Navigate", page_labels, index=current_idx, label_visibility="collapsed")
# st.session_state.current_page = page_keys[page_labels.index(selected)]

st.sidebar.markdown("---")

# Tech stack
st.sidebar.markdown("""
<div style='background:rgba(99,102,241,0.06); border:1px solid #1e3a5f; border-left:3px solid #6366f1; border-radius:0 10px 10px 0; padding:0.9rem 1rem; margin-bottom:0.8rem;'>
    <p style='color:#334155; font-size:0.62rem; margin:0 0 0.6rem 0; text-transform:uppercase; letter-spacing:1.2px; font-weight:700;'>POWERED BY</p>
    <div style='display:flex; flex-direction:column; gap:0.45rem;'>
        <span style='color:#94a3b8; font-size:0.8rem;'>🌲 &nbsp;Random Forest ML</span>
        <span style='color:#94a3b8; font-size:0.8rem;'>🧠 &nbsp;SHAP Explainability</span>
        <span style='color:#94a3b8; font-size:0.8rem;'>🤖 &nbsp;Gemini 2.0 Flash</span>
    </div>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("🚪 Logout", use_container_width=True, key="logout_btn"): logout()

st.sidebar.markdown("""
<div style='text-align:center; padding:1.5rem 0 0.5rem 0; border-top:1px solid #0f2040; margin-top:1rem;'>
    <p style='color:#1e3a5f; font-size:0.68rem; margin:0; line-height:1.8;'>© 2026 Smart Churn Intelligence<br><span style='color:#162032;'>Version 1.0 · All rights reserved</span></p>
</div>
""", unsafe_allow_html=True)

# ─── ROUTING ───
page = st.session_state.current_page

if page == "Home":
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0a1628 0%,#0d1f3c 60%,#0a1628 100%); border:1px solid #1e3a5f; border-radius:20px; padding:3.5rem 2rem; text-align:center; margin-bottom:2rem; box-shadow:0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(99,102,241,0.15);'>
        <div style='font-size:4rem; margin-bottom:0.8rem; line-height:1;'>🦉</div>
        <h1 style='color:#e2e8f0; font-size:2.6rem; font-weight:800; margin:0; letter-spacing:-1px; line-height:1.2;'>Customer Churn Intelligence</h1>
        <p style='color:#475569; font-size:1.05rem; margin:0.8rem 0 0.4rem 0;'>Predict · Understand · Retain — Powered by AI</p>
        <p style='color:#6366f1; font-size:0.9rem; margin:0; font-weight:500;'>Welcome back, <b style='color:#818cf8;'>{st.session_state.username}</b> 👋</p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Acquisition Cost", "$200–500")
    with m2: st.metric("Retention Cost", "$50–100", delta="-75%")
    with m3: st.metric("Avg Churn Impact", "15–25%", delta="Revenue Loss", delta_color="inverse")
    with m4: st.metric("Early Detection ROI", "5–10×", delta="+500%")

    st.markdown("---")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#0a1628,#0d1f3c); border:1px solid #1e3a5f; border-radius:14px; padding:1.8rem;'>
            <h3 style='color:#e2e8f0; margin:0 0 1.2rem 0; font-size:1.05rem; font-weight:700;'>🎯 What This Platform Does</h3>
            <div style='display:flex; flex-direction:column; gap:1rem;'>
                <div style='display:flex; gap:0.8rem; align-items:flex-start;'>
                    <span style='background:rgba(99,102,241,0.15); border-radius:8px; padding:0.35rem 0.55rem; font-size:1rem;'>🌲</span>
                    <div><p style='color:#e2e8f0; font-size:0.87rem; font-weight:600; margin:0;'>Random Forest ML</p><p style='color:#475569; font-size:0.77rem; margin:0.1rem 0 0 0;'>Score customers instantly with 94% test accuracy</p></div>
                </div>
                <div style='display:flex; gap:0.8rem; align-items:flex-start;'>
                    <span style='background:rgba(99,102,241,0.15); border-radius:8px; padding:0.35rem 0.55rem; font-size:1rem;'>🧠</span>
                    <div><p style='color:#e2e8f0; font-size:0.87rem; font-weight:600; margin:0;'>SHAP Explainability</p><p style='color:#475569; font-size:0.77rem; margin:0.1rem 0 0 0;'>Understand exactly why a customer might leave</p></div>
                </div>
                <div style='display:flex; gap:0.8rem; align-items:flex-start;'>
                    <span style='background:rgba(99,102,241,0.15); border-radius:8px; padding:0.35rem 0.55rem; font-size:1rem;'>🤖</span>
                    <div><p style='color:#e2e8f0; font-size:0.87rem; font-weight:600; margin:0;'>Gemini AI Strategies</p><p style='color:#475569; font-size:0.77rem; margin:0.1rem 0 0 0;'>Personalised retention actions per customer</p></div>
                </div>
                <div style='display:flex; gap:0.8rem; align-items:flex-start;'>
                    <span style='background:rgba(99,102,241,0.15); border-radius:8px; padding:0.35rem 0.55rem; font-size:1rem;'>📊</span>
                    <div><p style='color:#e2e8f0; font-size:0.87rem; font-weight:600; margin:0;'>Analytics Dashboard</p><p style='color:#475569; font-size:0.77rem; margin:0.1rem 0 0 0;'>Explore historical churn patterns and trends</p></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#0a1628,#0d1f3c); border:1px solid #1e3a5f; border-radius:14px; padding:1.8rem;'>
            <h3 style='color:#e2e8f0; margin:0 0 1.2rem 0; font-size:1.05rem; font-weight:700;'>🚀 How to Use</h3>
            <div style='display:flex; flex-direction:column; gap:0.9rem;'>
                <div style='display:flex; gap:0.8rem; align-items:flex-start;'><span style='background:linear-gradient(135deg,#6366f1,#4f46e5); border-radius:50%; width:26px; height:26px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:0.72rem; font-weight:700;'>1</span><p style='color:#94a3b8; font-size:0.84rem; padding-top:3px;'>Click <b style='color:#818cf8;'>🔮 Churn Prediction</b> in the sidebar</p></div>
                <div style='display:flex; gap:0.8rem; align-items:flex-start;'><span style='background:linear-gradient(135deg,#6366f1,#4f46e5); border-radius:50%; width:26px; height:26px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:0.72rem; font-weight:700;'>2</span><p style='color:#94a3b8; font-size:0.84rem; padding-top:3px;'>Fill in the customer attributes form</p></div>
                <div style='display:flex; gap:0.8rem; align-items:flex-start;'><span style='background:linear-gradient(135deg,#6366f1,#4f46e5); border-radius:50%; width:26px; height:26px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:0.72rem; font-weight:700;'>3</span><p style='color:#94a3b8; font-size:0.84rem; padding-top:3px;'>Hit <b style='color:#818cf8;'>Predict Churn Risk</b> to get the score</p></div>
                <div style='display:flex; gap:0.8rem; align-items:flex-start;'><span style='background:linear-gradient(135deg,#6366f1,#4f46e5); border-radius:50%; width:26px; height:26px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:0.72rem; font-weight:700;'>4</span><p style='color:#94a3b8; font-size:0.84rem; padding-top:3px;'>Review <b style='color:#818cf8;'>SHAP charts</b> — see which features drive the prediction</p></div>
                <div style='display:flex; gap:0.8rem; align-items:flex-start;'><span style='background:linear-gradient(135deg,#6366f1,#4f46e5); border-radius:50%; width:26px; height:26px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:0.72rem; font-weight:700;'>5</span><p style='color:#94a3b8; font-size:0.84rem; padding-top:3px;'>Get <b style='color:#818cf8;'>Gemini AI</b> retention strategy & download report</p></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.success("✅ **Ready to start?** Click **🔮 Churn Prediction** in the sidebar!")

elif page == "Prediction":
    from pages.prediction import show_prediction_page
    show_prediction_page()

elif page == "Analytics":
    from pages.historical import show_historical_page
    show_historical_page()
