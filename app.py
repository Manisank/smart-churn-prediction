
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
/* 1. Reset the radio group container */
[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 10px !important;
    padding-top: 1rem !important;
}

/* 2. Style the Button Boxes */
[data-testid="stSidebar"] div[role="radiogroup"] label {
    background-color: rgba(255, 255, 255, 0.05) !important; 
    border: 1px solid #1e3a5f !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.2rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important; /* Align text to left */
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
}

/* 3. THE MAGIC FIX: This forces the text to be visible and white */
[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: #ffffff !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    line-height: 1.4 !important;
    text-shadow: none !important;
}

/* 4. Ensure the container doesn't collapse the text */
[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {
    display: block !important;
    visibility: visible !important;
    width: 100% !important;
}

/* 5. Hide ONLY the default radio dot */
[data-testid="stSidebar"] div[role="radiogroup"] [data-baseweb="radio"] div:first-child {
    display: none !important;
}

/* 6. Selected state styling */
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
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
