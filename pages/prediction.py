import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import google.generativeai as genai
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ─── DARK THEME CONSTANTS ────────────────────────────────────────
BG_DARK      = '#0a1628'
BG_CARD      = '#0d1f3c'
BORDER       = '#1e3a5f'
TEXT_PRIMARY = '#e2e8f0'
TEXT_MUTED   = '#94a3b8'
ACCENT       = '#6366f1'
ACCENT_LIGHT = '#818cf8'

def apply_dark_axes(fig, ax):
    """Apply dark theme to any matplotlib figure/axes."""
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.tick_params(colors=TEXT_PRIMARY, labelsize=10)
    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)
    ax.title.set_color(TEXT_PRIMARY)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

# ─── SHAP HELPERS ────────────────────────────────────────────────
def get_shap_churn_values(shap_values):
    if isinstance(shap_values, list):
        return shap_values[1]  # For Random Forest classifiers
    elif len(shap_values.shape) == 3:
        return shap_values[:, :, 1]
    return shap_values

def get_shap_base_value(explainer):
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        return float(ev[1])
    return float(ev)

# ─── LOAD ARTIFACTS ──────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    try:
        model         = joblib.load('models/rf_churn_model.pkl')
        explainer     = joblib.load('models/shap_explainer.pkl')
        encoders      = joblib.load('models/encoders.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        st.success(f"✅ Model loaded successfully")
        return model, encoders, feature_names, explainer
    except Exception as e:
        st.error(f"❌ Error loading artifacts: {e}")
        return None, None, None, None

# ─── GEMINI ──────────────────────────────────────────────────────
def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        return True
    return False

def generate_retention_strategy(customer_data, churn_prob, top_factors):
    if not configure_gemini():
        return "⚠️ Gemini API key not configured."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Customer churn probability: {churn_prob:.1%}. Risk Factors: {', '.join(top_factors)}. Provide 3 brief retention strategies."
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ─── MAIN PAGE ───────────────────────────────────────────────────
def show_prediction_page():
    st.title("🔮 Customer Churn Prediction")
    st.markdown("---")

    model, encoders, feature_names, explainer = load_model_artifacts()
    if model is None: return

    # 📝 Input Form
    st.markdown("## 📝 Enter Customer Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.number_input("Tenure (months)", 0, 100, 12)
        pref_login = st.selectbox("Preferred Login Device", ["Mobile Phone", "Computer", "Tablet"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        warehouse = st.number_input("Warehouse to Home (km)", 0, 200, 15)

    with col2:
        payment = st.selectbox("Payment Mode", ["Debit Card", "Credit Card", "E wallet", "UPI", "COD"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        hours = st.number_input("Hours on App", 0, 24, 3)
        devices = st.number_input("Number of Devices", 1, 10, 2)

    with col3:
        cat = st.selectbox("Preferred Category", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"])
        sat_score = st.slider("Satisfaction Score", 1, 5, 3)
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        addresses = st.number_input("Number of Addresses", 1, 20, 2)

    col4, col5, col6 = st.columns(3)
    with col4:
        complain = st.selectbox("Has Complained?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        order_count = st.number_input("Order Count", 0, 50, 5)
    with col5:
        days_since = st.number_input("Days Since Last Order", 0, 50, 7)
        cashback = st.number_input("Cashback Amount ($)", 0.0, 500.0, 100.0)
    with col6:
        hike = st.number_input("Order Amount Hike (%)", 0, 50, 15)
        coupons = st.number_input("Coupons Used", 0, 20, 3)

    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        # ✅ FIXED: Added missing features to prevent KeyError
        input_data = {
            'Tenure': tenure,
            'PreferredLoginDevice': pref_login,
            'CityTier': city_tier,
            'WarehouseToHome': warehouse,
            'PreferredPaymentMode': payment,
            'Gender': gender,
            'HourSpendOnApp': hours,
            'NumberOfDeviceRegistered': devices,
            'PreferedOrderCat': cat,
            'SatisfactionScore': sat_score,
            'MaritalStatus': marital,
            'NumberOfAddress': addresses,
            'Complain': complain,
            'OrderAmountHikeFromlastYear': hike,
            'CouponUsed': coupons,
            'OrderCount': order_count,
            'DaySinceLastOrder': days_since,
            'CashbackAmount': cashback
        }

        df_input = pd.DataFrame([input_data])
        for col, encoder in encoders.items():
            if col in df_input.columns:
                df_input[col] = encoder.transform(df_input[col])

        df_input = df_input[feature_names]
        churn_prob = model.predict_proba(df_input)[0][1]
        churn_label = "HIGH RISK" if churn_prob > 0.5 else "LOW RISK"

        st.markdown("## 🎯 Prediction Results")
        st.metric("Churn Probability", f"{churn_prob:.1%}")
        if churn_prob > 0.5: st.error(f"### ⚠️ {churn_label}")
        else: st.success(f"### ✅ {churn_label}")

        st.markdown("---")
        st.markdown("## 🧠 Explainable AI (SHAP)")

        with st.spinner("Calculating SHAP..."):
            try:
                shap_values = explainer.shap_values(df_input)
            except:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(df_input)
            
            sv_churn = get_shap_churn_values(shap_values)
            base_val = get_shap_base_value(explainer)

        tab1, tab2 = st.tabs(["📊 Importance", "🎯 Force Plot"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(sv_churn, df_input, plot_type="bar", show=False, color=ACCENT)
            apply_dark_axes(fig, plt.gca())
            st.pyplot(fig)
            plt.close()

        with tab2:
            shap.force_plot(base_val, sv_churn[0], df_input.iloc[0], matplotlib=True, show=False)
            fig = plt.gcf()
            fig.patch.set_facecolor(BG_DARK)
            for ax in fig.get_axes(): ax.set_facecolor(BG_DARK)
            st.pyplot(fig)
            plt.close()

        # Recommendation logic
        top_factors = pd.Series(np.abs(sv_churn[0]), index=feature_names).nlargest(3).index.tolist()
        st.markdown("## 🦉 Smart Recommendations")
        st.info(generate_retention_strategy(input_data, churn_prob, top_factors))
