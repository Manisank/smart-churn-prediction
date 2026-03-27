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

# ─── LOAD ARTIFACTS ──────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    try:
        model         = joblib.load('models/rf_churn_model.pkl')
        explainer     = joblib.load('models/shap_explainer.pkl')
        encoders      = joblib.load('models/encoders.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        st.success(f"✅ Model loaded successfully with {len(feature_names)} features")
        return model, encoders, feature_names, explainer
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
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
        # Using stable 1.5-flash for deployment reliability
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        You are an E-commerce retention strategist. A customer has a {churn_prob:.1%} churn probability.
        Top Risk Factors: {', '.join(top_factors)}
        Provide 3-5 specific, actionable retention strategies in bullet points. Keep it under 150 words.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error generating strategy: {str(e)}"

# ─── HELPER FOR SHAP OUTPUTS ─────────────────────────────────────
def get_churn_shap_outputs(explainer, shap_vals):
    """
    Normalize SHAP outputs for binary RF:
    - sv_churn: SHAP values for churn class
    - base_val: scalar expected value for churn class
    """
    # Handle SHAP values
    if isinstance(shap_vals, list):
        # list of [class0, class1] -> take churn/positive class = index 1
        sv_churn = shap_vals[1]
    elif np.ndim(shap_vals) == 3:
        # shape (n_samples, n_features, n_classes) or similar
        sv_churn = shap_vals[0, :, 1]
    else:
        sv_churn = shap_vals

    # Handle expected_value (can be scalar, list, or np array)
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = np.array(ev)
        if ev.ndim == 0:
            base_val = float(ev)
        else:
            # assume index 1 is churn/positive class
            base_val = float(ev[1]) if ev.size > 1 else float(ev.flatten()[0])
    else:
        base_val = float(ev)

    return sv_churn, base_val

# ─── MAIN PAGE ───────────────────────────────────────────────────
def show_prediction_page():
    st.title("🔮 Customer Churn Prediction")
    st.markdown("### Predict churn risk and get AI-powered retention strategies")
    st.markdown("---")

    model, encoders, feature_names, explainer = load_model_artifacts()
    if model is None:
        return

    st.markdown("## 📝 Enter Customer Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.number_input("Tenure (months)", 0, 100, 12)
        preferred_login_device = st.selectbox("Preferred Login Device", ["Mobile Phone", "Computer", "Tablet"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        warehouse_to_home = st.number_input("Warehouse to Home (km)", 0, 200, 15)

    with col2:
        preferred_payment_mode = st.selectbox("Payment Mode", ["Debit Card", "Credit Card", "E wallet", "UPI", "Cash on Delivery", "COD"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        hours_on_app = st.number_input("Hours on App", 0, 24, 3)
        device_registered = st.number_input("Number of Devices", 1, 10, 2)

    with col3:
        preferred_order_cat = st.selectbox("Preferred Category", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"])
        satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        address_count = st.number_input("Number of Addresses", 1, 20, 2)

    col4, col5 = st.columns(2)
    with col4:
        complain = st.selectbox("Has Complained?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        order_count = st.number_input("Order Count", 0, 50, 5)
    with col5:
        day_since_last_order = st.number_input("Days Since Last Order", 0, 50, 7)
        cashback_amount = st.number_input("Cashback Amount ($)", 0.0, 500.0, 100.0)

    st.markdown("---")

    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        input_data = {
            'Tenure': tenure,
            'PreferredLoginDevice': preferred_login_device,
            'CityTier': city_tier,
            'WarehouseToHome': warehouse_to_home,
            'PreferredPaymentMode': preferred_payment_mode,
            'Gender': gender,
            'HourSpendOnApp': hours_on_app,
            'NumberOfDeviceRegistered': device_registered,
            'PreferedOrderCat': preferred_order_cat,
            'SatisfactionScore': satisfaction_score,
            'MaritalStatus': marital_status,
            'NumberOfAddress': address_count,
            'Complain': complain,
            'OrderAmountHikeFromlastYear': 15,
            'CouponUsed': 3,
            'OrderCount': order_count,
            'DaySinceLastOrder': day_since_last_order,
            'CashbackAmount': cashback_amount
        }

        df_input = pd.DataFrame([input_data])

        # Apply encoders
        for col, encoder in encoders.items():
            if col in df_input.columns:
                df_input[col] = encoder.transform(df_input[col])

        # Ensure column order
        df_input = df_input[feature_names]

        # Prediction
        churn_prob = model.predict_proba(df_input)[0][1]
        churn_label = "HIGH RISK" if churn_prob > 0.5 else "LOW RISK"

        # 🎯 Results
        st.markdown("## 🎯 Prediction Results")
        if churn_prob > 0.5:
            st.error(f"### ⚠️ {churn_label}")
        else:
            st.success(f"### ✅ {churn_label}")

        st.metric("Churn Probability", f"{churn_prob:.1%}")

        # 🧠 SHAP Calculation and Plots
        st.markdown("---")
        st.markdown("## 🧠 Explainable AI (SHAP Analysis)")
        with st.spinner("Calculating SHAP values..."):
            try:
                shap_vals = explainer.shap_values(df_input)
            except Exception:
                # Re-init explainer on Cloud to fix any version issues
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(df_input)

            sv_churn, base_val = get_churn_shap_outputs(explainer, shap_vals)

        # Tabs
        t1, t2, t3 = st.tabs(["📊 Feature Importance", "🎯 Force Plot", "💧 Waterfall"])

        with t1:
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(sv_churn, df_input, plot_type="bar", show=False, color=ACCENT)
            apply_dark_axes(fig, plt.gca())
            st.pyplot(fig)
            plt.close()

        with t2:
            # Force Plot needs 1D values for a single row
            row_sv = sv_churn[0] if np.ndim(sv_churn) > 1 else sv_churn
            shap.force_plot(base_val, row_sv, df_input.iloc[0], matplotlib=True, show=False)
            fig = plt.gcf()
            fig.patch.set_facecolor(BG_DARK)
            for ax in fig.get_axes():
                ax.set_facecolor(BG_DARK)
            st.pyplot(fig)
            plt.close()

        with t3:
            row_sv = sv_churn[0] if np.ndim(sv_churn) > 1 else sv_churn
            explanation = shap.Explanation(
                values=row_sv,
                base_values=base_val,
                data=df_input.iloc[0].values,
                feature_names=feature_names
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(explanation, show=False)
            apply_dark_axes(fig, plt.gca())
            st.pyplot(fig)
            plt.close()

        # Owl Strategy
        st.markdown("---")
        st.markdown("## 🦉 Smart Owl Recommendations")

        row_sv = sv_churn[0] if np.ndim(sv_churn) > 1 else sv_churn
        top_factors = pd.Series(np.abs(row_sv), index=feature_names).sort_values(ascending=False).head(3).index.tolist()

        with st.spinner("Gemini is analyzing..."):
            strategy = generate_retention_strategy(input_data, churn_prob, top_factors)
            st.success(strategy)
