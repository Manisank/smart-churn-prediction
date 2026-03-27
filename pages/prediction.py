import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import google.generativeai as genai
import os

# ─── DARK THEME CONSTANTS ────────────────────────────────────────
BG_DARK      = '#0a1628'
BG_CARD      = '#0d1f3c'
BORDER       = '#1e3a5f'
TEXT_PRIMARY = '#e2e8f0'
TEXT_MUTED   = '#94a3b8'
ACCENT       = '#6366f1'

def apply_dark_theme(fig):
    """Apply dark theme to matplotlib figures."""
    fig.patch.set_facecolor(BG_DARK)
    for ax in fig.get_axes():
        ax.set_facecolor(BG_DARK)
        ax.tick_params(colors=TEXT_PRIMARY)
        ax.spines['bottom'].set_color(BORDER)
        ax.spines['left'].set_color(BORDER)

# ─── LOAD MODELS ─────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('models/rf_churn_model.pkl')
        encoders = joblib.load('models/encoders.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        st.success("✅ Models loaded")
        return model, encoders, feature_names
    except Exception as e:
        st.error(f"❌ Load error: {e}")
        return None, None, None

# ─── GEMINI ─────────────────────────────────────────────────────
def get_retention_strategy(churn_prob, top_factors):
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Customer churn risk: {churn_prob:.1%}. Top factors: {', '.join(top_factors[:3])}. Give 3 retention strategies."
        response = model.generate_content(prompt)
        return response.text
    except:
        return "• Offer personalized discounts\n• Send re-engagement emails\n• Improve app experience"

# ─── MAIN ───────────────────────────────────────────────────────
def show_prediction_page():
    st.title("🔮 Churn Prediction")
    st.markdown("Enter customer data to predict churn risk")

    model, encoders, feature_names = load_artifacts()
    if model is None:
        return

    # Inputs (same as before)
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.number_input("Tenure", 0, 100, 12)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        warehouse_to_home = st.number_input("Distance (km)", 0, 200, 15)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        hours_on_app = st.number_input("App Hours", 0, 24, 3)
        satisfaction_score = st.slider("Satisfaction", 1, 5, 3)
    with col3:
        preferred_order_cat = st.selectbox("Category", ["Fashion", "Electronics", "Others"])
        marital_status = st.selectbox("Status", ["Single", "Married"])
        order_count = st.number_input("Orders", 0, 50, 5)

    col4, col5 = st.columns(2)
    with col4:
        complain = st.checkbox("Complained")
        day_since_last_order = st.number_input("Days Since Order", 0, 50, 7)
    with col5:
        cashback_amount = st.number_input("Cashback ($)", 0.0, 500.0, 100.0)

    if st.button("🔮 PREDICT", use_container_width=True):
        # Prepare input
        input_data = {
            'Tenure': tenure, 'CityTier': city_tier, 'WarehouseToHome': warehouse_to_home,
            'Gender': gender, 'HourSpendOnApp': hours_on_app, 'SatisfactionScore': satisfaction_score,
            'PreferedOrderCat': preferred_order_cat, 'MaritalStatus': marital_status,
            'Complain': int(complain), 'OrderCount': order_count, 'DaySinceLastOrder': day_since_last_order,
            'CashbackAmount': cashback_amount, 'OrderAmountHikeFromlastYear': 15, 'CouponUsed': 3
        }

        df_input = pd.DataFrame([input_data])
        
        # Encode
        for col, encoder in encoders.items():
            if col in df_input:
                df_input[col] = encoder.transform(df_input[col])
        
        df_input = df_input[feature_names]

        # Predict
        churn_prob = model.predict_proba(df_input)[0][1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Churn Risk", f"{churn_prob:.1%}")
        with col2:
            if churn_prob > 0.5:
                st.error("🚨 HIGH RISK")
            else:
                st.success("✅ LOW RISK")

        # SHAP - DEPLOYMENT PROOF
        with st.spinner("Generating explanations..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(df_input)
            
            # Extract churn class (index 1)
            sv_churn = shap_values.values[:, :, 1] if shap_values.values.ndim == 3 else shap_values.values
            base_val = float(np.array(explainer.expected_value)[1])
            row_sv = sv_churn[0]

        # Plots
        tab1, tab2 = st.tabs(["📊 Top Features", "🎯 SHAP Force Plot"])
        
        with tab1:
            shap.summary_plot(shap_values[:, :, 1], df_input, plot_type="bar", show=False, max_display=10)
            plt.gcf().patch.set_facecolor(BG_DARK)
            st.pyplot(plt.gcf())
            plt.close()

        with tab2:
            shap.force_plot(base_val, row_sv, df_input.iloc[0], matplotlib=True, show=False)
            plt.gcf().patch.set_facecolor(BG_DARK)
            st.pyplot(plt.gcf())
            plt.close()

        # Recommendations
        top_features = pd.Series(np.abs(row_sv), index=feature_names).nlargest(3).index.tolist()
        st.markdown("## 🦉 Retention Strategies")
        st.success(get_retention_strategy(churn_prob, top_features))
