# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import shap
# import matplotlib.pyplot as plt
# import google.generativeai as genai
# import os
# from io import BytesIO

# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier


# # ✅ Helper: handles both old (list) and new (3D array) SHAP versions
# def get_shap_churn_values(shap_values):
#     if isinstance(shap_values, list):
#         return shap_values[1]        # old SHAP < 0.42
#     else:
#         return shap_values[:, :, 1]  # new SHAP >= 0.42 → 3D array

# def get_shap_base_value(explainer):
#     ev = explainer.expected_value
#     if isinstance(ev, (list, np.ndarray)):
#         return float(ev[1])          # class 1 base value
#     return float(ev)


# # Load model and encoders
# @st.cache_resource
# def load_model_artifacts():
#     try:
#         model        = joblib.load('models/rf_churn_model.pkl')
#         explainer    = joblib.load('models/shap_explainer.pkl')
#         encoders     = joblib.load('models/encoders.pkl')
#         feature_names = joblib.load('models/feature_names.pkl')

#         st.success(f"✅ Model loaded successfully with {len(feature_names)} features")
#         return model, encoders, feature_names, explainer

#     except FileNotFoundError as e:
#         st.error(f"❌ File not found: {str(e)}")
#         st.info("💡 Make sure all .pkl files are in the models/ directory")
#         return None, None, None, None

#     except ModuleNotFoundError as e:
#         st.error(f"❌ Missing module: {str(e)}")
#         st.info("💡 Install: pip install scikit-learn joblib shap")
#         return None, None, None, None

#     except Exception as e:
#         st.error(f"❌ Error loading model: {str(e)}")
#         import traceback
#         st.code(traceback.format_exc())
#         return None, None, None, None


# # Configure Gemini
# def configure_gemini():
#     api_key = os.getenv("GEMINI_API_KEY")
#     if api_key:
#         genai.configure(api_key=api_key)
#         return True
#     return False


# # Generate retention strategy
# def generate_retention_strategy(customer_data, churn_prob, top_factors):
#     if not configure_gemini():
#         return "⚠️ Gemini API key not configured. Please add it to your .env file."

#     try:
#         model = genai.GenerativeModel('gemini-2.5-flash-lite')

#         prompt = f"""
#         You are an E-commerce retention strategist. A customer has a {churn_prob:.1%} churn probability.

#         Customer Profile:
#         - Tenure: {customer_data.get('Tenure', 'N/A')} months
#         - Cashback: ${customer_data.get('CashbackAmount', 'N/A')}
#         - Complaints: {customer_data.get('Complain', 'N/A')}
#         - Days Since Last Order: {customer_data.get('DaySinceLastOrder', 'N/A')}
#         - Preferred Category: {customer_data.get('PreferredOrderCat', 'N/A')}

#         Top Risk Factors: {', '.join(top_factors)}

#         Provide specific, actionable retention strategies in bullet points. Be concise and give briefly.

#         Include:
#         - 3-5 immediate actions (with specific offers/discounts)
#         - Why each strategy targets the risk factors
#         - Expected outcomes

#         Keep it under 150 words total.
#         """

#         response = model.generate_content(prompt)
#         return response.text

#     except Exception as e:
#         return f"⚠️ Error generating strategy: {str(e)}"


# # Main prediction page
# def show_prediction_page():
#     st.title("🔮 Customer Churn Prediction")
#     st.markdown("### Predict churn risk and get AI-powered retention strategies")
#     st.markdown("---")

#     model, encoders, feature_names, explainer = load_model_artifacts()

#     if model is None:
#         st.error("❌ Cannot proceed without model artifacts. Please check the models/ directory.")
#         return

#     # Input Form
#     st.markdown("## 📝 Enter Customer Details")

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
#         preferred_login_device = st.selectbox("Preferred Login Device", ["Mobile Phone", "Computer", "Tablet"])
#         city_tier = st.selectbox("City Tier", [1, 2, 3])
#         warehouse_to_home = st.number_input("Warehouse to Home (km)", min_value=0, max_value=200, value=15)

#     with col2:
#         preferred_payment_mode = st.selectbox(
#             "Payment Mode",
#             ["Debit Card", "Credit Card", "E wallet", "UPI", "Cash on Delivery", "COD"]
#         )
#         gender = st.selectbox("Gender", ["Male", "Female"])
#         hours_on_app = st.number_input("Hours on App", min_value=0, max_value=24, value=3)
#         device_registered = st.number_input("Number of Devices", min_value=1, max_value=10, value=2)

#     with col3:
#         preferred_order_cat = st.selectbox(
#             "Preferred Category",
#             ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"]
#         )
#         satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)
#         marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
#         address_count = st.number_input("Number of Addresses", min_value=1, max_value=20, value=2)

#     col4, col5, col6 = st.columns(3)

#     with col4:
#         complain = st.selectbox("Has Complained?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
#         order_count = st.number_input("Order Count", min_value=0, max_value=50, value=5)

#     with col5:
#         day_since_last_order = st.number_input("Days Since Last Order", min_value=0, max_value=50, value=7)
#         cashback_amount = st.number_input("Cashback Amount ($)", min_value=0.0, max_value=500.0, value=100.0)

#     with col6:
#         order_amount_hike = st.number_input("Order Amount Hike (%)", min_value=0, max_value=50, value=15)
#         coupon_used = st.number_input("Coupons Used", min_value=0, max_value=20, value=3)

#     st.markdown("---")

#     if st.button("🔮 Predict Churn Risk", use_container_width=True):

#         input_data = {
#             'Tenure': tenure,
#             'PreferredLoginDevice': preferred_login_device,
#             'CityTier': city_tier,
#             'WarehouseToHome': warehouse_to_home,
#             'PreferredPaymentMode': preferred_payment_mode,
#             'Gender': gender,
#             'HourSpendOnApp': hours_on_app,
#             'NumberOfDeviceRegistered': device_registered,
#             'PreferedOrderCat': preferred_order_cat,
#             'SatisfactionScore': satisfaction_score,
#             'MaritalStatus': marital_status,
#             'NumberOfAddress': address_count,
#             'Complain': complain,
#             'OrderAmountHikeFromlastYear': order_amount_hike,
#             'CouponUsed': coupon_used,
#             'OrderCount': order_count,
#             'DaySinceLastOrder': day_since_last_order,
#             'CashbackAmount': cashback_amount
#         }

#         df_input = pd.DataFrame([input_data])

#         for col, encoder in encoders.items():
#             if col in df_input.columns:
#                 df_input[col] = encoder.transform(df_input[col])

#         df_input = df_input[feature_names]

#         churn_prob = model.predict_proba(df_input)[0][1]
#         churn_label = "HIGH RISK" if churn_prob > 0.5 else "LOW RISK"

#         # Results
#         st.markdown("---")
#         st.markdown("## 🎯 Prediction Results")

#         result_col1, result_col2, result_col3 = st.columns([1, 2, 1])

#         with result_col2:
#             if churn_prob > 0.7:
#                 st.error(f"### ⚠️ {churn_label}")
#                 st.metric("Churn Probability", f"{churn_prob:.1%}", delta=f"+{(churn_prob-0.5)*100:.0f}% above baseline")
#             elif churn_prob > 0.5:
#                 st.warning(f"### ⚠️ {churn_label}")
#                 st.metric("Churn Probability", f"{churn_prob:.1%}", delta=f"+{(churn_prob-0.5)*100:.0f}% above baseline")
#             else:
#                 st.success(f"### ✅ {churn_label}")
#                 st.metric("Churn Probability", f"{churn_prob:.1%}", delta=f"{(churn_prob-0.5)*100:.0f}% below baseline", delta_color="inverse")

#         st.markdown("---")

#         # ✅ Calculate SHAP values ONCE — reuse across all tabs
#         st.markdown("## 🧠 Explainable AI (SHAP Analysis)")

#         with st.spinner("Calculating SHAP values..."):
#             shap_values = explainer.shap_values(df_input)
#             sv_churn    = get_shap_churn_values(shap_values)   # ✅ shape: (1, n_features)
#             base_val    = get_shap_base_value(explainer)        # ✅ single float

#         tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "🎯 Force Plot", "💧 Waterfall Chart"])

#         with tab1:
#             st.markdown("### Global Feature Importance")
#             try:
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 shap.summary_plot(sv_churn, df_input, plot_type="bar", show=False)
#                 plt.title("Feature Importance (Mean |SHAP|)", fontsize=14, fontweight='bold')
#                 st.pyplot(fig)
#                 plt.close('all')
#                 st.info("**How to read:** Features at the top have the strongest influence on churn predictions.")
#             except Exception as e:
#                 st.error(f"Error: {str(e)}")

#         with tab2:
#             st.markdown("### Individual Prediction Breakdown")
#             try:
#                 shap.force_plot(
#                     base_val,
#                     sv_churn[0],          # ✅ first (only) sample
#                     df_input.iloc[0],
#                     matplotlib=True,
#                     show=False
#                 )
#                 fig = plt.gcf()
#                 fig.set_size_inches(16, 4)
#                 fig.tight_layout()
#                 st.pyplot(fig)
#                 plt.close('all')
#                 st.info("🔴 **Red arrows** = Increase churn risk | 🔵 **Blue arrows** = Decrease churn risk")
#             except Exception as e:
#                 st.warning("⚠️ Force plot rendering issue.")
#                 st.code(str(e))

#         with tab3:
#             st.markdown("### Waterfall Analysis")
#             try:
#                 explanation = shap.Explanation(
#                     values=sv_churn[0],            # ✅ 1D array of feature contributions
#                     base_values=base_val,           # ✅ single float
#                     data=df_input.iloc[0].values,
#                     feature_names=feature_names
#                 )
#                 fig, ax = plt.subplots(figsize=(12, 8))
#                 shap.waterfall_plot(explanation, show=False)
#                 plt.title("Individual Prediction Waterfall", fontsize=14, fontweight='bold')
#                 plt.tight_layout()
#                 st.pyplot(fig)
#                 plt.close('all')
#                 st.info("**Reads top to bottom:** Each feature adds or subtracts from the base prediction.")
#             except Exception as e:
#                 st.error(f"Error: {str(e)}")

#         # Top Risk Factors for Gemini
#         shap_importance = pd.DataFrame({
#             'feature': feature_names,
#             'importance': np.abs(sv_churn[0])    # ✅ 1D array
#         }).sort_values('importance', ascending=False).head(3)

#         top_factors = shap_importance['feature'].tolist()

#         # AI Strategy
#         st.markdown("---")
#         st.markdown("## 🦉 Smart Owl Recommendations")

#         with st.spinner("Gemini is analyzing the customer profile..."):
#             strategy = generate_retention_strategy(input_data, churn_prob, top_factors)
#             st.success(strategy)

#         # Download Report
#         st.markdown("---")
#         report = f"""
# SWOMII AI - Churn Prediction Report
# =====================================

# Customer Profile:
# - Tenure: {tenure} months
# - City Tier: {city_tier}
# - Satisfaction Score: {satisfaction_score}/5
# - Complaints: {'Yes' if complain == 1 else 'No'}
# - Days Since Last Order: {day_since_last_order}

# Prediction:
# - Churn Probability: {churn_prob:.2%}
# - Risk Level: {churn_label}

# Top Risk Factors:
# {chr(10).join([f"- {f}" for f in top_factors])}

# AI Recommendations:
# {strategy}

# Generated by SWOMII AI on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
#         """

#         st.download_button(
#             label="📥 Download Prediction Report",
#             data=report,
#             file_name=f"churn_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
#             mime="text/plain"
#         )


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
        return shap_values[1]
    else:
        return shap_values[:, :, 1]


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
        st.success(f"✅ Model loaded successfully with {len(feature_names)} features")
        return model, encoders, feature_names, explainer

    except FileNotFoundError as e:
        st.error(f"❌ File not found: {str(e)}")
        st.info("💡 Make sure all .pkl files are in the models/ directory")
        return None, None, None, None

    except ModuleNotFoundError as e:
        st.error(f"❌ Missing module: {str(e)}")
        st.info("💡 Install: pip install scikit-learn joblib shap")
        return None, None, None, None

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
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
        return "⚠️ Gemini API key not configured. Please add it to your .env file."

    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')

        prompt = f"""
        You are an E-commerce retention strategist. A customer has a {churn_prob:.1%} churn probability.

        Customer Profile:
        - Tenure: {customer_data.get('Tenure', 'N/A')} months
        - Cashback: ${customer_data.get('CashbackAmount', 'N/A')}
        - Complaints: {customer_data.get('Complain', 'N/A')}
        - Days Since Last Order: {customer_data.get('DaySinceLastOrder', 'N/A')}
        - Preferred Category: {customer_data.get('PreferredOrderCat', 'N/A')}

        Top Risk Factors: {', '.join(top_factors)}

        Provide specific, actionable retention strategies in bullet points. Be concise.

        Include:
        - 3-5 immediate actions (with specific offers/discounts)
        - Why each strategy targets the risk factors
        - Expected outcomes

        Keep it under 150 words total.
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"⚠️ Error generating strategy: {str(e)}"


# ─── MAIN PAGE ───────────────────────────────────────────────────
def show_prediction_page():
    st.title("🔮 Customer Churn Prediction")
    st.markdown("### Predict churn risk and get AI-powered retention strategies")
    st.markdown("---")

    model, encoders, feature_names, explainer = load_model_artifacts()

    if model is None:
        st.error("❌ Cannot proceed without model artifacts.")
        return

    # ── Input Form ───────────────────────────────────────────────
    st.markdown("## 📝 Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure                 = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        preferred_login_device = st.selectbox("Preferred Login Device", ["Mobile Phone", "Computer", "Tablet"])
        city_tier              = st.selectbox("City Tier", [1, 2, 3])
        warehouse_to_home      = st.number_input("Warehouse to Home (km)", min_value=0, max_value=200, value=15)

    with col2:
        preferred_payment_mode = st.selectbox(
            "Payment Mode",
            ["Debit Card", "Credit Card", "E wallet", "UPI", "Cash on Delivery", "COD"]
        )
        gender            = st.selectbox("Gender", ["Male", "Female"])
        hours_on_app      = st.number_input("Hours on App", min_value=0, max_value=24, value=3)
        device_registered = st.number_input("Number of Devices", min_value=1, max_value=10, value=2)

    with col3:
        preferred_order_cat = st.selectbox(
            "Preferred Category",
            ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"]
        )
        satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)
        marital_status     = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        address_count      = st.number_input("Number of Addresses", min_value=1, max_value=20, value=2)

    col4, col5, col6 = st.columns(3)

    with col4:
        complain    = st.selectbox("Has Complained?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        order_count = st.number_input("Order Count", min_value=0, max_value=50, value=5)

    with col5:
        day_since_last_order = st.number_input("Days Since Last Order", min_value=0, max_value=50, value=7)
        cashback_amount      = st.number_input("Cashback Amount ($)", min_value=0.0, max_value=500.0, value=100.0)

    with col6:
        order_amount_hike = st.number_input("Order Amount Hike (%)", min_value=0, max_value=50, value=15)
        coupon_used       = st.number_input("Coupons Used", min_value=0, max_value=20, value=3)

    st.markdown("---")

    if st.button("🔮 Predict Churn Risk", use_container_width=True):

        input_data = {
            'Tenure':                      tenure,
            'PreferredLoginDevice':        preferred_login_device,
            'CityTier':                    city_tier,
            'WarehouseToHome':             warehouse_to_home,
            'PreferredPaymentMode':        preferred_payment_mode,
            'Gender':                      gender,
            'HourSpendOnApp':              hours_on_app,
            'NumberOfDeviceRegistered':    device_registered,
            'PreferedOrderCat':            preferred_order_cat,
            'SatisfactionScore':           satisfaction_score,
            'MaritalStatus':               marital_status,
            'NumberOfAddress':             address_count,
            'Complain':                    complain,
            'OrderAmountHikeFromlastYear': order_amount_hike,
            'CouponUsed':                  coupon_used,
            'OrderCount':                  order_count,
            'DaySinceLastOrder':           day_since_last_order,
            'CashbackAmount':              cashback_amount
        }

        df_input = pd.DataFrame([input_data])

        for col, encoder in encoders.items():
            if col in df_input.columns:
                df_input[col] = encoder.transform(df_input[col])

        df_input   = df_input[feature_names]
        churn_prob = model.predict_proba(df_input)[0][1]
        churn_label = "HIGH RISK" if churn_prob > 0.5 else "LOW RISK"

        # ── Results ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")

        _, result_col2, _ = st.columns([1, 2, 1])

        with result_col2:
            if churn_prob > 0.7:
                st.error(f"### ⚠️ {churn_label}")
                st.metric("Churn Probability", f"{churn_prob:.1%}",
                          delta=f"+{(churn_prob-0.5)*100:.0f}% above baseline")
            elif churn_prob > 0.5:
                st.warning(f"### ⚠️ {churn_label}")
                st.metric("Churn Probability", f"{churn_prob:.1%}",
                          delta=f"+{(churn_prob-0.5)*100:.0f}% above baseline")
            else:
                st.success(f"### ✅ {churn_label}")
                st.metric("Churn Probability", f"{churn_prob:.1%}",
                          delta=f"{(churn_prob-0.5)*100:.0f}% below baseline",
                          delta_color="inverse")

        st.markdown("---")

        # ── SHAP — calculate ONCE ─────────────────────────────────
        st.markdown("## 🧠 Explainable AI (SHAP Analysis)")

        with st.spinner("Calculating SHAP values..."):
            shap_values = explainer.shap_values(df_input)
            sv_churn    = get_shap_churn_values(shap_values)
            base_val    = get_shap_base_value(explainer)

        tab1, tab2, tab3 = st.tabs([
            "📊 Feature Importance",
            "🎯 Force Plot",
            "💧 Waterfall Chart"
        ])

        # ── TAB 1: Feature Importance ─────────────────────────────
        with tab1:
            st.markdown("### Feature Importance (SHAP)")
            try:
                fig, ax = plt.subplots(figsize=(10, 6))

                shap.summary_plot(
                    sv_churn, df_input,
                    plot_type="bar",
                    show=False,
                    color=ACCENT        # ✅ indigo bars
                )

                # ✅ Force dark theme on the axes SHAP just drew
                ax = plt.gca()
                apply_dark_axes(fig, ax)

                # ✅ Force all bar text & tick labels to be visible
                ax.tick_params(axis='y', labelcolor=TEXT_PRIMARY, labelsize=10)
                ax.tick_params(axis='x', labelcolor=TEXT_MUTED,   labelsize=9)

                # ✅ Make all y-axis text (feature names) visible
                for label in ax.get_yticklabels():
                    label.set_color(TEXT_PRIMARY)
                    label.set_fontsize(10)

                ax.set_xlabel("Mean |SHAP Value|", color=TEXT_MUTED, fontsize=10)
                plt.title("Feature Importance (Mean |SHAP|)",
                          fontsize=13, fontweight='bold',
                          color=TEXT_PRIMARY, pad=14)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close('all')
                st.info("**How to read:** Features at the top have the strongest influence on churn predictions.")

            except Exception as e:
                st.error(f"Error: {str(e)}")

        # ── TAB 2: Force Plot ─────────────────────────────────────
        with tab2:
            st.markdown("### Individual Prediction Breakdown")
            try:
                fig_force, ax_force = plt.subplots(figsize=(16, 3))
                plt.close(fig_force)   # close empty fig first

                shap.force_plot(
                    base_val,
                    sv_churn[0],
                    df_input.iloc[0],
                    matplotlib=True,
                    show=False,
                    text_rotation=15
                )

                fig = plt.gcf()
                fig.set_size_inches(16, 4)
                fig.patch.set_facecolor(BG_DARK)

                for ax in fig.get_axes():
                    ax.set_facecolor(BG_DARK)
                    for text in ax.texts:
                        text.set_color(TEXT_PRIMARY)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close('all')
                st.info("🔴 **Red** = Increases churn risk &nbsp;|&nbsp; 🔵 **Blue** = Decreases churn risk")

            except Exception as e:
                st.warning("⚠️ Force plot rendering issue.")
                st.code(str(e))

        # ── TAB 3: Waterfall ──────────────────────────────────────
        with tab3:
            st.markdown("### Waterfall Analysis")
            try:
                explanation = shap.Explanation(
                    values=sv_churn[0],
                    base_values=base_val,
                    data=df_input.iloc[0].values,
                    feature_names=feature_names
                )

                fig, ax = plt.subplots(figsize=(12, 8))
                shap.waterfall_plot(explanation, show=False)

                # ✅ Apply dark theme after SHAP draws
                ax = plt.gca()
                apply_dark_axes(fig, ax)

                for label in ax.get_yticklabels():
                    label.set_color(TEXT_PRIMARY)
                    label.set_fontsize(10)
                for label in ax.get_xticklabels():
                    label.set_color(TEXT_MUTED)

                plt.title("Individual Prediction Waterfall",
                          fontsize=13, fontweight='bold',
                          color=TEXT_PRIMARY, pad=14)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close('all')
                st.info("**Reads top to bottom:** Each feature adds or subtracts from the base prediction.")

            except Exception as e:
                st.error(f"Error: {str(e)}")

        # ── Top Risk Factors ──────────────────────────────────────
        shap_importance = pd.DataFrame({
            'feature':    feature_names,
            'importance': np.abs(sv_churn[0])
        }).sort_values('importance', ascending=False).head(3)

        top_factors = shap_importance['feature'].tolist()

        # ── Gemini AI Strategy ────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🦉 Smart Owl Recommendations")

        with st.spinner("Gemini is analyzing the customer profile..."):
            strategy = generate_retention_strategy(input_data, churn_prob, top_factors)
            st.success(strategy)

        # ── Download Report ───────────────────────────────────────
        st.markdown("---")
        report = f"""
SWOMII AI - Churn Prediction Report
=====================================

Customer Profile:
- Tenure: {tenure} months
- City Tier: {city_tier}
- Satisfaction Score: {satisfaction_score}/5
- Complaints: {'Yes' if complain == 1 else 'No'}
- Days Since Last Order: {day_since_last_order}

Prediction:
- Churn Probability: {churn_prob:.2%}
- Risk Level: {churn_label}

Top Risk Factors:
{chr(10).join([f"- {f}" for f in top_factors])}

AI Recommendations:
{strategy}

Generated by SWOMII AI on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        st.download_button(
            label="📥 Download Prediction Report",
            data=report,
            file_name=f"churn_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )