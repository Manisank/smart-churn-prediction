import streamlit as st
import pandas as pd
import numpy as np
import joblib  # ✅ Changed from pickle
import shap
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
from io import BytesIO

# Import scikit-learn components (required for unpickling)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Load model and encoders
@st.cache_resource
def load_model_artifacts():
    """Load the trained model, encoders, and feature names using joblib"""
    try:
        # ✅ Use joblib.load() to match joblib.dump() from Colab
        model = joblib.load('models/xgboost_churn_model.pkl')
        encoders = joblib.load('models/encoders.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        
        st.success(f"✅ Model loaded successfully with {len(feature_names)} features")
        return model, encoders, feature_names
        
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {str(e)}")
        st.info("💡 Make sure all .pkl files are in the models/ directory")
        return None, None, None
        
    except ModuleNotFoundError as e:
        st.error(f"❌ Missing module: {str(e)}")
        st.info("💡 Install missing packages: pip install scikit-learn xgboost joblib")
        return None, None, None
        
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {str(e)}")
        st.error(f"📍 Error type: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

# Rest of your code remains the same...




# Configure Gemini
def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        return True
    return False

# Generate retention strategy
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
        
    Provide specific, actionable retention strategies in bullet points. Be concise and give briefly.

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

# Main prediction page
def show_prediction_page():
    st.title("🔮 Customer Churn Prediction")
    st.markdown("### Predict churn risk and get AI-powered retention strategies")
    st.markdown("---")
    
    # Load model
    model, encoders, feature_names = load_model_artifacts()
    
    if model is None:
        st.error("❌ Cannot proceed without model artifacts. Please check the models/ directory.")
        return
    
    # Input Form
    st.markdown("## 📝 Enter Customer Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        preferred_login_device = st.selectbox("Preferred Login Device", ["Mobile Phone", "Computer", "Tablet"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        warehouse_to_home = st.number_input("Warehouse to Home (km)", min_value=0, max_value=200, value=15)
    
    with col2:
        preferred_payment_mode = st.selectbox(
            "Payment Mode",
            ["Debit Card", "Credit Card", "E wallet", "UPI", "Cash on Delivery", "COD"]
        )
        gender = st.selectbox("Gender", ["Male", "Female"])
        hours_on_app = st.number_input("Hours on App", min_value=0, max_value=24, value=3)
        device_registered = st.number_input("Number of Devices", min_value=1, max_value=10, value=2)
    
    with col3:
        preferred_order_cat = st.selectbox(
            "Preferred Category",
            ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"]
        )
        satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        address_count = st.number_input("Number of Addresses", min_value=1, max_value=20, value=2)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        complain = st.selectbox("Has Complained?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        order_count = st.number_input("Order Count", min_value=0, max_value=50, value=5)
    
    with col5:
        day_since_last_order = st.number_input("Days Since Last Order", min_value=0, max_value=50, value=7)
        cashback_amount = st.number_input("Cashback Amount ($)", min_value=0.0, max_value=500.0, value=100.0)
    
    with col6:
        order_amount_hike = st.number_input("Order Amount Hike (%)", min_value=0, max_value=50, value=15)
        coupon_used = st.number_input("Coupons Used", min_value=0, max_value=20, value=3)
    
    st.markdown("---")
    
    # Prediction Button
    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        # Prepare input data
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
            'OrderAmountHikeFromlastYear': order_amount_hike,
            'CouponUsed': coupon_used,
            'OrderCount': order_count,
            'DaySinceLastOrder': day_since_last_order,
            'CashbackAmount': cashback_amount
        }
        
        # Encode categorical variables
        df_input = pd.DataFrame([input_data])
        
        for col, encoder in encoders.items():
            if col in df_input.columns:
                df_input[col] = encoder.transform(df_input[col])
        
        # Ensure correct feature order
        df_input = df_input[feature_names]
        
        # Make prediction
        churn_prob = model.predict_proba(df_input)[0][1]
        churn_label = "HIGH RISK" if churn_prob > 0.5 else "LOW RISK"
        
        # Display Results
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        
        result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
        
        with result_col2:
            if churn_prob > 0.7:
                st.error(f"### ⚠️ {churn_label}")
                st.metric("Churn Probability", f"{churn_prob:.1%}", delta=f"+{(churn_prob-0.5)*100:.0f}% above baseline")
            elif churn_prob > 0.5:
                st.warning(f"### ⚠️ {churn_label}")
                st.metric("Churn Probability", f"{churn_prob:.1%}", delta=f"+{(churn_prob-0.5)*100:.0f}% above baseline")
            else:
                st.success(f"### ✅ {churn_label}")
                st.metric("Churn Probability", f"{churn_prob:.1%}", delta=f"{(churn_prob-0.5)*100:.0f}% below baseline", delta_color="inverse")
        
        st.markdown("---")
        
        # # SHAP Explanation
        # st.markdown("## 🧠 Explainable AI (SHAP Analysis)")
        
        # tab1, tab2 = st.tabs(["📊 Feature Importance", "🎯 Force Plot"])
        
        # with tab1:
        #     with st.spinner("Calculating SHAP values..."):
        #         explainer = shap.TreeExplainer(model)
        #         shap_values = explainer.shap_values(df_input)
                
        #         fig, ax = plt.subplots(figsize=(10, 6))
        #         shap.summary_plot(shap_values, df_input, plot_type="bar", show=False)
        #         st.pyplot(fig)
        #         plt.close()
                
        #         st.info("**How to read this chart:** Features at the top have the strongest influence on this prediction.")
        

        # with tab2:
        #     with st.spinner("Generating force plot..."):
        #         try:
        #             # Generate force plot and capture it properly
        #             shap.force_plot(
        #                 explainer.expected_value,
        #                 shap_values[0],
        #                 df_input.iloc[0],
        #                 matplotlib=True,
        #                 show=False
        #             )
                    
        #             # Get current figure from matplotlib
        #             fig = plt.gcf()
        #             fig.set_size_inches(16, 4)
        #             fig.tight_layout()
                    
        #             st.pyplot(fig)
        #             plt.close('all')
                    
        #         except Exception as e:
        #             st.warning(f"⚠️ Could not generate force plot: {str(e)}")
        #             st.info("📊 Showing waterfall plot instead...")
                    
        #             # Alternative: Use waterfall plot (more reliable)
        #             fig, ax = plt.subplots(figsize=(12, 6))
        #             shap.waterfall_plot(
        #                 shap.Explanation(
        #                     values=shap_values[0],
        #                     base_values=explainer.expected_value,
        #                     data=df_input.iloc[0].values,
        #                     feature_names=feature_names
        #                 ),
        #                 show=False
        #             )
        #             st.pyplot(fig)
        #             plt.close('all')

                
        #         st.info("**Red = increases churn risk | Blue = decreases churn risk**")

        # SHAP Explanation
        st.markdown("---")
        st.markdown("## 🧠 Explainable AI (SHAP Analysis)")

        tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "🎯 Force Plot", "💧 Waterfall Chart"])

        with tab1:
            st.markdown("### Global Feature Importance")
            with st.spinner("Calculating SHAP values..."):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(df_input)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, df_input, plot_type="bar", show=False)
                plt.title("Feature Importance (Mean |SHAP|)", fontsize=14, fontweight='bold')
                st.pyplot(fig)
                plt.close('all')
                
                st.info("**How to read:** Features at the top have the strongest influence on churn predictions.")

        with tab2:
            st.markdown("### Individual Prediction Breakdown")
            with st.spinner("Generating force plot..."):
                try:
                    # SHAP force plot
                    shap.force_plot(
                        explainer.expected_value,
                        shap_values[0],
                        df_input.iloc[0],
                        matplotlib=True,
                        show=False
                    )
                    
                    fig = plt.gcf()
                    fig.set_size_inches(16, 4)
                    fig.tight_layout()
                    
                    st.pyplot(fig)
                    plt.close('all')
                    
                    st.info("🔴 **Red arrows** = Increase churn risk | 🔵 **Blue arrows** = Decrease churn risk")
                    
                except Exception as e:
                    st.warning("⚠️ Force plot rendering issue. Showing alternative view...")
                    st.code(str(e))

        with tab3:
            st.markdown("### Waterfall Analysis (Alternative View)")
            with st.spinner("Creating waterfall chart..."):
                try:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    explanation = shap.Explanation(
                        values=shap_values[0],
                        base_values=explainer.expected_value,
                        data=df_input.iloc[0].values,
                        feature_names=feature_names
                    )
                    
                    shap.waterfall_plot(explanation, show=False)
                    plt.title("Individual Prediction Waterfall", fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    plt.close('all')
                    
                    st.info("**Reads top to bottom:** Each feature adds or subtracts from the base prediction.")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        
        # Get top risk factors
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values[0])
        }).sort_values('importance', ascending=False).head(3)
        
        top_factors = shap_importance['feature'].tolist()
        
        # AI Strategy
        st.markdown("---")
        st.markdown("## 🦉 Smart Owl Recommendations")
        
        with st.spinner("Gemini is analyzing the customer profile..."):
            strategy = generate_retention_strategy(input_data, churn_prob, top_factors)
            st.success(strategy)
        
        # Download Report
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
