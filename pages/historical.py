import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load historical data
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('Ecommerce_Dataset.xlsx')
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# Main historical page
def show_historical_page():
    st.title("📊 Historical Churn Analytics")
    st.markdown("### Discover patterns and trends in customer behavior")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("❌ Cannot load historical data. Please check the dataset file.")
        return
    
    # Overview Metrics
    st.markdown("## 📈 Key Metrics")
    
    churn_rate = df['Churn'].mean() * 100
    total_customers = len(df)
    churned_customers = df['Churn'].sum()
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Total Customers", f"{total_customers:,}")
    
    with metric_col2:
        st.metric("Churned Customers", f"{churned_customers:,}")
    
    with metric_col3:
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    with metric_col4:
        avg_tenure = df['Tenure'].mean()
        st.metric("Avg Tenure", f"{avg_tenure:.1f} months")
    
    st.markdown("---")
    
    # Filters
    st.markdown("## 🔍 Filter Data")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        selected_category = st.multiselect(
            "Preferred Order Category",
            options=df['PreferedOrderCat'].unique().tolist(),
            default=df['PreferedOrderCat'].unique().tolist()
        )
    
    with filter_col2:
        selected_city_tier = st.multiselect(
            "City Tier",
            options=sorted(df['CityTier'].unique().tolist()),
            default=sorted(df['CityTier'].unique().tolist())
        )
    
    with filter_col3:
        tenure_range = st.slider(
            "Tenure Range (months)",
            int(df['Tenure'].min()),
            int(df['Tenure'].max()),
            (int(df['Tenure'].min()), int(df['Tenure'].max()))
        )
    
    # Apply filters
    filtered_df = df[
        (df['PreferedOrderCat'].isin(selected_category)) &
        (df['CityTier'].isin(selected_city_tier)) &
        (df['Tenure'] >= tenure_range[0]) &
        (df['Tenure'] <= tenure_range[1])
    ]
    
    st.info(f"📌 Showing {len(filtered_df):,} customers after filtering")
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("## 📊 Visual Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔴 Churn by Category", "🏙️ Geographic Analysis", "⏰ Tenure Analysis", "💰 Financial Patterns"])
    
    with tab1:
        st.markdown("### Churn Rate by Preferred Order Category")
        
        churn_by_cat = filtered_df.groupby('PreferedOrderCat')['Churn'].agg(['sum', 'count', 'mean']).reset_index()
        churn_by_cat.columns = ['Category', 'Churned', 'Total', 'Churn_Rate']
        churn_by_cat['Churn_Rate'] = churn_by_cat['Churn_Rate'] * 100
        
        fig1 = px.bar(
            churn_by_cat,
            x='Category',
            y='Churn_Rate',
            text='Churn_Rate',
            title='Churn Rate by Category',
            color='Churn_Rate',
            color_continuous_scale='Reds'
        )
        fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig1.update_layout(height=500)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.dataframe(churn_by_cat.style.format({'Churn_Rate': '{:.2f}%'}), use_container_width=True)
    
    with tab2:
        st.markdown("### Churn Distribution by City Tier")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            churn_by_city = filtered_df.groupby(['CityTier', 'Churn']).size().reset_index(name='Count')
            
            fig2 = px.bar(
                churn_by_city,
                x='CityTier',
                y='Count',
                color='Churn',
                barmode='group',
                title='Churned vs Retained by City Tier',
                labels={'Churn': 'Status'},
                color_discrete_map={0: '#10b981', 1: '#ef4444'}
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col_right:
            churn_rate_city = filtered_df.groupby('CityTier')['Churn'].mean().reset_index()
            churn_rate_city['Churn'] = churn_rate_city['Churn'] * 100
            
            fig3 = px.pie(
                churn_rate_city,
                names='CityTier',
                values='Churn',
                title='Churn Rate Distribution by City Tier',
                hole=0.4
            )
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.markdown("### Churn vs Tenure Analysis")
        
        # Bin tenure
        filtered_df['Tenure_Bin'] = pd.cut(filtered_df['Tenure'], bins=[0, 6, 12, 24, 100], labels=['0-6m', '6-12m', '12-24m', '24m+'])
        
        tenure_churn = filtered_df.groupby('Tenure_Bin')['Churn'].mean().reset_index()
        tenure_churn['Churn'] = tenure_churn['Churn'] * 100
        
        fig4 = px.line(
            tenure_churn,
            x='Tenure_Bin',
            y='Churn',
            markers=True,
            title='Churn Rate by Tenure Segment',
            labels={'Churn': 'Churn Rate (%)', 'Tenure_Bin': 'Tenure Range'}
        )
        fig4.update_traces(line_color='#6366f1', line_width=3, marker=dict(size=10))
        fig4.update_layout(height=500)
        st.plotly_chart(fig4, use_container_width=True)
        
        st.info("💡 **Insight:** Newer customers (0-6 months) typically have the highest churn risk.")
    
    with tab4:
        st.markdown("### Financial Behavior Analysis")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            # Cashback analysis
            cashback_churn = filtered_df.groupby('Churn')['CashbackAmount'].mean().reset_index()
            cashback_churn['Churn'] = cashback_churn['Churn'].map({0: 'Retained', 1: 'Churned'})
            
            fig5 = px.bar(
                cashback_churn,
                x='Churn',
                y='CashbackAmount',
                text='CashbackAmount',
                title='Average Cashback by Churn Status',
                color='Churn',
                color_discrete_map={'Retained': '#10b981', 'Churned': '#ef4444'}
            )
            fig5.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
            fig5.update_layout(height=400)
            st.plotly_chart(fig5, use_container_width=True)
        
        with col_b:
            # Satisfaction vs Churn
            satisfaction_churn = filtered_df.groupby('SatisfactionScore')['Churn'].mean().reset_index()
            satisfaction_churn['Churn'] = satisfaction_churn['Churn'] * 100
            
            fig6 = px.line(
                satisfaction_churn,
                x='SatisfactionScore',
                y='Churn',
                markers=True,
                title='Churn Rate by Satisfaction Score',
                labels={'Churn': 'Churn Rate (%)'}
            )
            fig6.update_traces(line_color='#f59e0b', line_width=3, marker=dict(size=10))
            fig6.update_layout(height=400)
            st.plotly_chart(fig6, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation Heatmap
    st.markdown("## 🔥 Feature Correlation with Churn")
    
    numeric_cols = ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 
                    'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
                    'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 
                    'OrderCount', 'DaySinceLastOrder', 'CashbackAmount', 'Churn']
    
    corr_matrix = filtered_df[numeric_cols].corr()['Churn'].sort_values(ascending=False).drop('Churn')
    
    fig7 = go.Figure(go.Bar(
        x=corr_matrix.values,
        y=corr_matrix.index,
        orientation='h',
        marker=dict(
            color=corr_matrix.values,
            colorscale='RdYlGn_r',
            showscale=True
        )
    ))
    fig7.update_layout(
        title='Feature Correlation with Churn',
        xaxis_title='Correlation Coefficient',
        height=600
    )
    st.plotly_chart(fig7, use_container_width=True)
    
    st.markdown("---")
    
    # Raw Data
    with st.expander("📋 View Raw Filtered Data"):
        st.dataframe(filtered_df, use_container_width=True)
        
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name=f"filtered_churn_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
