import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="E-commerce Customer Segmentation And Prediction",
    page_icon="🛍️",
    layout="wide"
)

# --- LOAD PRE-TRAINED MODELS AND DATA ---
@st.cache_data
def load_assets():
    """Loads all necessary pre-trained models and data."""
    try:
        scaler = joblib.load("saved_models/scaler_pred.pkl")
        encoder = joblib.load("saved_models/label_encoder.pkl")
        classifier = joblib.load("saved_models/rf_classifier.pkl")
        rfm_data = pd.read_csv("saved_models/rfm_with_segments.csv")

        return scaler, encoder, classifier, rfm_data
    except FileNotFoundError:
        st.error("Model files not found. Please run the `train_and_save_models.py` script first.")
        return None, None, None, None

scaler, encoder, classifier, rfm_data = load_assets()

# --- APP TITLE AND DESCRIPTION ---
st.title("🛍️ E-commerce Customer Segmentation & Prediction")
st.markdown("""
This application analyzes e-commerce customer data to create meaningful segments and predicts which segment a customer belongs to based on their purchasing behavior.
This project uses **RFM (Recency, Frequency, Monetary) analysis** for feature engineering, **K-Means clustering** for segmentation, and a **Random Forest model** for prediction.
""")

# --- TABS FOR DIFFERENT SECTIONS ---
if rfm_data is not None:
    tab1, tab2, tab3 = st.tabs(["📊 Customer Segmentation Analysis", "💡 Segment Prediction", "ℹ️ About the Segments"])

    # --- TAB 1: SEGMENTATION ANALYSIS ---
    with tab1:
        st.header("Customer Segmentation Overview")
        st.markdown("Customers have been segmented into 5 distinct groups using the K-Means clustering algorithm based on their RFM scores.")
        
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Segment Distribution")
            segment_counts = rfm_data['Segment'].value_counts()
            fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
            ax_pie.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', startangle=140, 
                       colors=sns.color_palette("viridis", len(segment_counts)))
            ax_pie.axis('equal')
            st.pyplot(fig_pie)

        with col2:
            st.subheader("Segment Profiles (Average RFM Values)")
            segment_profile = rfm_data.groupby('Segment').agg({
                'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean'
            }).sort_values(by='Monetary', ascending=False).reset_index()
            st.dataframe(segment_profile, use_container_width=True)

        st.subheader("Visualizing Segments with RFM values")
        # 3D Scatter Plot for RFM
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=rfm_data['Recency'],
            y=rfm_data['Frequency'],
            z=rfm_data['Monetary'],
            mode='markers',
            marker=dict(
                size=5,
                color=rfm_data['KMeans_Cluster'],
                colorscale='Viridis',
                opacity=0.8,
                showscale=True
            ),
            text=rfm_data['Segment'] # Hover text
        )])
        fig_3d.update_layout(
            title='3D Scatter Plot of Customer Segments',
            scene=dict(
                xaxis_title='Recency (Days)',
                yaxis_title='Frequency (Purchases)',
                zaxis_title='Monetary (Value)'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig_3d, use_container_width=True)


    # --- TAB 2: SEGMENT PREDICTION ---
    with tab2:
        st.header("Predict a Customer's Segment")
        st.markdown("Enter the Recency, Frequency, and Monetary values for a customer to predict their segment.")

        with st.form("prediction_form"):
            recency = st.number_input("Recency (days since last purchase)", min_value=1, step=1, value=50)
            frequency = st.number_input("Frequency (total number of purchases)", min_value=1, step=1, value=5)
            monetary = st.number_input("Monetary (total spending)", min_value=0.01, step=0.01, value=500.0)
            
            submitted = st.form_submit_button("Predict Segment")

            if submitted:
                # Create a dataframe from user input
                input_data = pd.DataFrame({
                    'Recency': [recency],
                    'Frequency': [frequency],
                    'Monetary': [monetary]
                })

                # Scale the input data using the pre-trained scaler
                scaled_input = scaler.transform(input_data)
                
                # Predict the segment
                prediction_encoded = classifier.predict(scaled_input)
                
                # Decode the prediction to the segment name
                prediction_segment = encoder.inverse_transform(prediction_encoded)
                
                st.success(f"Predicted Customer Segment: **{prediction_segment[0]}**")

        st.subheader("Model's Feature Importance")
        st.markdown("The chart below shows which features the Random Forest model found most important for prediction.")
        
        feature_importance = pd.DataFrame({
            'Feature': ['Recency', 'Frequency', 'Monetary'],
            'Importance': classifier.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        fig_importance, ax_importance = plt.subplots()
        sns.barplot(data=feature_importance, x='Importance', y='Feature', ax=ax_importance, palette='rocket')
        ax_importance.set_title('Feature Importance')
        st.pyplot(fig_importance)


    # --- TAB 3: ABOUT THE SEGMENTS ---
    with tab3:
        st.header("Detailed Segment Descriptions")
        st.markdown("""
        Here are the business-oriented descriptions for each customer segment identified by the model:

        - **Champions:**
          - **Who they are:** Your best and most valuable customers. They have purchased recently, buy often, and spend the most.
          - **RFM:** High Frequency, High Monetary, Low Recency.
          - **Strategy:** Reward them. Offer loyalty programs, exclusive access to new products, and build a strong relationship. They are your brand advocates.

        - **Loyal Customers:**
          - **Who they are:** Consistent buyers who spend a good amount. They are responsive to promotions and are the backbone of your business.
          - **RFM:** High Frequency, Good Monetary, Low Recency.
          - **Strategy:** Engage them with up-selling opportunities. Ask for reviews and leverage their loyalty.

        - **Potential Loyalists:**
          - **Who they are:** Recent customers with average frequency and spending. They have the potential to become Loyal Customers or even Champions.
          - **RFM:** Average Frequency, Average Monetary, Low Recency.
          - **Strategy:** Nurture them with personalized offers and build a relationship. Encourage them to purchase more frequently.

        - **At-Risk Spenders:**
          - **Who they are:** Customers who used to spend and purchase often but haven't been back in a while. They are at risk of churning.
          - **RFM:** High/Average Frequency & Monetary, High Recency.
          - **Strategy:** Win them back with personalized reactivation campaigns. Offer special discounts or remind them of products they've viewed.

        - **Lost Customers:**
          - **Who they are:** Customers with the lowest frequency and monetary value, who also haven't purchased in a very long time.
          - **RFM:** Low Frequency, Low Monetary, High Recency.
          - **Strategy:** While it might be difficult to re-engage them, a low-cost campaign could be attempted. Otherwise, focus efforts on more valuable segments.
        """)