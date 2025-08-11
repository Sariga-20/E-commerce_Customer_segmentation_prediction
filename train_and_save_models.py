import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

# --- 1. DATA LOADING AND CLEANING ---
print("Loading and cleaning data...")
# Load the dataset
df = pd.read_csv(r"C:\Users\Sariga\OneDrive\Documents\New_Deploy_Streamlit\data\e-commerce data.csv", encoding='ISO-8859-1')

# Drop rows with missing CustomerID
df.dropna(subset=['CustomerID'], inplace=True)
df['CustomerID'] = df['CustomerID'].astype(int)

# Remove cancelled transactions and invalid data
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[df['UnitPrice'] > 0]
df = df[df['Quantity'] > 0]

# Remove duplicates
df = df.drop_duplicates()

# Outlier Treatment
Q1_qty = df['Quantity'].quantile(0.25)
Q3_qty = df['Quantity'].quantile(0.75)
IQR_qty = Q3_qty - Q1_qty
df = df[~((df['Quantity'] < (Q1_qty - 1.5 * IQR_qty)) | (df['Quantity'] > (Q3_qty + 1.5 * IQR_qty)))]

Q1_price = df['UnitPrice'].quantile(0.25)
Q3_price = df['UnitPrice'].quantile(0.75)
IQR_price = Q3_price - Q1_price
df = df[~((df['UnitPrice'] < (Q1_price - 1.5 * IQR_price)) | (df['UnitPrice'] > (Q3_price + 1.5 * IQR_price)))]

# --- 2. RFM FEATURE ENGINEERING ---
print("Calculating RFM features...")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)

rfm_df = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})
rfm_df.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'}, inplace=True)

# --- 3. K-MEANS SEGMENTATION ---
print("Performing K-Means segmentation...")
# Log transform and scale data for clustering
rfm_log_df = rfm_df.copy()
rfm_log_df['Frequency'] = np.log1p(rfm_log_df['Frequency'])
rfm_log_df['Monetary'] = np.log1p(rfm_log_df['Monetary'])

scaler_cluster = StandardScaler()
rfm_scaled = scaler_cluster.fit_transform(rfm_log_df)

# Train K-Means model
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(rfm_scaled)
rfm_df['KMeans_Cluster'] = kmeans_labels

# Profile and name segments
segment_profile = rfm_df.groupby('KMeans_Cluster').agg({
    'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean'
}).sort_values(by='Monetary', ascending=False)

segment_names = ["Champions", "Loyal Customers", "Potential Loyalists", "At-Risk Spenders", "Lost Customers"]
segment_map = {row.Index: name for name, row in zip(segment_names, segment_profile.itertuples())}
rfm_df['Segment'] = rfm_df['KMeans_Cluster'].map(segment_map)

# --- 4. PREDICTIVE CLASSIFICATION (RANDOM FOREST) ---
print("Training Random Forest classifier...")
X = rfm_df[['Recency', 'Frequency', 'Monetary']]
y = rfm_df['Segment']

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Scale features for the classifier
scaler_pred = StandardScaler()
X_train_scaled = scaler_pred.fit_transform(X_train)

# Train Random Forest model
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_scaled, y_train_encoded)

# --- 5. SAVE MODELS AND OBJECTS ---
print("Saving models and objects...")
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

joblib.dump(scaler_pred, 'saved_models/scaler_pred.pkl')
joblib.dump(le, 'saved_models/label_encoder.pkl')
joblib.dump(rf_classifier, 'saved_models/rf_classifier.pkl')
rfm_df.to_csv('saved_models/rfm_with_segments.csv') # Save final RFM data for visualizations

print("Training complete. All necessary files are saved in the 'saved_models' directory.")