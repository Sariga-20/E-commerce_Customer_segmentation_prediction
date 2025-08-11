# 🛍️ E-Commerce Customer Segmentation & Prediction

An interactive **Streamlit** web application that segments e‑commerce customers using **RFM (Recency, Frequency, Monetary)** analysis and predicts the customer segment based on purchase behaviour using a **Random Forest Classifier**.

The segmentation logic uses **K-Means clustering** on RFM scores to identify five target customer groups, helping businesses run personalized marketing, predict churn, and boost retention.

---

## 📌 Features

- **RFM Analysis**: Calculates Recency, Frequency, Monetary values from transaction data.
- **Customer Segmentation**: Clusters customers using K-Means into business-friendly segments.
- **Interactive Dashboard**:
  - Segment distribution pie charts
  - Average RFM profiles
  - 3D scatter visualization of segments
- **Segment Prediction**: Enter R, F, M values → predict customer segment.
- **Insights Tab**: Detailed practical business strategies for each segment.

---
## 📂 Project Structure

The repository is organized as follows:

```plaintext
├── app.py                  # Streamlit application
├── Project_Code.ipynb      # Notebook: data processing, RFM, ML model training
├── e-commerce-data.csv     # Dataset
├── requirements.txt        # Python dependencies
├── saved_models/           # Pre-trained scaler, encoder, classifier, processed RFM data
│   ├── scaler_pred.pkl
│   ├── label_encoder.pkl
│   ├── rf_classifier.pkl
│   └── rfm_with_segments.csv
└── README.md


---

## 🛠 Tech Stack

**Programming Language:** Python  
**Libraries:**  
- Data Handling: `pandas`, `numpy`  
- Visualization: `matplotlib`, `seaborn`, `plotly`  
- Machine Learning: `scikit-learn`, `joblib`  
- Web App: `streamlit`  

---

## 📊 Methodology


1. **Data Preparation**
   - Clean data & remove nulls.
   - Filter valid transactions.

2. **RFM Calculation**
   - Recency = Days since last purchase  
   - Frequency = Number of purchases  
   - Monetary = Total spend

3. **Clustering**
   - Scale RFM values  
   - Apply K-Means to form customer groups

4. **Prediction Model**
   - Train Random Forest Classifier  
   - Save models with `joblib` for app use

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository

git clone  https://github.com/Sariga-20/E-commerce_Customer_segmentation_prediction.git


### 2️⃣ Create a Virtual Environment 

python -m venv venv
venv\Scripts\activate # Windows
source venv/bin/activate # macOS/Linux

### 3️⃣ Install Dependencies

pip install -r requirements.txt


### 4️⃣ Run the App

streamlit run app.py

Visit [**http://localhost:8501**](http://localhost:8501) in your browser.

---

## 🌐 Deploy on Streamlit Cloud

1. Push this repository to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **Create app** → Connect your GitHub repo → Select `app.py` as the main file.
4. Click **Deploy** — your app gets a public URL to share.

---

## 📈 Customer Segments

- **Champions** 🏆 – Frequent, high spenders ⇒ reward loyalty.
- **Loyal Customers** 💎 – Consistent buyers ⇒ upsell & engage.
- **Potential Loyalists** 🌱 – New customers ⇒ nurture engagement.
- **At-Risk Customers** ⚠️ – Previously active ⇒ run reactivation offers.
- **Lost Customers** ❌ – Low spend/engagement ⇒ minimal recovery focus.

---

## 📄 Dataset
The dataset contains e‑commerce transactions with:
- **InvoiceNo** – Invoice number  
- **StockCode** – Product code  
- **Description** – Product name  
- **Quantity** – Units purchased  
- **InvoiceDate** – Date of transaction  
- **UnitPrice** – Price per unit  
- **CustomerID** – Customer identifier  
- **Country** – Customer country  

_Source_: [Online Retail Dataset from UCI Machine Learning Repository] (update if public)

---

## 🤝 Contributing

Pull requests and suggestions are welcome.  
Open an issue to discuss improvements.

---

## 📜 License

This project is licensed under the MIT License — see the LICENSE file for details.

