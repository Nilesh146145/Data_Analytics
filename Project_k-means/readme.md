# 🛒 Customer Segmentation using K-Means Clustering

This project applies **K-Means clustering** to segment customers based on their purchasing behavior and demographics.  
It is built with **Python, Scikit-learn, and Streamlit**, and deployed on **Streamlit Cloud** for interactive use.  

---

## 📌 Project Overview
Customer segmentation helps businesses understand their customers better by grouping them into clusters.  
This allows targeted marketing, personalized recommendations, and improved customer satisfaction.  

In this project:
- We used the **iFood dataset** (customer data with spending, campaign response, demographics, etc.).
- Applied **data cleaning, preprocessing, scaling, and clustering**.
- Evaluated clusters using the **Elbow Method** and **Silhouette Score**.
- Built an **interactive Streamlit dashboard** to visualize and explore customer segments.

---

## 🔑 Key Features
- 🧹 **Data Preprocessing**: Handling missing values, scaling, feature selection.  
- 📊 **Clustering**: K-Means applied with optimal `k` selection.  
- 📉 **Dimensionality Reduction**: PCA for 2D visualization of clusters.  
- 📈 **Visualizations**:
  - Cluster scatter plots
  - Spending behavior comparisons
  - Heatmaps of cluster centers
- 🌐 **Streamlit App**:
  - Upload dataset
  - Choose number of clusters
  - View interactive cluster plots
  - See customer segment insights

---

## ⚙️ Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly)  
- **Streamlit** (frontend dashboard)  
- **GitHub + Streamlit Cloud** (deployment)  

---

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/data_analytics.git

Navigate to the project folder:

cd data_analytics/project\ K-means


Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py

🌍 Live Demo

👉 Click here to try the ap: https://dataanalytics-customersegment.streamlit.app/

📊 Business Insights

Example interpretations:

Cluster 1: High income, high spending → ideal for premium offers.

Cluster 2: Low income, low spending → focus on discounts/loyalty programs.

Cluster 3: Middle income, selective spending → targeted campaigns.

📂 Project Structure
project K-means/
│── app.py                   # Main Streamlit app
│── requirements.txt         # Dependencies
│── ifood_df.csv             # Sample dataset
│── customer_segmentation_model.pkl  # Trained model (optional)
│── notebooks/               # Jupyter notebooks for analysis
│   └── Customer_seg_updated.ipynb

👨‍💻 Author

Developed by: Swati Kumari, Pushpendra Singh, Nilesh Singh
📧 nileshs1595@gmail.com | 🔗 https://github.com/Nilesh146145/Data_Analytics/tree/main/Project_k-means