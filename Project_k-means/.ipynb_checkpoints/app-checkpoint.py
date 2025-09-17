import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

# ----------------------------
# Title
# ----------------------------
st.title("📊 Customer Segmentation Dashboard")
st.write("Interactive dashboard for customer segmentation using **KMeans Clustering**.")

# ----------------------------
# Load Pre-trained Pipeline (optional)
# ----------------------------
pipeline = None
try:
    pipeline = joblib.load("customer_segmentation_model.pkl")
    st.success("✅ Pre-trained pipeline loaded successfully!")
except:
    st.warning("⚠️ No pre-trained pipeline found. You can still try clustering with the slider below.")

# ----------------------------
# Upload dataset
# ----------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(data.head())

    # Select numeric columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    X = data[numeric_cols]

    # ----------------------------
    # Sidebar Controls
    # ----------------------------
    st.sidebar.header("Clustering Options")
    n_clusters = st.sidebar.slider("Select number of clusters (k)", 2, 10, 4)

    # ----------------------------
    # Apply clustering
    # ----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # If pipeline exists and matches k → use it, else retrain
    if pipeline and hasattr(pipeline.named_steps['kmeans'], 'n_clusters') \
       and pipeline.named_steps['kmeans'].n_clusters == n_clusters:
        labels = pipeline.predict(X)
        st.info(f"Using pre-trained pipeline with k={n_clusters}")
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        st.info(f"Trained new KMeans model with k={n_clusters}")

    data["Cluster"] = labels

    # ----------------------------
    # PCA for visualization
    # ----------------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7,5))
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="tab10")
    plt.title(f"Customer Segments (k={n_clusters}, PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    st.pyplot(fig)

    # ----------------------------
    # Cluster Profiles
    # ----------------------------
    st.subheader("📈 Cluster Profiles (Mean values)")
    cluster_profiles = data.groupby("Cluster")[numeric_cols].mean()
    st.write(cluster_profiles)

    # ----------------------------
    # Download Results
    # ----------------------------
    st.subheader("💾 Download Segmented Data")
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV with Cluster Labels", csv, "segmented_customers.csv", "text/csv")
