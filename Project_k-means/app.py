import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
from io import BytesIO

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
    # Tabs
    # ----------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Data & Clustering", "📈 Profiles & Charts", "📝 Insights", "💾 Download"]
    )

    # ----------------------------
    # Tab 1: Data & Clustering
    # ----------------------------
    with tab1:
        st.subheader("🔵 PCA Scatter Plot of Clusters")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(7,5))
        ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="tab10")
        plt.title(f"Customer Segments (k={n_clusters}, PCA projection)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        st.pyplot(fig)

        # ----------------------------
        # Cluster Size Summary
        # ----------------------------
        st.subheader("👥 Cluster Size Summary")
        cluster_counts = data["Cluster"].value_counts().sort_index()
        cluster_percentages = (cluster_counts / len(data)) * 100

        cluster_summary = pd.DataFrame({
            "Cluster": cluster_counts.index,
            "Count": cluster_counts.values,
            "Percentage": cluster_percentages.round(2).values
        })

        st.write(cluster_summary)

        fig, ax = plt.subplots(figsize=(6,4))
        cluster_counts.plot(kind="bar", ax=ax, color="teal", edgecolor="black")
        ax.set_title("Number of Customers per Cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # ----------------------------
    # Tab 2: Profiles & Charts
    # ----------------------------
    with tab2:
        st.subheader("📈 Cluster Profiles (Mean values)")
        cluster_profiles = data.groupby("Cluster")[numeric_cols].mean()
        st.write(cluster_profiles)

        st.subheader("📊 Feature Comparison Across Clusters")
        feature_choice = st.selectbox("Select a feature to compare (Bar Chart):", numeric_cols)

        fig, ax = plt.subplots(figsize=(7,5))
        cluster_profiles[feature_choice].plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
        ax.set_title(f"Average {feature_choice} by Cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel(f"Mean {feature_choice}")
        st.pyplot(fig)

        st.subheader("📦 Distribution of Feature by Cluster (Boxplot)")
        feature_choice2 = st.selectbox("Select a feature for distribution (Boxplot):", numeric_cols, index=0)

        fig, ax = plt.subplots(figsize=(7,5))
        sns.boxplot(x="Cluster", y=feature_choice2, data=data, palette="Set2", ax=ax)
        ax.set_title(f"Distribution of {feature_choice2} across Clusters")
        st.pyplot(fig)

        # ----------------------------
        # Heatmap (Normalized by Feature)
        # ----------------------------
        st.subheader("🔥 Normalized Heatmap of Cluster Profiles")

        # Z-score normalization across clusters for each feature
        cluster_profiles_normalized = cluster_profiles.apply(
            lambda x: (x - x.mean()) / x.std(), axis=1
        )

        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(cluster_profiles_normalized.T, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
        plt.title("Normalized Feature Values per Cluster (Z-scores)")
        st.pyplot(fig)

        # ----------------------------
        # Correlation Heatmap Toggle
        # ----------------------------
        show_corr = st.checkbox("🔍 Show Correlation Heatmaps", value=False)

        if show_corr:
            # ----------------------------
            # Overall Correlation Heatmap
            # ----------------------------
            st.subheader("🌍 Overall Correlation Heatmap (All Customers)")

            overall_corr = X.corr()
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(overall_corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
            plt.title("Correlation Heatmap - All Customers")
            st.pyplot(fig)

            # Save to PNG for download
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                label="📥 Download Overall Correlation Heatmap (PNG)",
                data=buf.getvalue(),
                file_name="overall_correlation_heatmap.png",
                mime="image/png"
            )

            # ----------------------------
            # Correlation Heatmaps per Cluster
            # ----------------------------
            st.subheader("🔗 Correlation Heatmaps per Cluster")

            for cluster in sorted(data["Cluster"].unique()):
                st.markdown(f"**Cluster {cluster} Correlation Heatmap**")
                cluster_data = data[data["Cluster"] == cluster][numeric_cols]
                corr = cluster_data.corr()

                fig, ax = plt.subplots(figsize=(8,6))
                sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
                plt.title(f"Correlation Heatmap - Cluster {cluster}")
                st.pyplot(fig)

                # Save each cluster heatmap to PNG for download
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.download_button(
                    label=f"📥 Download Cluster {cluster} Heatmap (PNG)",
                    data=buf.getvalue(),
                    file_name=f"cluster_{cluster}_correlation_heatmap.png",
                    mime="image/png"
                )

    # ----------------------------
    # Tab 3: Business Insights
    # ----------------------------
    with tab3:
        st.subheader("📝 Business-Oriented Cluster Insights")

        def generate_business_insights(cluster_profiles):
            summaries = []
            overall_mean = cluster_profiles.mean()

            for cluster in cluster_profiles.index:
                desc = f"**Cluster {cluster}:** "
                comparisons = []

                for feature in cluster_profiles.columns:
                    value = cluster_profiles.loc[cluster, feature]
                    if value > overall_mean[feature] * 1.2:
                        comparisons.append(f"high {feature}")
                    elif value < overall_mean[feature] * 0.8:
                        comparisons.append(f"low {feature}")

                if comparisons:
                    desc += f"This group shows **{' and '.join(comparisons)}** compared to other customers. "
                else:
                    desc += "This group has **average behavior across most features**. "

                if "Income" in cluster_profiles.columns and "Spending" in cluster_profiles.columns:
                    income = cluster_profiles.loc[cluster, "Income"]
                    spend = cluster_profiles.loc[cluster, "Spending"]
                    if income > overall_mean["Income"] and spend < overall_mean["Spending"]:
                        desc += "💡 These are **wealthy but cautious spenders** (premium potential)."
                    elif income < overall_mean["Income"] and spend > overall_mean["Spending"]:
                        desc += "💡 These are **value-driven customers** with high spending despite lower income."
                    elif income > overall_mean["Income"] and spend > overall_mean["Spending"]:
                        desc += "💡 These are **high-value customers** (ideal target group)."
                    else:
                        desc += "💡 These are **budget-conscious customers**."

                summaries.append(desc)
            return summaries

        insights = generate_business_insights(cluster_profiles)
        for insight in insights:
            st.markdown(f"- {insight}")

    # ----------------------------
    # Tab 4: Download Results
    # ----------------------------
    with tab4:
        st.subheader("💾 Download Segmented Data")
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV with Cluster Labels", csv, "segmented_customers.csv", "text/csv")

        st.subheader("💾 Download Cluster Profiles")
        csv_profiles = cluster_profiles.to_csv().encode("utf-8")
        st.download_button("Download Cluster Profiles (Raw)", csv_profiles, "cluster_profiles.csv", "text/csv")

        csv_profiles_norm = cluster_profiles_normalized.to_csv().encode("utf-8")
        st.download_button("Download Cluster Profiles (Normalized)", csv_profiles_norm, "cluster_profiles_normalized.csv", "text/csv")
