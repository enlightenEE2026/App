import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Configure the Streamlit page layout layout wide
st.set_page_config(page_title="Student Profiling Dashboard", layout="wide")

# =====================================================================
# PHASE 1: COMPREHENSIVE PIPELINE LOGIC (Cached to prevent re-training)
# =====================================================================
@st.cache_resource
def run_training_pipeline(csv_file_path='your_data.csv'):
    # Generate mock file automatically if it doesn't exist
    if not os.path.exists(csv_file_path):
        np.random.seed(42)
        mock_data = pd.DataFrame({
            'Hours_Studied': np.random.uniform(10, 30, 200),
            'Attendance': np.random.uniform(60, 95, 200),
            'Sleep_Hours': np.random.uniform(5, 9, 200),
            'Previous_Scores': np.random.uniform(60, 95, 200),
            'Tutoring_Sessions': np.random.randint(0, 5, 200),
            'Physical_Activity': np.random.uniform(1, 5, 200),
            'Exam_Score': np.random.uniform(50, 95, 200),
            'Study_Physical_Interaction': np.random.uniform(30, 100, 200),
            'Study_Extracurricular': np.random.uniform(5, 25, 200),
            'Gender_Cat': ['Male', 'Female'] * 100  # Automatically handled/dropped
        })
        mock_data.to_csv(csv_file_path, index=False)

    # 1. Load Data
    df = pd.read_csv(csv_file_path)

    # 2. Refined Preprocessing (Exclude categoricals, force strict numericals, drop NaN rows)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    df_numeric_only = df.drop(columns=categorical_cols)
    df_numeric_coerced = df_numeric_only.apply(pd.to_numeric, errors='coerce')
    df_numeric_clean = df_numeric_coerced.dropna().copy()
    feature_names = df_numeric_clean.columns.tolist()

    # 3. Z-Score Normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric_clean)

    # 4. K-Means Training (Optimized to your analyzed profiles)
    optimal_k = 3 
    kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(X_scaled)

    # 5. Core Mapping
    cluster_names = {
        0: "Struggling / Low Engagement",
        1: "High Effort / Balanced Achievers",
        2: "Consistent Attenders / High Achievers"
    }
    df_numeric_clean['Cluster_Labels'] = cluster_labels
    df_numeric_clean['Student_Profile'] = df_numeric_clean['Cluster_Labels'].map(cluster_names)

    # 6. Evaluation metrics
    sil_score = silhouette_score(X_scaled, cluster_labels)
    
    # 7. Model Asset Serialized Export Bundle
    pipeline_bundle = {
        'scaler': scaler,
        'model': kmeans_model,
        'features': feature_names,
        'labels': cluster_names,
        'X_scaled': X_scaled,
        'cluster_labels': cluster_labels,
        'df_final': df_numeric_clean
    }
    joblib.dump(pipeline_bundle, 'student_kmeans_pipeline.pkl')
    return pipeline_bundle

# Initialize dataset assets using caching pipeline wrapper
assets = run_training_pipeline()

# =====================================================================
# PHASE 2: STREAMLIT USER INTERFACE & LAYOUT
# =====================================================================
st.title("🎓 Student Behavioral Segmentation Dashboard")
st.markdown("This web dashboard analyzes student metrics, defines profile segments, and predicts behavioral categories.")

# Core layout partitions 
col_metrics, col_viz = st.columns([1, 1.2])

# Left Dashboard Panel: Live Interactive Inference
with col_metrics:
    st.header("🔮 Real-Time Profiling Input")
    st.write("Adjust parameters to dynamically segment a test student case:")
    
    input_values = []
    # Loop over original training features to auto-build UI interactive input widgets
    for feature in assets['features']:
        # Fetch clean base constraints to match sliders dynamically to min/max distribution
        min_val = float(assets['df_final'][feature].min())
        max_val = float(assets['df_final'][feature].max())
        mean_val = float(assets['df_final'][feature].mean())
        
        # Select box layout for integer targets, numerical slider maps for fractions
        if feature in ['Tutoring_Sessions']:
            val = st.number_input(f"{feature}", min_value=int(min_val), max_value=int(max_val), value=int(mean_val))
        else:
            val = st.slider(f"{feature}", min_value=min_val, max_value=max_val, value=mean_val)
        input_values.append(val)
        
    # Process inputs through inference logic engine
    input_df = pd.DataFrame([input_values], columns=assets['features'])
    input_scaled = assets['scaler'].transform(input_df)
    pred_cluster = assets['model'].predict(input_scaled)[0]
    profile_string = assets['labels'][pred_cluster]
    
    # Render Output Segment Alerts
    st.subheader("Analysis Prediction Output")
    st.success(f"**Assigned Profile:** {profile_string} (Cluster ID: {pred_cluster})")

# Right Dashboard Panel: Dynamic Analytics Visualizations
with col_viz:
    st.header("📊 Cluster Pipeline Verification")
    
    # 1. Performance KPI Cards row
    kpi_col1, kpi_col2 = st.columns(2)
    kpi_col1.metric("Silhouette Score (Clustering Quality)", f"{silhouette_score(assets['X_scaled'], assets['cluster_labels']):.4f}")
    kpi_col2.metric("Dataset Points Evaluated", f"{len(assets['df_final'])} Rows")
    
    # 2. PCA Clustering Geometry plot
    st.subheader("2D Cluster Mapping Space (PCA Method)")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(assets['X_scaled'])
    
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=assets['cluster_labels'], cmap='viridis', alpha=0.6, edgecolors='k', s=45)
    
    # Process entry point tracking vector dynamically onto layout footprint
    sample_scaled = assets['scaler'].transform(input_df)
    sample_pca = pca.transform(sample_scaled)
    ax.scatter(sample_pca[0, 0], sample_pca[0, 1], c='red', marker='*', s=350, label='Current Test Student', edgecolors='black')
    
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right')
    st.pyplot(fig)
    
    # 3. Structured Data Log Preview Output 
    st.subheader("Data Log Sample Preview")
    st.dataframe(assets['df_final'].head(5), use_container_width=True)
