import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_curve, auc, 
                             mean_squared_error, precision_score, 
                             recall_score, f1_score)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# --- APP CONFIG ---
st.set_page_config(page_title="Car Health Service Founder Dashboard", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        # Loading the dataset generated previously
        return pd.read_csv('dataset.csv')
    except:
        st.error("dataset.csv not found! Please ensure it's in the root folder.")
        return None

df = load_data()

if df is not None:
    st.sidebar.title("Founder's Command Center")
    nav = st.sidebar.radio("Analysis Stage", [
        "Overview & Descriptive", 
        "Diagnostic Analysis", 
        "Customer Segmentation (Clustering)", # New Segment
        "Predictive: Classification", 
        "Predictive: Association Rules", 
        "Predictive: Regression",
        "Prescriptive & New Lead Predictor"
    ])

    # 1. OVERVIEW & DESCRIPTIVE
    if nav == "Overview & Descriptive":
        st.title("📊 Market Snapshot (Hindsight)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Surveyed", len(df))
        c2.metric("Hot Leads", df['Switch_Intent'].sum())
        c3.metric("Avg Spend (INR)", f"₹{df['Annual_Spend'].mean():,.0f}")
        
        st.subheader("Income Distribution by City")
        fig = px.box(df, x="City", y="Income_Lakhs", color="City", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # 2. DIAGNOSTIC ANALYSIS
    elif nav == "Diagnostic Analysis":
        st.title("🔍 Diagnostic: Why Customers Switch?")
        st.write("Analyzing the relationship between digital literacy and service adoption.")
        fig = px.violin(df, x="Switch_Intent", y="Digital_Usage_Score", box=True, points="all",
                        title="Digital Usage Score vs Switch Intent", color="Switch_Intent")
        st.plotly_chart(fig)

    # 3. CUSTOMER SEGMENTATION (NEW MODULE)
    elif nav == "Customer Segmentation (Clustering)":
        st.title("👥 Predictive Customer Segmentation")
        st.markdown("This module groups customers into 'Tribes' based on Income, Car Age, and Digital Maturity.")
        
        # Select features for clustering
        features = ['Income_Lakhs', 'Car_Age', 'Digital_Usage_Score']
        X_clust = df[features]
        
        # Standardizing the data for better K-Means performance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clust)
        
        # User selection for clusters
        num_clusters = st.slider("Select number of segments to identify:", 2, 6, 3)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(X_scaled)
        df['Segment'] = kmeans.labels_
        
        # 3D Visual representation
        st.subheader("3D Segment Visualization")
        fig_seg = px.scatter_3d(df, x='Income_Lakhs', y='Car_Age', z='Digital_Usage_Score', 
                                color='Segment', title="AI-Driven Persona Clusters",
                                opacity=0.7, template="plotly_dark")
        st.plotly_chart(fig_seg, use_container_width=True)
        
        # Segment Profiling
        st.subheader("Segment Deep-Dive (Averages)")
        profile = df.groupby('Segment')[features + ['Annual_Spend', 'Switch_Intent']].mean()
        st.write("Use this table to define your marketing strategy for each group:")
        st.table(profile.style.background_gradient(cmap='Blues').format("{:.2f}"))

    # 4. PREDICTIVE: CLASSIFICATION
    elif nav == "Predictive: Classification":
        st.title("🎯 Propensity Modeling (Random Forest)")
        X = pd.get_dummies(df[['Income_Lakhs', 'Car_Age', 'Digital_Usage_Score', 'City']])
        y = df['Switch_Intent']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_probs = clf.predict_proba(X_test)[:, 1]

        st.subheader("Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{clf.score(X_test, y_test):.2%}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
        col3.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
        col4.metric("F1-Score", f"{f1_score(y_test, y_pred):.2%}")

        fpr, tpr, _ = roc_curve(y_test, y_probs)
        fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC: {auc(fpr, tpr):.2f})", 
                          labels={'x':'False Positive Rate', 'y':'True Positive Rate'})
        st.plotly_chart(fig_roc)

    # 5. PREDICTIVE: ASSOCIATION RULES
    elif nav == "Predictive: Association Rules":
        st.title("🛒 Product Association (Bundling Strategy)")
        # Simulating cross-category data for mining
        transactions = df[['Attire', 'Cookware', 'Hardware_Interest', 'Service_Interest']].values.tolist()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        basket = pd.DataFrame(te_ary, columns=te.columns_)
        
        freq_items = apriori(basket, min_support=0.05, use_colnames=True)
        rules = association_rules(freq_items, metric="lift", min_threshold=1.1)
        
        st.write("Top Product/Lifestyle Associations to drive sales:")
        st.dataframe(rules[['antecedents', 'consequents', 'confidence', 'lift']].sort_values('lift', ascending=False))

    # 6. PREDICTIVE: REGRESSION
    elif nav == "Predictive: Regression":
        st.title("💰 Revenue Forecasting")
        X_reg = pd.get_dummies(df[['Income_Lakhs', 'Car_Age', 'Digital_Usage_Score']])
        y_reg = df['Annual_Spend']
        reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_reg, y_reg)
        
        st.write(f"Model RMSE: ₹{np.sqrt(mean_squared_error(y_reg, reg.predict(X_reg))):,.2f}")
        fig = px.scatter(df, x="Annual_Spend", y=reg.predict(X_reg), 
                         title="Actual vs Predicted Spend", labels={'y':'Predicted Spend'})
        st.plotly_chart(fig)

    # 7. PRESCRIPTIVE & NEW LEAD PREDICTOR
    elif nav == "Prescriptive & New Lead Predictor":
        st.title("🚀 Prescriptive Strategy & Lead Prediction")
        st.success("PRESCRIPTION: Prioritize Segment with highest 'Switch_Intent' from the Clustering module.")
        
        st.divider()
        uploaded_file = st.file_uploader("Upload New Leads (CSV)", type="csv")
        if uploaded_file:
            new_leads = pd.read_csv(uploaded_file)
            # Example lead scoring logic
            new_leads['Interest_Score'] = np.random.uniform(0.1, 0.95, len(new_leads))
            new_leads['Prescribed_Action'] = np.where(new_leads['Interest_Score'] > 0.7, 'High Priority: Direct Call', 'Medium Priority: WhatsApp Nurture')
            st.write("Processed Leads Scoring:")
            st.dataframe(new_leads)
