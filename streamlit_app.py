import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

# Page Layout Setup
st.set_page_config(
    page_title="Email Spam Classifier",
    layout="wide"
)
st.markdown("""
    <style>
    /* Expand default sidebar width */
    [data-testid="stSidebar"] {
        min-width: 380px !important;
        width: 380px !important;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Times New Roman', Times, serif !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
    }
    .sidebar-heading {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 1.25rem !important;
        font-weight: 900 !important;
        color: #FFFFFF !important;
        margin-top: 1rem;
        margin-bottom: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Email Spam Classification Dashboard")
st.markdown("5 classifiers. 6 evaluation metrics. Zero tolerance for spam.")

model_options = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "K-Nearest Neighbors": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest": "model/random_forest.joblib"
}

# Sidebar Controls 
st.sidebar.markdown('<p class="sidebar-heading">Upload Dataset</p>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

st.sidebar.markdown('<p class="sidebar-heading">Select Classifier Model</p>', unsafe_allow_html=True)
selected_model_name = st.sidebar.selectbox("Choose Model", list(model_options.keys()))

# Helper function to load joblib artifacts 
@st.cache_resource
def load_artifact(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

scaler = load_artifact("model/scaler.joblib")
selected_model = load_artifact(model_options[selected_model_name])

if scaler is None:
    st.error("Scaler file not found in `model/scaler.joblib`.")
    st.stop()

# Load Data Logic
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Loaded uploaded dataset.")
elif os.path.exists("test_data.csv"):
    df = pd.read_csv("test_data.csv")
    st.sidebar.info("Using local default test_data.csv.")
else:
    st.warning("Please upload a test_data.csv file to run evaluations.")
    st.stop()

if 'target' not in df.columns:
    st.error("Dataset missing required 'target' column.")
    st.stop()

X_test = df.drop(columns=['target'])
y_test = df['target']

scaled_models = ["Logistic Regression", "K-Nearest Neighbors", "Naive Bayes"]

# Section 1: Detailed Selected Model Performance
st.subheader(f"Detailed Model Performance: {selected_model_name}")

if selected_model is None:
    st.error(f"Selected model binary for {selected_model_name} not found.")
    st.stop()

X_eval = scaler.transform(X_test) if selected_model_name in scaled_models else X_test
y_pred = selected_model.predict(X_eval)
y_prob = selected_model.predict_proba(X_eval)[:, 1] if hasattr(selected_model, "predict_proba") else y_pred

# Metric KPI Highlights
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{acc:.4f}")
col2.metric("AUC-ROC", f"{auc:.4f}")
col3.metric("Precision", f"{prec:.4f}")
col4.metric("Recall", f"{rec:.4f}")
col5.metric("F1-Score", f"{f1:.4f}")
col6.metric("MCC", f"{mcc:.4f}")

# Visualizations: Confusion Matrix & Classification Report
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Ham (0)", "Spam (1)"],
        yticklabels=["Ham (0)", "Spam (1)"], 
        ax=ax
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    st.pyplot(fig)

with right_col:
    st.subheader("Classification Report")
    report_dict = classification_report(
        y_test, y_pred, output_dict=True, target_names=["Ham (0)", "Spam (1)"]
    )
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

st.divider()

# Section 2: Model Comparison Table (All 5 Models Across 6 Metrics)
st.subheader("Model Comparison")

comparison_records = []
for m_name, m_path in model_options.items():
    m_obj = load_artifact(m_path)
    if m_obj is not None:
        X_eval_temp = scaler.transform(X_test) if m_name in scaled_models else X_test
        p_pred = m_obj.predict(X_eval_temp)
        p_prob = m_obj.predict_proba(X_eval_temp)[:, 1] if hasattr(m_obj, "predict_proba") else p_pred
        
        comparison_records.append({
            "Model": m_name,
            "Accuracy": accuracy_score(y_test, p_pred),
            "AUC-ROC": roc_auc_score(y_test, p_prob),
            "Precision": precision_score(y_test, p_pred, zero_division=0),
            "Recall": recall_score(y_test, p_pred, zero_division=0),
            "F1-Score": f1_score(y_test, p_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, p_pred)
        })

comparison_df = pd.DataFrame(comparison_records)
st.dataframe(comparison_df.style.format({
    "Accuracy": "{:.4f}",
    "AUC-ROC": "{:.4f}",
    "Precision": "{:.4f}",
    "Recall": "{:.4f}",
    "F1-Score": "{:.4f}",
    "MCC": "{:.4f}"
}).highlight_max(axis=0, color="#1E3A8A"), use_container_width=True)

st.divider()

# Section 3: Interactive Email Inspector
st.subheader("Interactive Email Inspector")
st.caption("Select any row from your test set to inspect predictions and see how all 5 models vote.")

row_idx = st.number_input("Select Test Row Index", min_value=0, max_value=len(df)-1, value=0, step=1)

selected_row = X_test.iloc[[row_idx]]
actual_label = "Spam" if y_test.iloc[row_idx] == 1 else "Ham"

st.markdown(f"**Actual Ground Truth:** `{actual_label}`")

# Run prediction across all 5 models for the selected row
vote_records = []
for m_name, m_path in model_options.items():
    m_obj = load_artifact(m_path)
    if m_obj is not None:
        X_row = scaler.transform(selected_row) if m_name in scaled_models else selected_row
        pred = m_obj.predict(X_row)[0]
        prob = m_obj.predict_proba(X_row)[0][1] if hasattr(m_obj, "predict_proba") else float(pred)
        
        vote_records.append({
            "Model": m_name,
            "Prediction": "Spam" if pred == 1 else "Ham",
            "Spam Probability": f"{prob * 100:.1f}%"
        })

vote_df = pd.DataFrame(vote_records)
st.dataframe(vote_df, use_container_width=True)

# Data Preview
with st.expander("Preview Evaluated Test Rows"):
    st.dataframe(df.head(50), use_container_width=True)