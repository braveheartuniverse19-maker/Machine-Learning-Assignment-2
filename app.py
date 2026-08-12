import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ------------------------------------------------------
# Page configuration
# ------------------------------------------------------
st.set_page_config(
    page_title="ML Classification Model Demo",
    page_icon="🤖",
    layout="wide"
)


# ------------------------------------------------------
# The App title
# ------------------------------------------------------
st.title("Machine Learning Classification Model Demo")
st.write(
    "This Streamlit app allows users to upload test data, select a trained ML model, "
    "and view predictions, evaluation metrics, confusion matrix, and classification report."
)


# ------------------------------------------------------
# Constants
# ------------------------------------------------------
TARGET_COLUMN = "default.payment.next.month"

MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}

SCALER_PATH = "model/scaler.pkl"


# ------------------------------------------------------
# Load model function
# ------------------------------------------------------
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


# ------------------------------------------------------
# Sidebar
# ------------------------------------------------------
st.sidebar.header("User Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload test CSV file",
    type=["csv"]
)

selected_model_name = st.sidebar.selectbox(
    "Select ML Model",
    list(MODEL_PATHS.keys())
)


# ------------------------------------------------------
# Primary logic
# ------------------------------------------------------
if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset Preview")
        st.dataframe(df.head())

        st.write("Dataset shape:", df.shape)

        # Dropping ID column if present
        if "ID" in df.columns:
            df = df.drop("ID", axis=1)

        # Checking if target column exists
        if TARGET_COLUMN not in df.columns:
            st.warning(
                f"Target column '{TARGET_COLUMN}' not found in uploaded data. "
                "The app will show predictions only, not evaluation metrics."
            )

            X_test = df.copy()
            y_test = None

        else:
            X_test = df.drop(TARGET_COLUMN, axis=1)
            y_test = df[TARGET_COLUMN]

        # Load selected model
        model_path = MODEL_PATHS[selected_model_name]

        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.stop()

        model = load_model(model_path)

        # Apply scaling only for Logistic Regression and KNN
        if selected_model_name in ["Logistic Regression", "KNN"]:

            if not os.path.exists(SCALER_PATH):
                st.error("Scaler file not found. Please ensure model/scaler.pkl exists.")
                st.stop()

            scaler = load_scaler()
            X_test_final = scaler.transform(X_test)

        else:
            X_test_final = X_test

        # Predictions
        y_pred = model.predict(X_test_final)

        # Predicted probabilities for AUC
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_final)[:, 1]

        st.subheader("Prediction Results")

        result_df = X_test.copy()
        result_df["Predicted Class"] = y_pred

        if y_proba is not None:
            result_df["Predicted Probability"] = y_proba

        st.dataframe(result_df.head(20))

        # ------------------------------------------------------
        # Metrics 
        # ------------------------------------------------------
        if y_test is not None:

            st.subheader("Evaluation Metrics")

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)

            if y_proba is not None:
                auc = roc_auc_score(y_test, y_proba)
            else:
                auc = np.nan

            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("AUC", f"{auc:.4f}")
            col3.metric("Precision", f"{precision:.4f}")
            col4.metric("Recall", f"{recall:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC", f"{mcc:.4f}")

            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
                "Value": [accuracy, auc, precision, recall, f1, mcc]
            })

            st.write("Metrics Table")
            st.dataframe(metrics_df)

            # ------------------------------------------------------
            # Confusion matrix
            # ------------------------------------------------------
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"],
                ax=ax
            )

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix - {selected_model_name}")

            st.pyplot(fig)

            # ------------------------------------------------------
            # Classification report
            # ------------------------------------------------------
            st.subheader("Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

        else:
            st.info(
                "Since the uploaded file does not contain the target column, "
                "only predictions are displayed."
            )

        # ------------------------------------------------------
        # Download predictions
        # ------------------------------------------------------
        csv_output = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Predictions as CSV",
            data=csv_output,
            file_name="predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("An error occurred while processing the file.")
        st.write(e)

else:
    st.info("Please upload a CSV file from the sidebar to start.")
