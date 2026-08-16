
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
import joblib
import os

os.makedirs('model', exist_ok=True)
df = pd.read_csv("/home/cloud/Desktop/ML Assignment 2/UCI_Credit_Card.csv")

print("Shape (rows, columns):", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nTotal missing values:", df.isnull().sum().sum())

print("\nTarget column distribution:")
print(df['default.payment.next.month'].value_counts())

df = df.drop("ID", axis=1)

X = df.drop("default.payment.next.month", axis=1)
y = df["default.payment.next.month"]

print("Missing values in each column:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(y.value_counts())

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,      # 20% test data
    random_state=42,     # for my reproducibility
    stratify=y           # to maintain class distribution
)

# Shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Scaler initialization
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("Scaled X_train shape:", X_train_scaled.shape)
print("Scaled X_test shape:", X_test_scaled.shape)

print("\nBefore scaling (X_train.describe()):")
print(X_train.describe().loc[['mean', 'std']])

print("\nAfter scaling (X_train_scaled.describe()):")
print(X_train_scaled.describe().loc[['mean', 'std']])

# ------------------------------------------------------
# 1. Logistic Regression 
# ------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

log_reg_pred = log_reg.predict(X_test_scaled)
log_reg_proba = log_reg.predict_proba(X_test_scaled)[:, 1]   # probability of class 1

print("Logistic Regression trained.")

# ------------------------------------------------------
# 2. Decision Tree
# ------------------------------------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
dt_proba = dt.predict_proba(X_test)[:, 1]

print("Decision Tree trained.")

# ------------------------------------------------------
# 3. K-Nearest Neighbors
# ------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

knn_pred = knn.predict(X_test_scaled)
knn_proba = knn.predict_proba(X_test_scaled)[:, 1]

print("KNN trained.")

# ------------------------------------------------------
# 4. Gaussian Naive Bayes 
# ------------------------------------------------------
nb = GaussianNB()
nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)
nb_proba = nb.predict_proba(X_test)[:, 1]

print("Naive Bayes trained.")

# ------------------------------------------------------
# 5. Random Forest 
# ------------------------------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print("Random Forest trained.")

print("\nAll 5 models trained successfully!")

def get_metrics(y_true, y_pred, y_proba):
    """Computing all 6 required metrics for a given model's predictions."""
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "AUC":       roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall":    recall_score(y_true, y_pred),
        "F1 Score":  f1_score(y_true, y_pred),
        "MCC":       matthews_corrcoef(y_true, y_pred)
    }

# ------------------------------------------------------
# Computing metrics for each of the 5 models
# ------------------------------------------------------
log_reg_metrics = get_metrics(y_test, log_reg_pred, log_reg_proba)
dt_metrics      = get_metrics(y_test, dt_pred, dt_proba)
knn_metrics     = get_metrics(y_test, knn_pred, knn_proba)
nb_metrics      = get_metrics(y_test, nb_pred, nb_proba)
rf_metrics      = get_metrics(y_test, rf_pred, rf_proba)

# ------------------------------------------------------
# Printing each one to verify
# ------------------------------------------------------
print("Logistic Regression:", log_reg_metrics)
print("Decision Tree:      ", dt_metrics)
print("KNN:                ", knn_metrics)
print("Naive Bayes:        ", nb_metrics)
print("Random Forest:      ", rf_metrics)

# ------------------------------------------------------
# Combining all 5 model metrics into one dictionary
# ------------------------------------------------------
all_results = {
    "Logistic Regression": log_reg_metrics,
    "Decision Tree":       dt_metrics,
    "KNN":                 knn_metrics,
    "Naive Bayes":         nb_metrics,
    "Random Forest":       rf_metrics
}

# ------------------------------------------------------
# Converting to a pandas DataFrame
# ------------------------------------------------------
comparison_df = pd.DataFrame(all_results).T   

# Round for cleaner display (keeping 4 decimal places)
comparison_df = comparison_df.round(4)

# Sorting by MCC (most reliable metric for imbalanced data) - descending, to make best model on top
comparison_df = comparison_df.sort_values(by="MCC", ascending=False)

print("MODEL COMPARISON TABLE")
print(comparison_df)

comparison_df.to_csv("model_comparison.csv")
print("\nSaved as model_comparison.csv")

# ------------------------------------------------------
# Creating the model/ folder
# ------------------------------------------------------
os.makedirs("model", exist_ok=True)

# ------------------------------------------------------
# Saving each trained model
# ------------------------------------------------------
joblib.dump(log_reg, "model/logistic_regression.pkl")
joblib.dump(dt,      "model/decision_tree.pkl")
joblib.dump(knn,     "model/knn.pkl")
joblib.dump(nb,      "model/naive_bayes.pkl")
joblib.dump(rf,      "model/random_forest.pkl")

joblib.dump(scaler, "model/scaler.pkl")

print("All 5 models + scaler saved successfully in the 'model/' folder:")
print(os.listdir("model"))

