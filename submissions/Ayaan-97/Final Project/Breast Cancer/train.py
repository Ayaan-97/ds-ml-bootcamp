import json

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

# Data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Classification Algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Evaluation
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)

# --------------------------------
# 1) Load Dataset
# --------------------------------

CSV_PATH = "./dataset/cleaned_data.csv"

df = pd.read_csv(CSV_PATH)

print("\n=== DATASET HEAD ===")
print(df.head(5))

# Understand Dataset
print("\nDataset Shape:")
print(df.shape)

print("\n=== DATASET INFO ===")
print(df.info())

print("\n=== DATASET DESCRIPTION ===")
print(df.describe())

#  Split Features
X = df.drop('diagnosis', axis=1)

y = df['diagnosis']

# Safety check: remove any remaining missing values in the feature set.
X = X.fillna(X.median(numeric_only=True))

#  Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#  Feature Scaling
scaler = StandardScaler()   
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

#  Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
accuracy_score(y_test, pred_rf)

#  Train SVM
svm = SVC()

svm.fit(x_train_scaled, y_train)

pred_svm = svm.predict(x_test_scaled)

accuracy_score(y_test, pred_svm)

#  Train XGBoost
xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)
xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)


#  Compare Models
print("Random Forest:", accuracy_score(y_test,pred_rf))
print("SVM:", accuracy_score(y_test,pred_svm))
print("XGBoost:", accuracy_score(y_test,xgb_pred))

#  Display Classification Reports
print("\n=== CLASSIFICATION REPORT - RANDOM FOREST ===")
print(classification_report(y_test, pred_rf))

print("\n=== CLASSIFICATION REPORT - SVM ===")
print(classification_report(y_test, pred_svm))

print("\n=== CLASSIFICATION REPORT - XGBOOST ===")
print(classification_report(y_test, xgb_pred))

#  Display Confusion Matrices
print("\n=== CONFUSION MATRIX - SVM ===")
cm = confusion_matrix(y_test,pred_svm)
print(cm)

# Feature Importance
importance = rf.feature_importances_

features = X.columns

feature_importance = pd.DataFrame({
    "Feature":features,
    "Importance":importance
})

feature_importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

import os
import joblib
from sklearn.svm import SVC
# Create models folder
os.makedirs("models", exist_ok=True)

# Train model
svm = SVC(random_state=42, probability=True)
svm.fit(x_train_scaled, y_train)

# Save model
joblib.dump(svm, "models/breast_model.pkl")

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

print("Model and scaler saved successfully!")

# # Save scaler + training feature order for the API (same idea as house deployment)
# os.makedirs("models", exist_ok=True)
# joblib.dump(scaler, "models/breast_scaler.pkl")
# joblib.dump(scale_cols, "models/scale_cols.joblib")

# TRAIN_COLUMNS = df.drop(columns=["Approved"]).columns.tolist()
# json.dump(TRAIN_COLUMNS, open("models/train_columns.json", "w"))
# print(f"\nSaved scaler + {len(TRAIN_COLUMNS)} train columns → models/")