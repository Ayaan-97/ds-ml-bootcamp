import pandas as pd
import numpy as np
import joblib
import os

# Load Dataset

CSV_PATH = './dataset/data.csv'
df = pd.read_csv(CSV_PATH)
df.head(5)

print("\n=== INITIAL INFO ===")
print(df.info())

print("\n=== INITIAL MISSING VALUES ===")
print(df.isnull().sum())
print(df.describe())

#  Check Target
print("\n=== TARGET VALUE COUNTS ===")
print(df['diagnosis'].value_counts())

# Encode the target as numeric labels.
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
print("\n=== DATASET HEAD AFTER ENCODING ===")
print(df.head(5))

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

# Remove duplicates
before = df.shape
df = df.drop_duplicates()
after = df.shape

print(f"Dropped duplicates: {before} → {after}")

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)

IQR = Q3 - Q1

X_clean = X[
    ~((X < (Q1 - 1.5 * IQR)) |
      (X > (Q3 + 1.5 * IQR))).any(axis=1)
]
y_clean = y.loc[X_clean.index]

print("Original shape:", X.shape)
print("After removing outliers:", X_clean.shape)
# Removing unnecessary columns and 
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
print("\n=== DATASET HEAD ===")
print(df.head(5))

OUT_PATH = "./dataset/cleaned_data.csv"
df.to_csv(OUT_PATH, index=False)

print(
     f"Saved cleaned dataset to {OUT_PATH}" )