# ==== Drug Classification: cleaned & fixed (single run) ====
# - Fixes: correct scaler usage (no leakage), proper classifier, robust label encoding
# - Adds: accuracy comparison, confusion matrix plot, readable report

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# -------------------------------------------------------------------
# 1) Load data (edit CSV_PATH if needed)
# -------------------------------------------------------------------
CSV_PATH = r"C:\Users\hp\Desktop\ML_Projects\ML DataSets\drug200.csv"
if not os.path.exists(CSV_PATH):
    CSV_PATH = "drug200.csv"  # fallback if you copy the csv next to this script

df = pd.read_csv(CSV_PATH)

# Quick sanity checks (optional: comment out if you don’t want console output)
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# -------------------------------------------------------------------
# 2) Basic cleaning & standardization of text columns
# -------------------------------------------------------------------
df.columns = [c.strip() for c in df.columns]
for col in ["Sex", "BP", "Cholesterol", "Drug"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Ensure required columns exist
required = {"Age", "Sex", "BP", "Cholesterol", "Na_to_K", "Drug"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# -------------------------------------------------------------------
# 3) Encode target with LabelEncoder (avoids case/casing issues)
# -------------------------------------------------------------------
le_drug = LabelEncoder()
y = le_drug.fit_transform(df["Drug"])
target_names = list(le_drug.classes_)  # for readable reports

# -------------------------------------------------------------------
# 4) Encode features
#   Sex: M/F -> 1/0
#   BP: LOW/NORMAL/HIGH -> 0/1/2
#   Cholesterol: NORMAL/HIGH -> 0/1
# -------------------------------------------------------------------
def map_ci(series, mapping):
    # case-insensitive mapping; unmapped -> NaN
    lut = {k.upper(): v for k, v in mapping.items()}
    return series.str.upper().map(lut)

X = df.copy()
X["Sex"] = map_ci(X["Sex"], {"M": 1, "F": 0})
X["BP"] = map_ci(X["BP"], {"LOW": 0, "NORMAL": 1, "HIGH": 2})
X["Cholesterol"] = map_ci(X["Cholesterol"], {"NORMAL": 0, "HIGH": 1})

if X[["Sex", "BP", "Cholesterol"]].isna().any().any():
    bad = X[X[["Sex", "BP", "Cholesterol"]].isna().any(axis=1)][["Sex", "BP", "Cholesterol"]]
    raise ValueError("Unexpected category in Sex/BP/Cholesterol. Offending rows (head):\n"
                     f"{bad.head()}")

X = X.drop(columns=["Drug"])

# -------------------------------------------------------------------
# 5) Train/test split (stratify to keep class balance)
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -------------------------------------------------------------------
# 6) Scale numeric columns ONLY; fit on train, transform test (no leakage)
# -------------------------------------------------------------------
num_cols = ["Age", "Na_to_K"]
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# -------------------------------------------------------------------
# 7) Train models
# -------------------------------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "RandomForestClassifier": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    ),
}

accuracies = {}
preds = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    preds[name] = (model, y_pred)

# -------------------------------------------------------------------
# 8) Show accuracies
# -------------------------------------------------------------------
print("\n=== Test Accuracies ===")
for name, acc in sorted(accuracies.items(), key=lambda kv: kv[1], reverse=True):
    print(f"{name:>24}: {acc*100:6.2f}%")

best_name = max(accuracies, key=accuracies.get)
best_model, best_pred = preds[best_name]
print(f"\nBest model: {best_name}")

# -------------------------------------------------------------------
# 9) Confusion matrix + classification report for best model
# -------------------------------------------------------------------
cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="g", cbar=False)
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("\n=== Classification Report (Best Model) ===")
print(classification_report(y_test, best_pred, target_names=target_names, zero_division=0))

# -------------------------------------------------------------------
# 10) Optional: feature importances for tree-based models
# -------------------------------------------------------------------
if best_name in ["DecisionTreeClassifier", "RandomForestClassifier"]:
    importances = best_model.feature_importances_
    fi = pd.Series(importances, index=X_train_scaled.columns).sort_values(ascending=False)
    print("\nTop Feature Importances:")
    print(fi)

    plt.figure(figsize=(7, 4))
    fi.head(10).iloc[::-1].plot(kind="barh")
    plt.title(f"Top Feature Importances - {best_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
