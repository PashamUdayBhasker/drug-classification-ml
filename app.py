
# Drug Classification

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1) Load the dataset

df = pd.read_csv(r"/content/drug200.csv")


print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2) Pie Chart: Drug Distribution

plt.figure(figsize=(6,6))
df["Drug"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, cmap="tab20")
plt.ylabel("")
plt.title("Drug Distribution in Dataset")
plt.show()

# 3) Encode target and categorical features

le = LabelEncoder()
y = le.fit_transform(df["Drug"])
target_names = le.classes_

df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
df["BP"] = df["BP"].map({"LOW": 0, "NORMAL": 1, "HIGH": 2})
df["Cholesterol"] = df["Cholesterol"].map({"NORMAL": 0, "HIGH": 1})

X = df.drop(columns=["Drug"])

# 4) Train/Test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) Scale numeric features

scaler = StandardScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train[["Age", "Na_to_K"]] = scaler.fit_transform(X_train[["Age", "Na_to_K"]])
X_test[["Age", "Na_to_K"]] = scaler.transform(X_test[["Age", "Na_to_K"]])

# 6) Train models

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42)
}

accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name}: {acc*100:.2f}%")

# 7) Bar Chart: Model Accuracies

plt.figure(figsize=(7,5))
plt.bar(accuracies.keys(), accuracies.values(), color="skyblue", edgecolor="black")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0,1)
for i, v in enumerate(accuracies.values()):
    plt.text(i, v+0.01, f"{v*100:.2f}%", ha='center')
plt.show()

# 8) Evaluate Best Model

best_model_name = max(accuracies, key=accuracies.get)
print(f"\nBest Model: {best_model_name}")

best_model = models[best_model_name]
y_pred = best_model.predict(X_test)

# Confusion Matrix with Drug Names

plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="g", cbar=False,
            xticklabels=target_names, yticklabels=target_names, cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 9) Feature Importances (if tree-based)

if best_model_name in ["Decision Tree", "Random Forest"]:
    importances = best_model.feature_importances_
    fi = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

    plt.figure(figsize=(8,5))
    fi.plot(kind="bar", color="coral", edgecolor="black")
    plt.title(f"Feature Importances - {best_model_name}")
    plt.ylabel("Importance")
    plt.show()


#########################################################################
#### Simple Interactive CLI #########################

print("=== Drug Prediction CLI ===")
print("Type 'quit' anytime to exit.\n")

while True:
    age = input("Enter Age (e.g. 45): ")
    if age.lower() == "quit":
        break
    age = float(age)
    
    sex = input("Enter Sex (M/F): ")
    if sex.lower() == "quit":
        break
    sex = 1 if sex.upper() == "M" else 0

    bp = input("Enter BP (LOW/NORMAL/HIGH): ")
    if bp.lower() == "quit":
        break
    bp_dict = {"LOW":0, "NORMAL":1, "HIGH":2}
    bp = bp_dict[bp.upper()]

    chol = input("Enter Cholesterol (NORMAL/HIGH): ")
    if chol.lower() == "quit":
        break
    chol_dict = {"NORMAL":0, "HIGH":1}
    chol = chol_dict[chol.upper()]

    na_to_k = input("Enter Na_to_K (e.g. 15.5): ")
    if na_to_k.lower() == "quit":
        break
    na_to_k = float(na_to_k)

    # Prepare input for prediction
    
    user_df = pd.DataFrame([[age, sex, bp, chol, na_to_k]], columns=X.columns)
    user_df[["Age", "Na_to_K"]] = scaler.transform(user_df[["Age", "Na_to_K"]])

    # Predict
    pred_encoded = model.predict(user_df)[0]
    pred_drug = target_names[pred_encoded]  # Get original drug name
    print(f"\nPredicted Drug: {pred_drug}\n")
