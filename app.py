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
CSV_PATH = r"C:\Users\hp\Desktop\ML_Projects\ML DataSets\drug200.csv"
df = pd.read_csv(CSV_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2) Encode target
le = LabelEncoder()
y = le.fit_transform(df["Drug"])
target_names = le.classes_

# 3) Encode categorical features
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

# 6) Train 3 models
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

# 7) Evaluate best model
best_model_name = max(accuracies, key=accuracies.get)
print(f"\nBest Model: {best_model_name}")

best_model = models[best_model_name]
y_pred = best_model.predict(X_test)

# Confusion Matrix
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="g", cbar=False, xticklabels=target_names, yticklabels=target_names)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print(classification_report(y_test, y_pred, target_names=target_names))
