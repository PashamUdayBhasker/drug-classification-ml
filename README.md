

# 💊 Drug Classification — Machine Learning Project

## 📌 Project Overview

This project predicts the most suitable **drug type** (DrugA, DrugB, DrugC, DrugX, DrugY) for a patient based on health indicators using **Machine Learning**.
It showcases a **complete pipeline**: data loading, cleaning, encoding, scaling, model training, evaluation, and visualization.

---

## 📂 Project Structure

```
📁 Drug-Classification-ML
│── 📄 drug_classification.py   # Main Python script
│── 📄 drug200.csv              # Dataset (local)
│── 📄 requirements.txt         # Dependencies
│── 📄 README.md                 # Project documentation
```

---

## 📊 Dataset Details

**Features:**

* `Age` → Patient’s age
* `Sex` → M / F
* `BP` → Blood Pressure level (`LOW`, `NORMAL`, `HIGH`)
* `Cholesterol` → (`NORMAL`, `HIGH`)
* `Na_to_K` → Sodium-to-Potassium ratio

**Target:**

* `Drug` → (`DrugA`, `DrugB`, `DrugC`, `DrugX`, `DrugY`)

---

## ⚙️ Workflow

1️⃣ **Load & Inspect Data** → check shape, missing values
2️⃣ **Clean & Normalize** → remove spaces, case consistency
3️⃣ **Encode Features** → map categorical to numeric
4️⃣ **Train-Test Split** → stratified split (80-20)
5️⃣ **Scale Numeric Features** → avoid leakage by fitting only on training data
6️⃣ **Train Models** → Logistic Regression, Decision Tree, Random Forest
7️⃣ **Evaluate** → accuracy, confusion matrix, classification report
8️⃣ **Select Best Model** → highest accuracy chosen
9️⃣ **Visualize** → heatmap, feature importance (if tree-based)

---

## 🖥 Installation & Run

**1. Clone Repository**

```bash
git clone https://github.com/your-username/drug-classification.git
cd drug-classification
```

**2. Install Dependencies**

```bash
pip install -r requirements.txt
```

**3. Run Project**

```bash
python drug_classification.py
```

---

## 📦 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## 📈 Example Output

* **Accuracy Table** comparing models
* **Confusion Matrix Heatmap** for the best model
* **Classification Report** with precision, recall, F1-score
* **Feature Importance** chart (if applicable)



