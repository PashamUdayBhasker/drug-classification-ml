# Drug Classification Machine Learning Project

This project predicts which type of drug is most suitable for a patient based on medical attributes like **Age**, **Sex**, **Blood Pressure**, **Cholesterol level**, and **Sodium-to-Potassium ratio**.  
It uses multiple machine learning models to compare performance and select the best one.

---

## 📂 Project Files
- **drug_classification.py** – Main Python script for data preprocessing, model training, and evaluation.
- **requirements.txt** – List of Python libraries required to run the project.
- **drug200.csv** – Dataset containing patient details and drug types (not included in repo for size/privacy reasons — add it locally).

---

## 📊 Dataset Information
The dataset contains the following columns:
| Column       | Description |
|--------------|-------------|
| Age          | Age of the patient |
| Sex          | Male (M) / Female (F) |
| BP           | Blood pressure level: LOW / NORMAL / HIGH |
| Cholesterol  | Cholesterol level: NORMAL / HIGH |
| Na_to_K      | Sodium-to-Potassium ratio in the blood |
| Drug         | Drug type (DrugA, DrugB, DrugC, DrugX, DrugY) |

---

## ⚙️ Project Workflow
1. **Load & Explore Data** – Read the dataset, check structure, and inspect missing values.
2. **Data Preprocessing** – Encode categorical variables, scale numeric features.
3. **Train/Test Split** – Split data into 80% training and 20% testing.
4. **Model Training** – Train the following ML models:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
5. **Model Evaluation** – Compare models using accuracy score, confusion matrix, and classification report.
6. **Select Best Model** – Based on highest accuracy.

---

## 📈 Example Output
- Accuracy comparison of models:
