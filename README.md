# 👨‍💼 Employee Attrition Prediction – Human Resources Analytics

---

## 🚀 Business Objective

Hiring and retaining talent is one of the **most expensive and time-consuming challenges** for companies today.

📊 **Industry Stats:**

- Small business owners spend **40%** of working hours on tasks like hiring (non-revenue generating).
- It takes **52 days on average** to fill a vacant role.
- Hiring a new employee costs an average of **$7,645** (0–500 employee companies).
- Companies lose **15–20% of an employee’s salary** just to recruit a replacement.
- Time taken to train a new hire costs **1–2.5% of company revenue**.

👉 **Goal:** Build a model that can **predict which employees are at risk of leaving**, helping HR reduce turnover, save hiring costs, and maintain productivity.

---

## 📂 Dataset Overview

- 📈 **Source:** [IBM HR Analytics Dataset on Kaggle](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- 📄 **Size:** 1,470 employees
- 🎯 **Target:** `Attrition`
  - 1 = Will Leave
  - 0 = Will Stay
- 🧩 **Features include:**
  - `JobInvolvement`, `JobSatisfaction`, `WorkLifeBalance`, `Education`, `RelationshipSatisfaction`, `PerformanceRating`, `OverTime`, `DistanceFromHome`, and more

---

## 🛠️ Data Preprocessing Steps

1. **Label Encoding:** `Attrition`, `OverTime`, and `Over18` converted to 0/1
2. **OneHot Encoding:** Applied to categorical variables like `BusinessTravel`, `Gender`, `JobRole`
3. **Removed Features:** Dropped `EmployeeNumber`, `EmployeeCount`, `StandardHours`, `Over18`
4. **Feature Scaling:** Applied `MinMaxScaler` for normalization
5. **Train-Test Split:** 75% training, 25% testing

---

## ⚠️ Tackling Class Imbalance

Only ~16% of employees left in the dataset — leading to biased models.

✅ Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training data before fitting models:

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=0)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

---

## 🤖 Models Compared

| # | Model               | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Accuracy |
| - | ------------------- | ------------------- | ---------------- | ------------------ | -------- |
| 0 | Logistic Regression | 0.38                | ✅ **0.66**       | 0.48               | 0.75     |
| 1 | Random Forest       | 0.70                | 0.25             | 0.37               | 0.85     |
| 2 | Deep Neural Network | 0.51                | 0.45             | 0.48               | 0.83     |
| 3 | XGBoost (🔼 Added)  | 0.56                | 0.36             | 0.44               | 0.84     |
| 4 | LightGBM (🔼 Added) | 0.69                | 0.34             | 0.46               | 0.86     |
| 5 | CatBoost (🔼 Added) | 0.68                | 0.33             | 0.44               | 0.86     |

> ✅ Models 3–5 (XGBoost, LightGBM, CatBoost) were additionally implemented by **Vaibhav Sharma** to extend the project.

---

## 📈 Model Evaluation Insights

- **Logistic Regression** had the **highest recall**, meaning it flagged most employees who might leave — useful when HR wants to minimize attrition risk.
- **DNN (Deep Neural Network)** offered balanced performance with moderate F1 and accuracy.
- **XGBoost, LightGBM, CatBoost** had **higher accuracy and precision**, making them good choices when we want to avoid false positives.
- **SMOTE** helped improve recall across all models.

---

## 📊 Visualizations

- Heatmaps for correlations and missing values
- KDE plots to compare leavers vs. stayers
- Countplots for `JobRole`, `MaritalStatus`, `JobLevel`, `JobInvolvement`
- Boxplots for income distribution by `Gender` and `JobRole`
- DNN training loss & accuracy progression

---

## ✅ How to Use

1. Upload your HR data in the same format (`.csv`)
2. Run the preprocessing pipeline
3. Apply SMOTE for balanced sampling
4. Train any of the 6 available models
5. Evaluate and choose the best performing model
6. Flag employees with high attrition risk for intervention

---

## 💡 Future Enhancements

- Add **hyperparameter tuning** (e.g. GridSearchCV, Optuna)
- Integrate **explainability tools** (e.g. SHAP, LIME)
- Build a **Streamlit or Flask dashboard**
- Use **MLflow** for experiment tracking and model registry

---

## 📎 References

- Kaggle Dataset: [IBM HR Analytics](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Udemy Course: *Solve 6 Real Business Problems – Section 2: Human Resources*
- SMOTE: `imbalanced-learn`
- Models: `sklearn`, `xgboost`, `lightgbm`, `catboost`, `tensorflow`

**Enhanced by:** *Vaibhav Sharma* (Data Science Enthusiast)
---

**Maintained by:** [Vaibhav Sharma](https://github.com/vaisharma16)  
📍 *System Engineer at TCS | Data Science Enthusiast*

---
