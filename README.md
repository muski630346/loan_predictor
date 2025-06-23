# loan_predictor
# Loan Eligibility Predictor

A machine learning model that predicts whether a loan should be approved based on applicant details such as income, credit history, employment, and education. Built using Logistic Regression and Random Forest classifiers.

---

# Problem Statement

Loan defaulters pose major risks to banks. By predicting loan eligibility early, financial institutions can reduce risk and streamline approvals.

---

# Objective

Build a supervised ML model to predict loan approval (`Yes` or `No`) using structured applicant data.

---

# Dataset

- **Source**: [Kaggle - Loan Prediction Dataset](https://www.kaggle.com/datasets/dipam7/loan-prediction-problem-dataset)
- **Format**: CSV (`loan_prediction.csv`)
- Features: Gender, Marital Status, Dependents, Education, Income, Credit History, Loan Amount, etc.
- Target: `Loan_Status` (`Y` = Approved, `N` = Not Approved)

---

# Tools & Libraries

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Seaborn, Matplotlib  
- Logistic Regression & Random Forest Classifier

---

# Model Pipeline

1. Data cleaning and imputation for missing values  
2. Label encoding for categorical variables  
3. Feature scaling and train/test split  
4. Model training:
   - Logistic Regression (baseline)
   - Random Forest Classifier (advanced)
5. Evaluation: Accuracy, Precision, Recall, ROC-AUC  
6. Save trained model (`.pkl` optional)

---

#Results

| Model              | Accuracy | ROC AUC |
|-------------------|----------|---------|
| Logistic Regression | ~82%    | ~0.84   |
| Random Forest       | ~87% ✅ | ~0.89 ✅ |

---

# How to Run

1. Clone this repo or open the notebook
2. Install required libraries:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
