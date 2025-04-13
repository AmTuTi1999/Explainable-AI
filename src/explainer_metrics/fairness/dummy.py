import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)

# Step 1: Generate synthetic data with a causal structure
N = 1000
gender = np.random.binomial(1, 0.5, N)  # 0 = male, 1 = female
education = np.random.normal(3, 1, N)  # years of higher education
income = 30 + 10 * education - 5 * gender + np.random.normal(0, 2, N)  # income depends on gender
loan_approved = (income > 60).astype(int)

data = pd.DataFrame({
    'gender': gender,
    'education': education,
    'income': income,
    'loan_approved': loan_approved
})

# Step 2: Train a model using all features (including gender)
X = data[['gender', 'education', 'income']]
y = data['loan_approved']
model = LogisticRegression()
model.fit(X, y)

# Step 3: Create counterfactuals (flip gender, recompute income using causal formula)
def generate_counterfactual(row):
    counterfactual_gender = 1 - row['gender']
    counterfactual_income = 30 + 10 * row['education'] - 5 * counterfactual_gender
    return pd.Series({
        'gender': counterfactual_gender,
        'education': row['education'],
        'income': counterfactual_income
    })

X_cf = data.apply(generate_counterfactual, axis=1)

# Step 4: Predict actual and counterfactual outcomes
y_pred = model.predict(X)
y_pred_cf = model.predict(X_cf)

# Step 5: Evaluate counterfactual fairness
unchanged = y_pred == y_pred_cf
cf_fairness = np.mean(unchanged)
print(f"Counterfactual fairness score: {cf_fairness:.2f} (1.0 = fully fair)")

# Optional: show examples of unfair predictions
unfair_cases = data.loc[~unchanged].copy()
unfair_cases['y_pred'] = y_pred[~unchanged]
unfair_cases['y_pred_cf'] = y_pred_cf[~unchanged]
print("\nExamples of counterfactual unfairness:")
print(unfair_cases[['gender', 'education', 'income', 'y_pred', 'y_pred_cf']].head())
