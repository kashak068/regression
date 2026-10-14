"""
Part 1 - 50 Startups: Multiple Linear Regression via CRISP-DM
=============================================================
Dataset : 50_Startups.csv (loaded directly from GitHub Gist)
Target  : Profit
Features: R&D Spend, Administration, Marketing Spend, State
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import sys
import io
import requests
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend -- no GUI window opened
import matplotlib.pyplot as plt

from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import OneHotEncoder
from sklearn.pipeline        import Pipeline
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (r2_score, mean_absolute_error,
                                     mean_squared_error)

# =============================================================================
# PHASE 1 - BUSINESS UNDERSTANDING
# =============================================================================
print("=" * 65)
print("PHASE 1 - BUSINESS UNDERSTANDING")
print("=" * 65)
print("""
Objective
---------
A venture-capital firm wants to estimate the annual profit of start-ups
based on three expenditure categories (R&D, Administration, Marketing)
and the US State in which the company operates.

Why does profit prediction matter?
  - Investors can rank portfolio companies by expected return.
  - Start-up founders can allocate budgets more effectively by
    understanding which spending category drives profit most.
  - Early identification of under-performing start-ups allows timely
    strategic intervention.

Problem type : REGRESSION
  We are predicting a continuous numerical target (Profit) rather than
  a discrete class label, so we use Multiple Linear Regression.
""")

# =============================================================================
# PHASE 2 - DATA UNDERSTANDING
# =============================================================================
print("=" * 65)
print("PHASE 2 - DATA UNDERSTANDING")
print("=" * 65)

DATASET_URL = (
    "https://gist.githubusercontent.com/GaneshSparkz/"
    "b5662effbdae8746f7f7d8ed70c42b2d/raw/"
    "faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv"
)

print("\nDownloading dataset ...")
try:
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()
    print(f"  [OK] Status {response.status_code} - download successful.")
except requests.exceptions.HTTPError as http_err:
    sys.exit(f"  [ERR] HTTP error: {http_err}")
except requests.exceptions.ConnectionError:
    sys.exit("  [ERR] Connection error. Check your internet connection.")
except requests.exceptions.Timeout:
    sys.exit("  [ERR] Request timed out.")
except requests.exceptions.RequestException as err:
    sys.exit(f"  [ERR] Unexpected error: {err}")

df = pd.read_csv(io.StringIO(response.text))

print("\n-- First five rows --------------------------------------------------")
print(df.head().to_string(index=False))

print("\n-- Column names -----------------------------------------------------")
print(list(df.columns))

print("\n-- Dataset dimensions -----------------------------------------------")
print(f"  Rows : {df.shape[0]}")
print(f"  Cols : {df.shape[1]}")

print("\n-- Data types -------------------------------------------------------")
print(df.dtypes.to_string())

print("\n-- Missing values ---------------------------------------------------")
print(df.isnull().sum().to_string())

print("\n-- Descriptive statistics (numerical columns) -----------------------")
print(df.describe().round(2).to_string())

print("""
Key observations
  - Numerical features : R&D Spend, Administration, Marketing Spend
  - Categorical feature : State  (New York / California / Florida)
  - Target variable    : Profit  (continuous -- confirms regression task)
  - Dataset size        : 50 rows  (small, so we are careful about splits)
  - No missing values detected.
""")

# =============================================================================
# PHASE 3 - DATA PREPARATION
# =============================================================================
print("=" * 65)
print("PHASE 3 - DATA PREPARATION")
print("=" * 65)

REQUIRED_COLS = {"R&D Spend", "Administration", "Marketing Spend",
                 "State", "Profit"}
missing = REQUIRED_COLS - set(df.columns)
if missing:
    sys.exit(f"  [ERR] Missing columns in dataset: {missing}")

# Separate features and target
X = df.drop(columns=["Profit"])
y = df["Profit"]

# Column groups
numerical_cols   = ["R&D Spend", "Administration", "Marketing Spend"]
categorical_cols = ["State"]

# Preprocessor: one-hot-encode State, pass through numerical columns
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numerical_cols),
])

# Train / test split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print(f"\n  Training samples : {X_train.shape[0]}")
print(f"  Test samples     : {X_test.shape[0]}")
print("""
Encoding strategy
  - OneHotEncoder(drop='first') removes one dummy column per categorical
    variable to avoid multicollinearity (dummy-variable trap).
  - handle_unknown='ignore' makes the encoder safe if an unseen category
    appears at prediction time.
  - Preprocessing is embedded inside a scikit-learn Pipeline so the
    encoder is NEVER fitted on test data.
""")

# =============================================================================
# PHASE 4 - MODELING
# =============================================================================
print("=" * 65)
print("PHASE 4 - MODELING")
print("=" * 65)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor",    LinearRegression()),
])

pipeline.fit(X_train, y_train)
print("\n  [OK] LinearRegression model trained successfully.")
print("""
What is Multiple Linear Regression?
  It fits a straight-line (hyperplane) relationship between the input
  features and the target by minimising the Sum of Squared Errors.

      Profit ~ b0 + b1*(R&D Spend) + b2*(Administration)
                  + b3*(Marketing Spend) + b4*(State_dummy...)

  The fitted coefficients tell us the expected change in Profit for a
  one-unit increase in each feature, all else equal.
""")

# =============================================================================
# PHASE 5 - EVALUATION
# =============================================================================
print("=" * 65)
print("PHASE 5 - EVALUATION")
print("=" * 65)

y_pred = pipeline.predict(X_test)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\n  R2   (R-squared)            : {r2:.4f}")
print(f"  MAE  (Mean Absolute Error)  : {mae:,.2f}")
print(f"  MSE  (Mean Squared Error)   : {mse:,.2f}")
print(f"  RMSE (Root Mean Sq. Error)  : {rmse:,.2f}")

print("""
What each metric means (beginner-friendly)
  R2   - "How much of the variation in Profit does our model explain?"
          A value of 1.0 is perfect; 0.0 means no better than the mean.

  MAE  - "On average, how many dollars is each prediction off by?"
          Easy to interpret because it is in the same units as Profit.

  MSE  - "Average of squared errors."
          Penalises large errors more heavily than MAE. Its units are
          dollars-squared, so harder to interpret directly.

  RMSE - sqrt(MSE), back in dollar units. This is the most commonly
          reported regression error metric because it is interpretable
          and sensitive to large individual errors.
""")

# Actual vs. Predicted scatter plot
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred, color="#4C72B0", edgecolors="white",
           s=80, alpha=0.85, label="Predicted vs Actual")

# y = x reference line
line_min = min(y_test.min(), y_pred.min())
line_max = max(y_test.max(), y_pred.max())
ax.plot([line_min, line_max], [line_min, line_max],
        color="#DD4444", linewidth=1.5, linestyle="--", label="Ideal (y = x)")

ax.set_xlabel("Actual Profit",    fontsize=12)
ax.set_ylabel("Predicted Profit", fontsize=12)
ax.set_title("Actual vs Predicted Profit", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
print("  [OK] Plot saved as 'actual_vs_predicted.png'")

# =============================================================================
# PHASE 6 - DEPLOYMENT
# =============================================================================
print("=" * 65)
print("PHASE 6 - DEPLOYMENT")
print("=" * 65)
print("""
How to deploy the model
  1. Save the fitted pipeline to disk using joblib:
         import joblib
         joblib.dump(pipeline, 'startup_profit_model.pkl')

  2. In a web service (e.g. Flask / FastAPI), load the model and call
     predict() on incoming JSON data converted to a DataFrame:
         loaded_model = joblib.load('startup_profit_model.pkl')
         prediction   = loaded_model.predict(new_data_df)

  3. Return the prediction as JSON to the client.

Example: predicting profit for a new start-up
""")

# Example new start-up data (must match the training feature schema)
new_startup = pd.DataFrame({
    "R&D Spend"      : [150000.0],
    "Administration" : [120000.0],
    "Marketing Spend": [300000.0],
    "State"          : ["California"],
})

predicted_profit = pipeline.predict(new_startup)
print(f"  Input data:\n{new_startup.to_string(index=False)}")
print(f"\n  Predicted Profit : ${predicted_profit[0]:,.2f}")
print("""
Important notes
  - The input DataFrame must have exactly the same column names and
    data types as the training data.
  - The Pipeline handles encoding automatically -- no manual pre-
    processing is needed at prediction time.
  - Unknown states (e.g. 'Texas') are handled gracefully because
    we set handle_unknown='ignore' in the encoder.
""")

print("=" * 65)
print("PART 1 COMPLETE")
print("=" * 65)
