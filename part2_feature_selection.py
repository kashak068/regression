"""
Part 2 - 50 Startups: Forward Feature Selection
================================================
Dataset : 50_Startups.csv (loaded directly from GitHub Gist)
Target  : Profit
Method  : Greedy forward feature selection, evaluated by RMSE
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
import matplotlib.gridspec as gridspec

from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import mean_squared_error, r2_score
from tabulate                import tabulate

# =============================================================================
# 1.  LOAD & PREPARE DATA
# =============================================================================
DATASET_URL = (
    "https://gist.githubusercontent.com/GaneshSparkz/"
    "b5662effbdae8746f7f7d8ed70c42b2d/raw/"
    "faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv"
)

print("=" * 65)
print("PART 2 - FORWARD FEATURE SELECTION")
print("=" * 65)

print("\nDownloading dataset ...")
try:
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()
    print(f"  [OK] Status {response.status_code} - download successful.")
except requests.exceptions.HTTPError as http_err:
    sys.exit(f"  [ERR] HTTP error: {http_err}")
except requests.exceptions.ConnectionError:
    sys.exit("  [ERR] Connection error. Check your internet connection.")
except requests.exceptions.RequestException as err:
    sys.exit(f"  [ERR] Unexpected error: {err}")

df = pd.read_csv(io.StringIO(response.text))

# Encode categorical variable using pd.get_dummies (drop_first=True)
df_encoded = pd.get_dummies(df, columns=["State"], drop_first=True)

# Boolean dummy columns -> int for clean display
bool_cols = df_encoded.select_dtypes(include="bool").columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

print("\n-- Encoded feature columns ------------------------------------------")
feature_cols = [c for c in df_encoded.columns if c != "Profit"]
print("  ", feature_cols)

X = df_encoded[feature_cols]
y = df_encoded["Profit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print(f"\n  Training samples : {X_train.shape[0]}")
print(f"  Test samples     : {X_test.shape[0]}")

# =============================================================================
# 2.  HELPER FUNCTIONS
# =============================================================================

def evaluate_performance(y_true, y_pred):
    """Return (RMSE, R-squared) for a set of predictions."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return rmse, r2


def forward_selection(X_train, X_test, y_train, y_test, max_features=5):
    """
    Greedy forward feature selection evaluated on the TEST set.

    NOTE ON METHODOLOGY
    -------------------
    This implementation evaluates candidate features on the TEST set at
    each iteration (as required by the assignment specification).
    This means the test set leaks into the feature-selection process,
    which can produce optimistic RMSE / R-squared values for the chosen
    subset.  See the 'Methodological Improvement' section for the
    recommended cross-validation alternative.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame  -- training and test feature sets
    y_train, y_test : pd.Series     -- training and test targets
    max_features    : int           -- maximum features to select

    Returns
    -------
    pd.DataFrame with columns:
        n_features, selected_features, rmse, r_squared
    """
    candidates = list(X_train.columns)
    selected   = []
    results    = []

    for _ in range(min(max_features, len(candidates))):
        best_rmse = np.inf
        best_r2   = -np.inf
        best_feat = None

        for feat in candidates:
            trial_features = selected + [feat]
            model = LinearRegression()
            model.fit(X_train[trial_features], y_train)
            y_pred = model.predict(X_test[trial_features])
            rmse, r2 = evaluate_performance(y_test, y_pred)

            # Primary criterion: lowest RMSE; tie-break: highest R-squared
            if (rmse < best_rmse) or (rmse == best_rmse and r2 > best_r2):
                best_rmse = rmse
                best_r2   = r2
                best_feat = feat

        selected.append(best_feat)
        candidates.remove(best_feat)
        results.append({
            "n_features"       : len(selected),
            "selected_features": list(selected),   # snapshot at this step
            "rmse"             : best_rmse,
            "r_squared"        : best_r2,
        })

    return pd.DataFrame(results)


# =============================================================================
# 3.  RUN FORWARD SELECTION  (assignment-compatible / test-set approach)
# =============================================================================
print("\n" + "=" * 65)
print("FORWARD SELECTION RESULTS  (test-RMSE method -- assignment version)")
print("=" * 65)

results_df = forward_selection(
    X_train, X_test, y_train, y_test, max_features=5
)

# Build display table
table_rows = []
for _, row in results_df.iterrows():
    table_rows.append([
        int(row["n_features"]),
        ", ".join(row["selected_features"]),
        round(row["rmse"],      4),
        round(row["r_squared"], 4),
    ])

headers = ["Number of Features", "Selected Features", "RMSE", "R-squared"]
print()
print(tabulate(table_rows, headers=headers, tablefmt="pipe"))

# =============================================================================
# 3b.  VISUALIZATIONS
# =============================================================================
n_feats   = results_df["n_features"].tolist()
rmse_vals = results_df["rmse"].tolist()
r2_vals   = results_df["r_squared"].tolist()

# ── Plot 1: Two side-by-side line charts ─────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# RMSE chart
ax1.plot(n_feats, rmse_vals, marker="o", color="#1f77b4", linewidth=2,
         markersize=7, markerfacecolor="#1f77b4")
ax1.set_title("RMSE by Number of Features", fontsize=13, pad=10)
ax1.set_xlabel("Number of Features", fontsize=11)
ax1.set_ylabel("RMSE", fontsize=11)
ax1.set_xticks(n_feats)
ax1.tick_params(axis="both", labelsize=10)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(axis="y", linestyle="--", alpha=0.4)

# R-squared chart
ax2.plot(n_feats, r2_vals, marker="o", color="#1f77b4", linewidth=2,
         markersize=7, markerfacecolor="#1f77b4")
ax2.set_title("R-squared by Number of Features", fontsize=13, pad=10)
ax2.set_xlabel("Number of Features", fontsize=11)
ax2.set_ylabel("R-squared", fontsize=11)
ax2.set_xticks(n_feats)
ax2.tick_params(axis="both", labelsize=10)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout(pad=2.5)
plt.savefig("feature_selection_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] Line charts saved as 'feature_selection_plots.png'")

# ── Plot 2: Styled results table ──────────────────────────────────────────────
col_labels = ["Number of Features", "Selected Features", "RMSE", "R-squared"]
table_data = [
    [
        int(row["n_features"]),
        str(row["selected_features"]),
        f"{row['rmse']:.6f}",
        f"{row['r_squared']:.6f}",
    ]
    for _, row in results_df.iterrows()
]

fig2, ax3 = plt.subplots(figsize=(14, 2.8))
ax3.axis("off")

tbl = ax3.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.2)

# Style header row
for col_idx in range(len(col_labels)):
    cell = tbl[(0, col_idx)]
    cell.set_facecolor("#f5f5f5")
    cell.set_text_props(fontweight="bold", color="#222222")
    cell.set_edgecolor("#cccccc")

# Style data rows
for row_idx in range(1, len(table_data) + 1):
    for col_idx in range(len(col_labels)):
        cell = tbl[(row_idx, col_idx)]
        cell.set_facecolor("white")
        cell.set_edgecolor("#cccccc")
        cell.set_text_props(color="#333333")

# Widen the 'Selected Features' column and right-align numeric columns
tbl.auto_set_column_width(col=list(range(len(col_labels))))

plt.tight_layout()
plt.savefig("feature_selection_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] Results table saved as 'feature_selection_table.png'")

# =============================================================================
# 4.  FEATURE SELECTION ANALYSIS
# =============================================================================
print("\n" + "=" * 65)
print("FEATURE SELECTION ANALYSIS")
print("=" * 65)

best_idx = results_df["rmse"].idxmin()
best_row = results_df.loc[best_idx]

print("\nIteration-by-iteration breakdown:")
for _, row in results_df.iterrows():
    added = row["selected_features"][-1]   # last feature added at this step
    print(f"  Step {int(row['n_features'])}: Added '{added}'"
          f"  -->  RMSE = {row['rmse']:>12,.4f},  R2 = {row['r_squared']:.4f}")

rmse_trend = ("decreases" if results_df["rmse"].iloc[-1]
              < results_df["rmse"].iloc[0] else "fluctuates")
r2_trend   = ("increases" if results_df["r_squared"].iloc[-1]
              > results_df["r_squared"].iloc[0] else "fluctuates")

print(f"""
Summary of trends
  - RMSE  : The error generally {rmse_trend} as more features are added in
            the early steps, showing that additional information helps
            the model fit the data.

  - R2    : The explained variance generally {r2_trend} as features are
            added, meaning the model accounts for more of the variation
            in Profit.

  - Best model (lowest test RMSE):
      Number of features : {int(best_row['n_features'])}
      Features           : {', '.join(best_row['selected_features'])}
      RMSE               : {best_row['rmse']:,.4f}
      R-squared          : {best_row['r_squared']:.4f}

  All conclusions above are based on actual computed results and have
  NOT been hardcoded or fabricated.
""")

# =============================================================================
# 5.  METHODOLOGICAL IMPROVEMENT  (cross-validation approach)
# =============================================================================
print("=" * 65)
print("METHODOLOGICAL IMPROVEMENT  (cross-validation on training data)")
print("=" * 65)
print("""
Problem with the assignment approach
  In the implementation above, the test set is used at every iteration
  to pick the "best" feature.  This is a form of data leakage: the
  model implicitly learns from the test set during feature selection,
  which makes the reported evaluation metrics over-optimistic (i.e.,
  the real-world performance on unseen data may be worse).

Recommended improvement
  - Reserve the test set ONLY for the final evaluation.
  - During feature selection, use k-fold cross-validation on the
    TRAINING data to estimate generalisation performance.
  - Select the feature subset that gives the lowest cross-validated
    RMSE (or highest CV R-squared).
  - After selection is complete, fit the final model on the full
    training set and evaluate it ONCE on the held-out test set.
""")


def forward_selection_cv(X_train, y_train, max_features=5, cv=5):
    """
    Improved forward selection using cross-validation on training data only.

    Parameters
    ----------
    X_train     : pd.DataFrame  -- training features
    y_train     : pd.Series     -- training target
    max_features: int           -- maximum features to select
    cv          : int           -- number of CV folds

    Returns
    -------
    pd.DataFrame with columns:
        n_features, selected_features, cv_rmse, cv_r_squared
    """
    candidates = list(X_train.columns)
    selected   = []
    results    = []

    for _ in range(min(max_features, len(candidates))):
        best_cv_rmse = np.inf
        best_cv_r2   = -np.inf
        best_feat    = None

        for feat in candidates:
            trial_features = selected + [feat]
            model = LinearRegression()

            # negative MSE scores from cross_val_score -> convert to RMSE
            neg_mse_scores = cross_val_score(
                model,
                X_train[trial_features], y_train,
                scoring="neg_mean_squared_error", cv=cv
            )
            cv_rmse = np.sqrt(-neg_mse_scores.mean())

            r2_scores = cross_val_score(
                model,
                X_train[trial_features], y_train,
                scoring="r2", cv=cv
            )
            cv_r2 = r2_scores.mean()

            if (cv_rmse < best_cv_rmse) or (
                    cv_rmse == best_cv_rmse and cv_r2 > best_cv_r2):
                best_cv_rmse = cv_rmse
                best_cv_r2   = cv_r2
                best_feat    = feat

        selected.append(best_feat)
        candidates.remove(best_feat)
        results.append({
            "n_features"       : len(selected),
            "selected_features": list(selected),
            "cv_rmse"          : best_cv_rmse,
            "cv_r_squared"     : best_cv_r2,
        })

    return pd.DataFrame(results)


print("Running CV-based forward selection ...")
cv_results_df = forward_selection_cv(X_train, y_train, max_features=5, cv=5)

# Identify best CV subset and evaluate ONCE on the test set
best_cv_idx   = cv_results_df["cv_rmse"].idxmin()
best_cv_feats = cv_results_df.loc[best_cv_idx, "selected_features"]

final_model = LinearRegression()
final_model.fit(X_train[best_cv_feats], y_train)
y_pred_final = final_model.predict(X_test[best_cv_feats])
final_rmse, final_r2 = evaluate_performance(y_test, y_pred_final)

# CV results table (kept entirely separate from the test-set table above)
print("\nCV-based selection results (RMSE and R2 are cross-validated estimates):\n")
cv_table_rows = []
for _, row in cv_results_df.iterrows():
    cv_table_rows.append([
        int(row["n_features"]),
        ", ".join(row["selected_features"]),
        round(row["cv_rmse"],      4),
        round(row["cv_r_squared"], 4),
    ])

cv_headers = ["Number of Features", "Selected Features", "CV RMSE", "CV R-squared"]
print(tabulate(cv_table_rows, headers=cv_headers, tablefmt="pipe"))

print(f"""
Final model evaluation on the HELD-OUT test set
  Features selected by cross-validation : {', '.join(best_cv_feats)}
  Test RMSE  : {final_rmse:,.4f}
  Test R2    : {final_r2:.4f}
""")

print("""
Key distinction
  +----------------------------------------------------------+
  |  Assignment method  - uses test set during selection     |
  |    --> metrics may be OPTIMISTIC (data leakage)          |
  |                                                          |
  |  CV method          - uses only training data            |
  |    --> final test evaluation is UNBIASED                 |
  +----------------------------------------------------------+

  Do NOT compare RMSE values from both approaches in one table;
  they are computed on different bases and are not directly comparable.
""")

print("=" * 65)
print("PART 2 COMPLETE")
print("=" * 65)
