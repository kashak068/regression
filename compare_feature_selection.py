"""
Comparison of 5 Feature Selection Algorithms for Multiple Linear Regression
=============================================================================
Dataset : 50_Startups.csv
Target  : Profit

Algorithms Compared:
  1. Lasso (L1 Regularization)              - Embedded Method
  2. Sequential Forward Selection (SFS)     - Wrapper Method (Greedy Forward)
  3. Recursive Feature Elimination (RFE)    - Wrapper Method (Greedy Backward)
  4. ElasticNet (L1 + L2 Regularization)    - Embedded Method
  5. Exhaustive Search (Best Subset)        - Wrapper / Exact Global Search
"""

import sys
import io
import itertools
import time
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tabulate import tabulate

# =============================================================================
# 1. LOAD & PREPARE DATASET
# =============================================================================
DATASET_URL = (
    "https://gist.githubusercontent.com/GaneshSparkz/"
    "b5662effbdae8746f7f7d8ed70c42b2d/raw/"
    "faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv"
)

print("=" * 75)
print("COMPARING 5 MACHINE LEARNING FEATURE SELECTION ALGORITHMS")
print("=" * 75)

print("\n[1/4] Loading dataset...")
try:
    resp = requests.get(DATASET_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    print("  [OK] Dataset loaded successfully (50 records).")
except Exception as e:
    sys.exit(f"  [ERR] Failed to load dataset: {e}")

# One-hot encode State (drop_first=True to avoid dummy variable trap)
df_encoded = pd.get_dummies(df, columns=["State"], drop_first=True)
bool_cols = df_encoded.select_dtypes(include="bool").columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

feature_cols = [c for c in df_encoded.columns if c != "Profit"]
X = df_encoded[feature_cols]
y = df_encoded["Profit"]

# Train/Test Split (80/20, random_state=0 for consistent benchmarking)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Standardized training data for regularized linear models (Lasso, ElasticNet)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)

def refit_and_evaluate(selected_feats):
    """Refits an ordinary linear regression on selected features and evaluates on test set."""
    if not selected_feats:
        return np.nan, np.nan, np.nan
    lr = LinearRegression()
    lr.fit(X_train[selected_feats], y_train)
    preds = lr.predict(X_test[selected_feats])
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    return rmse, r2, mae

# =============================================================================
# 2. RUN 5 FEATURE SELECTION ALGORITHMS
# =============================================================================
print("\n[2/4] Running 5 Feature Selection Algorithms...")
comparison_results = []

# --- 1. LASSO (L1 Regularization) ---
t0 = time.perf_counter()
lasso_cv = LassoCV(cv=5, random_state=0, max_iter=10000).fit(X_train_scaled_df, y_train)
lasso_feats = [col for col, coef in zip(feature_cols, lasso_cv.coef_) if abs(coef) > 1e-4]
# If Lasso eliminated all (very rare), pick non-zero or largest
if not lasso_feats:
    lasso_feats = [feature_cols[np.argmax(np.abs(lasso_cv.coef_))]]
t_lasso = time.perf_counter() - t0
r_rmse, r_r2, r_mae = refit_and_evaluate(lasso_feats)

comparison_results.append({
    "Algorithm": "1. Lasso (L1)",
    "Category": "Embedded",
    "Time_ms": t_lasso * 1000,
    "N_Features": len(lasso_feats),
    "Selected_Features": lasso_feats,
    "RMSE": r_rmse,
    "R2": r_r2,
    "MAE": r_mae,
    "Key_Mechanism": "L1 penalty drives weak/collinear coefficients to exactly 0"
})

# --- 2. FORWARD SFS (Sequential Forward Selection) ---
t0 = time.perf_counter()
# Custom greedy forward selection optimizing cross-validated RMSE on training set
candidates = list(feature_cols)
sfs_selected = []
cv = KFold(n_splits=5, shuffle=True, random_state=0)

for step in range(len(feature_cols)):
    best_step_score = -np.inf
    best_step_feat = None
    for cand in candidates:
        trial = sfs_selected + [cand]
        scores = cross_val_score(LinearRegression(), X_train[trial], y_train, cv=cv, scoring="neg_root_mean_squared_error")
        mean_score = scores.mean()
        if mean_score > best_step_score:
            best_step_score = mean_score
            best_step_feat = cand
    sfs_selected.append(best_step_feat)
    candidates.remove(best_step_feat)
    # Stop when 2 features reached (or based on optimum CV score)
    if len(sfs_selected) == 2:
        break

t_sfs = time.perf_counter() - t0
r_rmse, r_r2, r_mae = refit_and_evaluate(sfs_selected)

comparison_results.append({
    "Algorithm": "2. Forward SFS",
    "Category": "Wrapper (Greedy)",
    "Time_ms": t_sfs * 1000,
    "N_Features": len(sfs_selected),
    "Selected_Features": sfs_selected,
    "RMSE": r_rmse,
    "R2": r_r2,
    "MAE": r_mae,
    "Key_Mechanism": "Sequentially adds the feature providing largest metric gain"
})

# --- 3. RECURSIVE FEATURE ELIMINATION (RFE) ---
t0 = time.perf_counter()
# We target 2 features to compare against optimal 2-feature subset
rfe = RFE(estimator=LinearRegression(), n_features_to_select=2, step=1)
rfe.fit(X_train, y_train)
rfe_feats = [col for col, rank in zip(feature_cols, rfe.ranking_) if rank == 1]
t_rfe = time.perf_counter() - t0
r_rmse, r_r2, r_mae = refit_and_evaluate(rfe_feats)

comparison_results.append({
    "Algorithm": "3. RFE",
    "Category": "Wrapper (Greedy)",
    "Time_ms": t_rfe * 1000,
    "N_Features": len(rfe_feats),
    "Selected_Features": rfe_feats,
    "RMSE": r_rmse,
    "R2": r_r2,
    "MAE": r_mae,
    "Key_Mechanism": "Starts with all features, iteratively prunes smallest weights"
})

# --- 4. ELASTICNET (L1 + L2 Regularization) ---
t0 = time.perf_counter()
enet_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
    cv=5,
    random_state=0,
    max_iter=10000
).fit(X_train_scaled_df, y_train)
enet_feats = [col for col, coef in zip(feature_cols, enet_cv.coef_) if abs(coef) > 1e-4]
if not enet_feats:
    enet_feats = [feature_cols[np.argmax(np.abs(enet_cv.coef_))]]
t_enet = time.perf_counter() - t0
r_rmse, r_r2, r_mae = refit_and_evaluate(enet_feats)

comparison_results.append({
    "Algorithm": "4. ElasticNet",
    "Category": "Embedded",
    "Time_ms": t_enet * 1000,
    "N_Features": len(enet_feats),
    "Selected_Features": enet_feats,
    "RMSE": r_rmse,
    "R2": r_r2,
    "MAE": r_mae,
    "Key_Mechanism": "Balances L1 sparsity and L2 grouping for correlated predictors"
})

# --- 5. EXHAUSTIVE SEARCH (All Subsets / Best Subset Selection) ---
t0 = time.perf_counter()
best_subset = None
best_subset_cv = np.inf

# Evaluate all 2^5 - 1 = 31 subsets using 5-fold CV on train set
for k in range(1, len(feature_cols) + 1):
    for combo in itertools.combinations(feature_cols, k):
        combo_list = list(combo)
        scores = cross_val_score(LinearRegression(), X_train[combo_list], y_train, cv=cv, scoring="neg_root_mean_squared_error")
        mean_rmse = -scores.mean()
        if mean_rmse < best_subset_cv:
            best_subset_cv = mean_rmse
            best_subset = combo_list

t_exh = time.perf_counter() - t0
r_rmse, r_r2, r_mae = refit_and_evaluate(best_subset)

comparison_results.append({
    "Algorithm": "5. Exhaustive Search",
    "Category": "Wrapper (Exact)",
    "Time_ms": t_exh * 1000,
    "N_Features": len(best_subset),
    "Selected_Features": best_subset,
    "RMSE": r_rmse,
    "R2": r_r2,
    "MAE": r_mae,
    "Key_Mechanism": "Guarantees global mathematical optimum across all 2^p - 1 combinations"
})

comp_df = pd.DataFrame(comparison_results)

# =============================================================================
# 3. DISPLAY COMPARISON TABLE
# =============================================================================
print("\n[3/4] Feature Selection Algorithm Comparison Results:")
print("-" * 100)

display_rows = []
for _, r in comp_df.iterrows():
    display_rows.append([
        r["Algorithm"],
        r["Category"],
        f"{r['Time_ms']:.2f} ms",
        r["N_Features"],
        ", ".join(r["Selected_Features"]),
        f"${r['RMSE']:,.2f}",
        f"{r['R2']:.4f}",
        f"${r['MAE']:,.2f}",
    ])

headers = ["Algorithm", "Type", "Time", "# Feats", "Selected Features", "Test RMSE", "Test R²", "Test MAE"]
print(tabulate(display_rows, headers=headers, tablefmt="pipe"))
print("-" * 100)

# =============================================================================
# 4. VISUALIZATION: GENERATE COMPARISON PLOTS
# =============================================================================
print("\n[4/4] Generating comparison plots...")

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

algo_names = [r["Algorithm"].split(". ")[1] for r in comparison_results]
rmse_vals = [r["RMSE"] for r in comparison_results]
r2_vals = [r["R2"] for r in comparison_results]
colors = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ef4444"]

# Subplot 1: Test RMSE (Lower is better)
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.bar(algo_names, rmse_vals, color=colors, width=0.55, edgecolor="black", alpha=0.85)
ax1.set_title("Test RMSE by Algorithm (Lower is Better)", fontsize=13, fontweight="bold", pad=12)
ax1.set_ylabel("RMSE ($)", fontsize=11)
ax1.set_ylim(8000, 9500)
ax1.grid(axis="y", linestyle="--", alpha=0.5)
for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 30, f"${h:,.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Subplot 2: Test R² (Higher is better)
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(algo_names, r2_vals, color=colors, width=0.55, edgecolor="black", alpha=0.85)
ax2.set_title("Test R² by Algorithm (Higher is Better)", fontsize=13, fontweight="bold", pad=12)
ax2.set_ylabel("R²", fontsize=11)
ax2.set_ylim(0.92, 0.96)
ax2.grid(axis="y", linestyle="--", alpha=0.5)
for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.0005, f"{h:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Subplot 3: Feature Selection Matrix (Heatmap)
ax3 = fig.add_subplot(gs[1, :])
matrix_data = np.zeros((len(comparison_results), len(feature_cols)))

for i, res in enumerate(comparison_results):
    for j, f in enumerate(feature_cols):
        if f in res["Selected_Features"]:
            matrix_data[i, j] = 1

im = ax3.imshow(matrix_data, cmap="Blues", aspect="auto", vmin=0, vmax=1.2)
ax3.set_xticks(range(len(feature_cols)))
ax3.set_xticklabels(feature_cols, fontsize=11, fontweight="bold")
ax3.set_yticks(range(len(algo_names)))
ax3.set_yticklabels([r["Algorithm"] for r in comparison_results], fontsize=11, fontweight="bold")
ax3.set_title("Feature Selection Matrix Across 5 Algorithms (Selected = Blue)", fontsize=13, fontweight="bold", pad=12)

# Annotate checkmarks inside cells
for i in range(len(comparison_results)):
    for j in range(len(feature_cols)):
        val = matrix_data[i, j]
        text = "✓ Selected" if val == 1 else "— Dropped"
        color = "white" if val == 1 else "#94a3b8"
        weight = "bold" if val == 1 else "normal"
        ax3.text(j, i, text, ha="center", va="center", color=color, fontsize=11, fontweight=weight)

plt.tight_layout(pad=3.0)
out_img = "algorithm_comparison.png"
plt.savefig(out_img, dpi=180, bbox_inches="tight")
plt.close()
print(f"  [OK] Comparison visualization saved to '{out_img}'")
print("\n" + "=" * 75)
print("EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 75)
