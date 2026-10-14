# 50 Startups — Multiple Linear Regression & Feature Selection

## Project Objective

Predict the annual **Profit** of 50 start-up companies using their R&D,
Administration, and Marketing expenditure together with the US State
in which they operate.

- **Part 1** follows the full **CRISP-DM** process (6 phases) using
  Multiple Linear Regression with a scikit-learn Pipeline.
- **Part 2** implements **Greedy Forward Feature Selection** and compares
  feature subsets using RMSE and R-squared.

---

## Dataset

| Item       | Detail |
|------------|--------|
| Source     | GitHub Gist (GaneshSparkz) |
| URL        | `https://gist.githubusercontent.com/GaneshSparkz/b5662effbdae8746f7f7d8ed70c42b2d/raw/faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv` |
| Rows       | 50 |
| Columns    | R&D Spend, Administration, Marketing Spend, State, Profit |
| Target     | **Profit** |

---

## Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib requests tabulate
```

---

## Project Structure

```
startup_regression/
├── part1_crisp_dm.py          ← CRISP-DM (6-phase) regression solution
├── part2_feature_selection.py ← Forward feature selection + comparison
├── regression-web/            ← Interactive Next.js web dashboard for Vercel
└── README.md                  ← This file
```

### Web Dashboard (Vercel)
The `regression-web/` folder contains a Next.js web application featuring:
- Interactive profit predictor with live calculation
- Visual charts for Greedy Forward Feature Selection (R² and RMSE)
- Actual vs. Predicted scatter plot
- Summary of CRISP-DM phases and evaluation metrics

`actual_vs_predicted.png` is saved to the **same directory** from which
Part 1 is run.

---

## How to Run

Open a terminal, navigate to the `startup_regression/` folder, and run:

```bash
# Part 1 – CRISP-DM Multiple Linear Regression
python part1_crisp_dm.py

# Part 2 – Forward Feature Selection
python part2_feature_selection.py
```

Both scripts download the CSV automatically — an active internet
connection is required.

---

## Expected Outputs

### Part 1 (`part1_crisp_dm.py`)
- Console output covering all 6 CRISP-DM phases with plain-English explanations.
- Regression metrics printed to the terminal: **R², MAE, MSE, RMSE**.
- A scatter plot (`actual_vs_predicted.png`) showing Actual vs Predicted Profit.
- A worked example of `model.predict()` for a hypothetical new start-up.

### Part 2 (`part2_feature_selection.py`)
- A Markdown-style table (pipe format) showing feature subsets selected
  from 1 to 5 features with their RMSE and R-squared values (test-set method).
- Automated, data-driven analysis of each iteration.
- A separate CV-based table using cross-validation on training data only,
  with a single, unbiased final evaluation on the held-out test set.
- Clear explanation of the methodological difference between the two approaches.
