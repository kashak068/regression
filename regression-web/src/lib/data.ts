// Model coefficients extracted from the trained sklearn LinearRegression pipeline
// (trained on 80% of the 50-Startups dataset, random_state=0)

export const MODEL = {
    intercept: 42554.168,
    coefs: {
        "State_Florida": -959.284,
        "State_New York": 699.369,
        "R&D Spend": 0.773,
        Administration: 0.033,
        "Marketing Spend": 0.037,
    },
};

export type State = "California" | "Florida" | "New York";

export function predictProfit(
    rdSpend: number,
    admin: number,
    marketing: number,
    state: State
): number {
    const isFL = state === "Florida" ? 1 : 0;
    const isNY = state === "New York" ? 1 : 0;
    return (
        MODEL.intercept +
        MODEL.coefs["State_Florida"] * isFL +
        MODEL.coefs["State_New York"] * isNY +
        MODEL.coefs["R&D Spend"] * rdSpend +
        MODEL.coefs["Administration"] * admin +
        MODEL.coefs["Marketing Spend"] * marketing
    );
}

// ── Part 1 evaluation metrics ─────────────────────────────────────────────────
export const METRICS = [
    { label: "R²", full: "R-squared", value: "0.9347", sub: "93.47% variance explained", color: "#3b82f6" },
    { label: "MAE", full: "Mean Absolute Error", value: "$7,514", sub: "Avg. dollar error per prediction", color: "#8b5cf6" },
    { label: "MSE", full: "Mean Squared Error", value: "83.5M", sub: "Penalises large errors more", color: "#10b981" },
    { label: "RMSE", full: "Root Mean Square Error", value: "$9,138", sub: "Primary benchmark for Part 2", color: "#f59e0b" },
];

// ── CRISP-DM phases ───────────────────────────────────────────────────────────
export const PHASES = [
    { num: "Phase 01", name: "Business Understanding", desc: "Predict startup profit to guide investor decisions and budget allocation. Regression problem." },
    { num: "Phase 02", name: "Data Understanding", desc: "50 rows · 5 columns. No missing values. 3 numerical + 1 categorical feature. Target: Profit." },
    { num: "Phase 03", name: "Data Preparation", desc: "OneHotEncoder(drop='first') for State. 80/20 train-test split. Pipeline prevents leakage." },
    { num: "Phase 04", name: "Modelling", desc: "sklearn LinearRegression in a Pipeline. Fitted on 40 training samples (random_state=0)." },
    { num: "Phase 05", name: "Evaluation", desc: "R²=0.9347, MAE=$7,514, RMSE=$9,138 on 10 test samples. Actual vs. Predicted scatter plot." },
    { num: "Phase 06", name: "Deployment", desc: "Model serialisable via joblib. New predictions via pipeline.predict(new_df). This website!" },
];

// ── Forward feature selection results (test-RMSE method) ─────────────────────
export const FS_RESULTS = [
    { n: 1, features: ["R&D Spend"], rmse: 8274.868, r2: 0.9465 },
    { n: 2, features: ["R&D Spend", "Marketing Spend"], rmse: 8198.797, r2: 0.9474 },
    { n: 3, features: ["R&D Spend", "Marketing Spend", "State_New York"], rmse: 8309.060, r2: 0.9460 },
    { n: 4, features: ["R&D Spend", "Marketing Spend", "State_New York", "State_Florida"], rmse: 8409.917, r2: 0.9447 },
    { n: 5, features: ["R&D Spend", "Marketing Spend", "State_New York", "State_Florida", "Administration"], rmse: 9137.990, r2: 0.9347 },
];

// ── Actual vs Predicted (test set, 10 samples) ────────────────────────────────
export const ACTUAL_VS_PRED = [
    { actual: 103282.38, predicted: 103015.20 },
    { actual: 144259.40, predicted: 132582.28 },
    { actual: 146121.95, predicted: 132447.74 },
    { actual: 77798.83, predicted: 71976.10 },
    { actual: 191050.39, predicted: 178537.48 },
    { actual: 105008.31, predicted: 116161.24 },
    { actual: 81229.06, predicted: 67851.69 },
    { actual: 97483.56, predicted: 98791.73 },
    { actual: 110352.25, predicted: 113969.44 },
    { actual: 166187.94, predicted: 167921.07 },
];

// ── 5 Feature Selection Algorithms Comparison ───────────────────────────────
export interface AlgoItem {
    name: string;
    shortName: string;
    category: "Embedded" | "Wrapper";
    type: string;
    complexity: string;
    features: string[];
    rmse: number;
    r2: number;
    color: string;
    mechanism: string;
    pros: string;
    cons: string;
}

export const ALGO_COMPARISON: AlgoItem[] = [
    {
        name: "Lasso Regression (L1)",
        shortName: "Lasso",
        category: "Embedded",
        type: "Regularization (L1 norm)",
        complexity: "O(p × n) · 1.2 ms",
        features: ["R&D Spend"],
        rmse: 8274.87,
        r2: 0.9465,
        color: "#3b82f6",
        mechanism: "Strong L1 penalty shrinks weaker predictors to zero, selecting the single dominant feature.",
        pros: "Fast, convex optimization; yields the most sparse, parsimonious model with 1 feature.",
        cons: "Drops Marketing Spend because of slight collinearity with R&D Spend."
    },
    {
        name: "Forward SFS (Stepwise)",
        shortName: "Forward SFS",
        category: "Wrapper",
        type: "Greedy Stepwise Additive",
        complexity: "O(k × p × n) · 4.8 ms",
        features: ["R&D Spend", "Marketing Spend"],
        rmse: 8198.80,
        r2: 0.9474,
        color: "#10b981",
        mechanism: "Starts empty, greedily adds R&D Spend, then Marketing Spend where RMSE reduction is maximized.",
        pros: "Discovers the optimal 2-feature subset with minimal function evaluations.",
        cons: "Greedy heuristic cannot re-evaluate earlier feature decisions ('nesting effect')."
    },
    {
        name: "RFE (Recursive Elimination)",
        shortName: "RFE",
        category: "Wrapper",
        type: "Greedy Stepwise Backward",
        complexity: "O(p² × n) · 6.5 ms",
        features: ["R&D Spend", "Marketing Spend", "State_New York"],
        rmse: 8309.06,
        r2: 0.9460,
        color: "#8b5cf6",
        mechanism: "Starts with all 5 features, successively prunes Administration and State_Florida.",
        pros: "Models full multivariate interactions from the start before discarding weak signals.",
        cons: "Retained State_New York before termination, slightly increasing test variance."
    },
    {
        name: "ElasticNet (L1 + L2)",
        shortName: "ElasticNet",
        category: "Embedded",
        type: "Hybrid Regularization",
        complexity: "O(p × n) · 18.4 ms",
        features: ["R&D Spend", "Marketing Spend", "Administration"],
        rmse: 8485.62,
        r2: 0.9436,
        color: "#f59e0b",
        mechanism: "Combines L1 sparsity with L2 Ridge grouping effect to keep correlated expenditure features.",
        pros: "Does not arbitrarily discard correlated predictors; groups R&D, Marketing, and Admin.",
        cons: "Retaining Administration adds mild noise, resulting in slightly higher test RMSE."
    },
    {
        name: "Exhaustive Search (Best Subset)",
        shortName: "Exhaustive",
        category: "Wrapper",
        type: "Exact Global Optimization",
        complexity: "O(2ᵖ × n) · 45.2 ms",
        features: ["R&D Spend", "Marketing Spend"],
        rmse: 8198.80,
        r2: 0.9474,
        color: "#ef4444",
        mechanism: "Computes all 2⁵ - 1 = 31 combinations across 1~5 features with 5-fold cross-validation.",
        pros: "Mathematically guaranteed to find the true global optimum across the entire hypothesis space.",
        cons: "Exponential complexity O(2^p); only practical when p is small (p ≤ 20)."
    }
];
