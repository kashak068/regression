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
