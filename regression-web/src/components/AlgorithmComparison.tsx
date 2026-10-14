"use client";

import React, { useState } from "react";
import { ALGO_COMPARISON, AlgoItem } from "@/lib/data";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from "recharts";

export default function AlgorithmComparison() {
    const [selectedIdx, setSelectedIdx] = useState(0);
    const selectedAlgo = ALGO_COMPARISON[selectedIdx];

    const chartData = ALGO_COMPARISON.map(a => ({
        name: a.shortName,
        rmse: a.rmse,
        r2: a.r2,
        color: a.color
    }));

    return (
        <div className="card" style={{ marginTop: "24px" }}>
            <div style={{ marginBottom: "20px" }}>
                <span className="badge-best" style={{ marginBottom: "8px", display: "inline-block" }}>
                    Machine Learning Benchmark
                </span>
                <h3 style={{ fontSize: "1.35rem", fontWeight: "700", color: "var(--text)" }}>
                    5 Feature Selection Algorithms Compared
                </h3>
                <p style={{ color: "var(--text-muted)", fontSize: "0.92rem", marginTop: "4px" }}>
                    Evaluating Embedded (Lasso, ElasticNet), Greedy Wrappers (Forward SFS, RFE), and Exact Global Search (Exhaustive) on the 50 Startups dataset.
                </p>
            </div>

            {/* Algorithm Selector Tabs */}
            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", marginBottom: "20px" }}>
                {ALGO_COMPARISON.map((algo, i) => (
                    <button
                        key={algo.name}
                        onClick={() => setSelectedIdx(i)}
                        style={{
                            padding: "8px 16px",
                            borderRadius: "8px",
                            border: selectedIdx === i ? `2px solid ${algo.color}` : "1px solid var(--border)",
                            background: selectedIdx === i ? "rgba(59, 130, 246, 0.12)" : "var(--card-bg, #1e293b)",
                            color: selectedIdx === i ? "var(--text, #fff)" : "var(--text-muted, #94a3b8)",
                            fontWeight: selectedIdx === i ? "700" : "500",
                            cursor: "pointer",
                            transition: "all 0.2s ease",
                            display: "flex",
                            alignItems: "center",
                            gap: "8px"
                        }}
                    >
                        <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: algo.color }}></span>
                        {algo.shortName}
                    </button>
                ))}
            </div>

            {/* Selected Algorithm Deep-Dive Card */}
            <div
                style={{
                    background: "rgba(15, 23, 42, 0.5)",
                    border: "1px solid var(--border)",
                    borderRadius: "12px",
                    padding: "18px 20px",
                    marginBottom: "28px"
                }}
            >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "10px" }}>
                    <div>
                        <h4 style={{ fontSize: "1.15rem", fontWeight: "700", color: selectedAlgo.color }}>
                            {selectedAlgo.name}
                        </h4>
                        <div style={{ display: "flex", gap: "8px", marginTop: "6px" }}>
                            <span className="pill" style={{ background: "rgba(59, 130, 246, 0.15)", color: "#60a5fa" }}>
                                {selectedAlgo.category} Method
                            </span>
                            <span className="pill" style={{ background: "rgba(148, 163, 184, 0.12)", color: "#cbd5e1" }}>
                                {selectedAlgo.type}
                            </span>
                            <span className="pill" style={{ background: "rgba(16, 185, 129, 0.15)", color: "#34d399" }}>
                                {selectedAlgo.complexity}
                            </span>
                        </div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                        <div style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>Test RMSE</div>
                        <div style={{ fontSize: "1.3rem", fontWeight: "800", color: "#10b981" }}>${selectedAlgo.rmse.toFixed(2)}</div>
                        <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>R² = {selectedAlgo.r2.toFixed(4)}</div>
                    </div>
                </div>

                <div style={{ marginTop: "14px", borderTop: "1px solid var(--border)", paddingTop: "12px", fontSize: "0.9rem" }}>
                    <p style={{ margin: "4px 0", color: "var(--text)" }}>
                        <strong>Core Mechanism:</strong> {selectedAlgo.mechanism}
                    </p>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "12px", marginTop: "10px" }}>
                        <div style={{ background: "rgba(16, 185, 129, 0.08)", padding: "10px", borderRadius: "8px", borderLeft: "3px solid #10b981" }}>
                            <strong style={{ color: "#34d399" }}>✓ Key Advantages:</strong>
                            <div style={{ color: "var(--text)", marginTop: "2px" }}>{selectedAlgo.pros}</div>
                        </div>
                        <div style={{ background: "rgba(239, 68, 68, 0.08)", padding: "10px", borderRadius: "8px", borderLeft: "3px solid #ef4444" }}>
                            <strong style={{ color: "#f87171" }}>✗ Trade-offs:</strong>
                            <div style={{ color: "var(--text)", marginTop: "2px" }}>{selectedAlgo.cons}</div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Comparison Charts */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "20px", marginBottom: "28px" }}>
                <div className="card-sm">
                    <h4 style={{ marginBottom: "10px", color: "var(--text)" }}>Test RMSE ($) — Lower is Better</h4>
                    <ResponsiveContainer width="100%" height={230}>
                        <BarChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                            <YAxis domain={[8000, 8500]} />
                            <Tooltip />
                            <Bar dataKey="rmse" radius={[4, 4, 0, 0]}>
                                {chartData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                <div className="card-sm">
                    <h4 style={{ marginBottom: "10px", color: "var(--text)" }}>Test R² (Variance Explained) — Higher is Better</h4>
                    <ResponsiveContainer width="100%" height={230}>
                        <BarChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                            <YAxis domain={[0.94, 0.95]} />
                            <Tooltip />
                            <Bar dataKey="r2" radius={[4, 4, 0, 0]}>
                                {chartData.map((entry, index) => (
                                    <Cell key={`cell-r2-${index}`} fill={entry.color} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Cross-Algorithm Summary Matrix */}
            <h4 style={{ fontSize: "1.05rem", fontWeight: "700", marginBottom: "12px", color: "var(--text)" }}>
                Feature Selection Matrix
            </h4>
            <div style={{ overflowX: "auto" }}>
                <table className="ftable">
                    <thead>
                        <tr>
                            <th>Algorithm</th>
                            <th>Paradigm</th>
                            <th>Execution Time</th>
                            <th>R&amp;D Spend</th>
                            <th>Marketing Spend</th>
                            <th>State_NY</th>
                            <th>State_FL</th>
                            <th>Admin</th>
                            <th>Selected Subset</th>
                        </tr>
                    </thead>
                    <tbody>
                        {ALGO_COMPARISON.map(a => {
                            const has = (feat: string) => a.features.includes(feat);
                            return (
                                <tr key={a.name}>
                                    <td style={{ fontWeight: "700" }}>
                                        <span style={{ color: a.color }}>●</span> {a.name}
                                    </td>
                                    <td>
                                        <span className="pill">{a.category}</span>
                                    </td>
                                    <td style={{ fontFamily: "monospace", fontSize: "0.85rem" }}>
                                        {a.complexity.split("·")[1]?.trim() || a.complexity}
                                    </td>
                                    <td style={{ color: has("R&D Spend") ? "#10b981" : "#64748b", fontWeight: has("R&D Spend") ? "700" : "normal" }}>
                                        {has("R&D Spend") ? "✓ Retained" : "— Dropped"}
                                    </td>
                                    <td style={{ color: has("Marketing Spend") ? "#10b981" : "#64748b", fontWeight: has("Marketing Spend") ? "700" : "normal" }}>
                                        {has("Marketing Spend") ? "✓ Retained" : "— Dropped"}
                                    </td>
                                    <td style={{ color: has("State_New York") ? "#10b981" : "#64748b", fontWeight: has("State_New York") ? "700" : "normal" }}>
                                        {has("State_New York") ? "✓ Retained" : "— Dropped"}
                                    </td>
                                    <td style={{ color: has("State_Florida") ? "#10b981" : "#64748b", fontWeight: has("State_Florida") ? "700" : "normal" }}>
                                        {has("State_Florida") ? "✓ Retained" : "— Dropped"}
                                    </td>
                                    <td style={{ color: has("Administration") ? "#10b981" : "#64748b", fontWeight: has("Administration") ? "700" : "normal" }}>
                                        {has("Administration") ? "✓ Retained" : "— Dropped"}
                                    </td>
                                    <td>
                                        <span className={a.features.length === 2 ? "badge-best" : "pill"}>
                                            {a.features.length} Features
                                        </span>
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>

            <div style={{ marginTop: "16px", padding: "12px 16px", borderRadius: "8px", background: "rgba(59, 130, 246, 0.08)", borderLeft: "4px solid #3b82f6" }}>
                <strong style={{ color: "#60a5fa" }}>💡 Empirical Insight:</strong> 
                <strong> R&amp;D Spend</strong> is universally selected as the #1 predictor by all 5 algorithms. 
                Both <strong>Forward SFS</strong> and <strong>Exhaustive Search</strong> identify <code>[R&amp;D Spend, Marketing Spend]</code> as the global optimal subset (RMSE: <strong>$8,198.80</strong>, R²: <strong>0.9474</strong>). 
                <strong>Lasso</strong> favors extreme sparsity (1 feature), while <strong>ElasticNet</strong> and <strong>RFE</strong> retain additional correlated predictors (Administration / State), trading slight variance for group retention.
            </div>
        </div>
    );
}
