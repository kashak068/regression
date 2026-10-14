"use client";

import React from "react";
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

interface Props {
    data: { actual: number; predicted: number }[];
}

export default function ScatterPlot({ data }: Props) {
    const chartData = data.map((d, i) => ({ index: i + 1, actual: d.actual, predicted: d.predicted }));
    return (
        <div className="card">
            <h3 style={{ marginBottom: "8px", color: "var(--text)" }}>Actual vs Predicted Profit</h3>
            <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis type="number" dataKey="index" name="Sample" label={{ value: "Sample #", position: "insideBottom", offset: -5 }} />
                    <YAxis type="number" dataKey="actual" name="Actual" label={{ value: "Profit ($)", angle: -90, position: "insideLeft" }} />
                    <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                    <Scatter name="Actual" data={chartData} fill="var(--accent)" />
                    <Scatter name="Predicted" data={chartData} fill="var(--accent3)" />
                </ScatterChart>
            </ResponsiveContainer>
        </div>
    );
}
