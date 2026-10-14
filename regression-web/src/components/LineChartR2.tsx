"use client";

import React from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

interface Props {
    data: { n: number; rmse: number; r2: number; features: string[] }[];
}

export default function LineChartR2({ data }: Props) {
    const chartData = data.map(d => ({ features: d.n, R2: Number(d.r2.toFixed(4)) }));
    return (
        <div className="card-sm">
            <h3 style={{ marginBottom: "8px", color: "var(--text)" }}>R² vs # Features</h3>
            <ResponsiveContainer width="100%" height={260}>
                <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="features" label={{ value: "# Features", position: "insideBottom", offset: -5 }} />
                    <YAxis domain={[0.92, 0.95]} label={{ value: "R²", angle: -90, position: "insideLeft" }} />
                    <Tooltip />
                    <Line type="monotone" dataKey="R2" stroke="var(--accent3)" strokeWidth={3} dot={{ r: 5 }} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}
