"use client";

import React from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

interface Props {
    data: { n: number; rmse: number; r2: number; features: string[] }[];
}

export default function LineChartRMSE({ data }: Props) {
    const chartData = data.map(d => ({ features: d.n, RMSE: Number(d.rmse.toFixed(2)) }));
    return (
        <div className="card-sm">
            <h3 style={{ marginBottom: "8px", color: "var(--text)" }}>RMSE vs # Features</h3>
            <ResponsiveContainer width="100%" height={260}>
                <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="features" label={{ value: "# Features", position: "insideBottom", offset: -5 }} />
                    <YAxis domain={[8000, 9200]} label={{ value: "RMSE", angle: -90, position: "insideLeft" }} />
                    <Tooltip />
                    <Line type="monotone" dataKey="RMSE" stroke="var(--accent)" strokeWidth={3} dot={{ r: 5 }} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}
