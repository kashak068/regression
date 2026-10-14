"use client";

import React, { useState } from "react";
import { predictProfit, State } from "@/lib/data";

export default function PredictorForm() {
    const [rdSpend, setRdSpend] = useState(0);
    const [admin, setAdmin] = useState(0);
    const [marketing, setMarketing] = useState(0);
    const [state, setState] = useState<State>("California");
    const [result, setResult] = useState<number | null>(null);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        const profit = predictProfit(rdSpend, admin, marketing, state);
        setResult(profit);
    };

    return (
        <form className="card" onSubmit={handleSubmit}>
            <div className="form-group">
                <label htmlFor="rdSpend">R&amp;D Spend ($)</label>
                <input id="rdSpend" type="number" value={rdSpend} onChange={e => setRdSpend(parseFloat(e.target.value) || 0)} />
            </div>
            <div className="form-group">
                <label htmlFor="admin">Administration ($)</label>
                <input id="admin" type="number" value={admin} onChange={e => setAdmin(parseFloat(e.target.value) || 0)} />
            </div>
            <div className="form-group">
                <label htmlFor="marketing">Marketing Spend ($)</label>
                <input id="marketing" type="number" value={marketing} onChange={e => setMarketing(parseFloat(e.target.value) || 0)} />
            </div>
            <div className="form-group">
                <label htmlFor="state">State</label>
                <select id="state" value={state} onChange={e => setState(e.target.value as State)}>
                    <option value="California">California</option>
                    <option value="Florida">Florida</option>
                    <option value="New York">New York</option>
                </select>
            </div>
            <button type="submit" className="btn-primary" style={{ width: "100%", marginTop: "12px" }}>Predict Profit</button>
            {result !== null && (
                <div className="result-box" style={{ marginTop: "24px" }}>
                    <div className="label-sm">Predicted Profit</div>
                    <div className="result-profit">${result.toFixed(2)}</div>
                    <div className="result-note">Based on the trained Linear Regression model.</div>
                </div>
            )}
        </form>
    );
}
