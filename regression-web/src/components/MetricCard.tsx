import React from "react";
interface MetricCardProps {
    label: string;
    value: string;
    sub: string;
    color: string;
}

export default function MetricCard({ label, value, sub, color }: MetricCardProps) {
    return (
        <div className="metric-card" style={{ borderColor: color }}>
            <div className="glow" style={{ background: color }} />
            <div className="label">{label}</div>
            <div className="value" style={{ color }}>{value}</div>
            <div className="sub">{sub}</div>
        </div>
    );
}
