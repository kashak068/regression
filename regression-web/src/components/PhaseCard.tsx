import React from "react";
interface PhaseCardProps {
    num: string;
    name: string;
    desc: string;
}

export default function PhaseCard({ num, name, desc }: PhaseCardProps) {
    return (
        <div className="phase-card">
            <div className="phase-num">{num}</div>
            <div className="phase-name">{name}</div>
            <div className="phase-desc">{desc}</div>
        </div>
    );
}
