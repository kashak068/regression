import React from "react";
import { FS_RESULTS } from "@/lib/data";

export default function FeatureTable({ data = FS_RESULTS }) {
    return (
        <div className="card">
            <table className="ftable">
                <thead>
                    <tr>
                        <th># Features</th>
                        <th>Selected Features</th>
                        <th>RMSE</th>
                        <th>R²</th>
                    </tr>
                </thead>
                <tbody>
                    {data.map(row => (
                        <tr key={row.n} className={row.n === 2 ? "best-row" : ""}>
                            <td>{row.n}</td>
                            <td>
                                <div className="feat-pills">
                                    {row.features.map(f => (
                                        <span key={f} className="pill">{f}</span>
                                    ))}
                                </div>
                                {row.n === 2 && <span className="badge-best">Best</span>}
                            </td>
                            <td>{row.rmse.toFixed(2)}</td>
                            <td>{row.r2.toFixed(4)}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
