import React from "react";
import { PHASES, METRICS, FS_RESULTS, ACTUAL_VS_PRED } from "@/lib/data";
import MetricCard from "@/components/MetricCard";
import PhaseCard from "@/components/PhaseCard";
import FeatureTable from "@/components/FeatureTable";
import PredictorForm from "@/components/PredictorForm";
import LineChartRMSE from "@/components/LineChartRMSE";
import LineChartR2 from "@/components/LineChartR2";
import ScatterPlot from "@/components/ScatterPlot";
import AlgorithmComparison from "@/components/AlgorithmComparison";

export default function Home() {
  return (
    <main className="page">
      {/* HERO */}
      <section className="hero">
        <div className="hero-badge">Machine Learning Demo</div>
        <h1>50 Startups – Regression Dashboard</h1>
        <p>Explore a Multiple Linear Regression model built with the CRISP‑DM methodology, compare 5 Feature Selection algorithms, and predict profit for any startup.</p>
        <div className="hero-links">
          <a href="#predictor" className="btn-primary">Live Profit Predictor</a>
          <a href="#comparison" className="btn-outline">5 Algorithms Compared</a>
          <a href="#features" className="btn-outline">Forward Selection</a>
        </div>
      </section>

      {/* METRICS */}
      <section className="section">
        <h2 className="section-title"><span className="icon">📊</span>Model Evaluation Metrics</h2>
        <div className="grid-4">
          {METRICS.map(m => (
            <MetricCard key={m.label} label={m.label} value={m.value} sub={m.sub} color={m.color} />
          ))}
        </div>
      </section>

      {/* CRISP‑DM PHASES */}
      <section className="section">
        <h2 className="section-title"><span className="icon">🧭</span>CRISP‑DM Process</h2>
        <div className="phase-grid">
          {PHASES.map(p => (
            <PhaseCard key={p.num} num={p.num} name={p.name} desc={p.desc} />
          ))}
        </div>
      </section>

      {/* PREDICTOR */}
      <section className="section" id="predictor">
        <h2 className="section-title"><span className="icon">💡</span>Live Profit Predictor</h2>
        <div className="predictor">
          <PredictorForm />
        </div>
      </section>

      {/* FEATURE SELECTION */}
      <section className="section" id="features">
        <h2 className="section-title"><span className="icon">🔎</span>Forward Feature Selection</h2>
        <FeatureTable data={FS_RESULTS} />
        <div className="grid-2" style={{ marginTop: "32px" }}>
          <LineChartRMSE data={FS_RESULTS} />
          <LineChartR2 data={FS_RESULTS} />
        </div>
        <div style={{ marginTop: "40px" }}>
          <ScatterPlot data={ACTUAL_VS_PRED} />
        </div>
      </section>

      {/* 5 ALGORITHMS BENCHMARK */}
      <section className="section" id="comparison">
        <h2 className="section-title"><span className="icon">⚔️</span>5 Feature Selection Algorithms Compared</h2>
        <AlgorithmComparison />
      </section>
    </main>
  );
}
