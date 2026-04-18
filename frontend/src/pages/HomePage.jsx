import React from 'react';
import { Activity, FileText, Cpu } from 'lucide-react';

const HomePage = ({ onNavigate }) => (
    <main className="page-shell">
        <div className="home-shell">

            {/* ── Hero ── */}
            <section className="home-hero">
                <div className="home-badge">Clinical Research Prototype // 2026</div>

                <div>
                    <h1 className="home-title">
                        AtrionNet <span>.</span>
                    </h1>
                    <h2 className="home-subtitle">
                        High-Grade AV Block Detection<br />
                        <strong>Powered by AtrionNet Fusion.</strong>
                    </h2>
                </div>

                <blockquote className="home-quote">
                    "An explainable neural framework engineered for the automated identification
                    of Atrial-Ventricular dissociation in 10-second ECG rhythms."
                </blockquote>

                <div className="home-actions">
                    <button className="btn-primary" onClick={() => onNavigate('analysis')} aria-label="Launch Analytics Dashboard">
                        <Activity style={{ width: 18, height: 18 }} aria-hidden="true" />
                        Launch Analytics
                    </button>
                    <button className="btn-outline" onClick={() => onNavigate('instructions')} aria-label="View Application Logic">
                        <FileText style={{ width: 18, height: 18 }} aria-hidden="true" />
                        View Logic
                    </button>
                </div>
            </section>

            {/* ── Technical Card ── */}
            <aside>
                <div className="home-card">
                    <div className="home-card-header">
                        <div className="home-card-icon" aria-hidden="true">
                            <Cpu style={{ width: 26, height: 26 }} />
                        </div>
                        <div>
                            <p className="home-card-label">Core Engine</p>
                            <p className="home-card-name">AtrionNet v1.0</p>
                        </div>
                    </div>

                    <div className="home-feature">
                        <span className="home-feature-scope">Research Scope</span>
                        <p className="home-feature-title">Multi-Task Attention Fusion</p>
                        <p className="home-feature-desc">
                            Parallel processing of wave boundaries and global classification labels
                            via a shared deep 1D backbone.
                        </p>
                    </div>

                    <hr className="home-divider" />

                    <div className="home-feature">
                        <span className="home-feature-scope">Research Scope</span>
                        <p className="home-feature-title">LUDB &amp; PTB-XL Optimized</p>
                        <p className="home-feature-desc">
                            Trained on cardiology-annotated datasets for 99.2% recall on QRS
                            landmarks and precise P-wave localization.
                        </p>
                    </div>

                    <hr className="home-divider" />

                    <div className="home-stats">
                        <div className="home-stat">
                            <p className="home-stat-label">Confidence</p>
                            <p className="home-stat-value" style={{ color: 'var(--text-primary)' }}>94.8%</p>
                        </div>
                        <div className="home-stat">
                            <p className="home-stat-label">Dice Score</p>
                            <p className="home-stat-value" style={{ color: 'var(--brand)' }}>0.82</p>
                        </div>
                    </div>
                </div>
            </aside>
        </div>
    </main>
);

export default HomePage;
