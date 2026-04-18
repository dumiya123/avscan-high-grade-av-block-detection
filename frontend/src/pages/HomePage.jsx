import React from 'react';
import { Activity, FileText, Cpu, ShieldCheck, Zap, BarChart3, ChevronRight } from 'lucide-react';

const HomePage = ({ onNavigate }) => (
    <main className="page-shell">
        <div className="home-shell">
            {/* ── Hero Section ── */}
            <section className="home-hero-section">
                <div className="home-hero-content">
                    <div className="home-badge">Clinical Research Prototype // 2026</div>
                    <h1 className="home-title" style={{ fontSize: 'clamp(48px, 6vw, 76px)', color: 'var(--text-primary)', fontWeight: 900 }}>
                        AtrionNet<span>.</span>
                    </h1>
                    <h2 className="home-subtitle" style={{ fontSize: 'clamp(20px, 2.5vw, 28px)', color: 'var(--text-secondary)', maxWidth: '600px', lineHeight: 1.3 }}>
                        Next-Generation <strong>A-V Block Detection</strong> engineered with Explainable AI.
                    </h2>
                    <p className="home-quote" style={{ fontStyle: 'normal', borderLeft: '4px solid var(--brand)', paddingLeft: '24px', fontSize: '18px', color: 'var(--text-secondary)', maxWidth: '550px' }}>
                        Empowering clinicians with anchor-free instance segmentation for precise cardiac landmark localization and diagnostic confidence.
                    </p>
                    <div className="home-actions" style={{ marginTop: '16px' }}>
                        <button className="btn-primary" style={{ padding: '14px 28px' }} onClick={() => onNavigate('analysis')} aria-label="Launch Dashboard">
                            <Activity style={{ width: 20, height: 20 }} />
                            Launch Dashboard
                        </button>
                        <button className="btn-outline" style={{ padding: '14px 28px' }} onClick={() => onNavigate('instructions')} aria-label="View Logic">
                            <FileText style={{ width: 20, height: 20 }} />
                            Technical Logic
                        </button>
                    </div>
                </div>
                <div className="home-hero-viz">
                    <img src="/assets/hero-viz.png" alt="AtrionNet AI Visualization" />
                </div>
            </section>

            {/* ── Stats Bar ── */}
            <div className="home-stats-bar">
                <div className="stat-item">
                    <span className="stat-item-value">94.8%</span>
                    <span className="stat-item-label">Detection Confidence</span>
                </div>
                <div className="stat-item">
                    <span className="stat-item-value">99.2%</span>
                    <span className="stat-item-label">Landmark Recall</span>
                </div>
                <div className="stat-item">
                    <span className="stat-item-value">0.82</span>
                    <span className="stat-item-label">Dice Coefficient</span>
                </div>
            </div>

            {/* ── Features Grid ── */}
            <section className="home-feature-grid">
                <div className="feature-card">
                    <div className="feature-card-icon">
                        <Cpu style={{ width: 28, height: 28 }} />
                    </div>
                    <h3 className="feature-card-title">Anchor-Free Fusion</h3>
                    <p className="feature-card-text">
                        A proprietary architecture that detects wave boundaries without morphological constraints, enabling detection of even the most subtle P-wave dissociations.
                    </p>
                </div>
                <div className="feature-card">
                    <div className="feature-card-icon">
                        <ShieldCheck style={{ width: 28, height: 28 }} />
                    </div>
                    <h3 className="feature-card-title">Clinically Interpretable</h3>
                    <p className="feature-card-text">
                        AtrionNet's "Attention Heatmap" identifies exactly where the model focuses, closing the gap between algorithmic prediction and clinical reasoning.
                    </p>
                </div>
                <div className="feature-card">
                    <div className="feature-card-icon">
                        <Zap style={{ width: 28, height: 28 }} />
                    </div>
                    <h3 className="feature-card-title">Real-time Inference</h3>
                    <p className="feature-card-text">
                        Optimized for low-latency analysis, providing immediate segmentation of 10-second ECG signals with deep contextual understanding of cardiac rhythms.
                    </p>
                </div>
                <div className="feature-card">
                    <div className="feature-card-icon">
                        <BarChart3 style={{ width: 28, height: 28 }} />
                    </div>
                    <h3 className="feature-card-title">Scientific Validation</h3>
                    <p className="feature-card-text">
                        Validated across diverse clinical datasets (LUDB, PTB-XL) to ensure robust performance across varied patient demographics and electrode placements.
                    </p>
                </div>
            </section>

            {/* ── CTA Section ── */}
            <section className="cta-section">
                <h2 className="cta-title">Ready to transform your ECG diagnostic workflow?</h2>
                <p className="cta-desc">
                    Join the clinical research community utilizing AtrionNet for automated, explainable cardiac analysis.
                </p>
                <button 
                  className="btn-cta-glow"
                  onClick={() => onNavigate('analysis')}
                >
                    Start Diagnostic Run
                    <ChevronRight style={{ width: 22, height: 22 }} />
                </button>
            </section>
        </div>
    </main>
);

export default HomePage;


