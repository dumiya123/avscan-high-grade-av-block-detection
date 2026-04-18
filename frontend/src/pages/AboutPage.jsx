import React from 'react';
import { Target, AlertCircle, Cpu, ShieldCheck, User } from 'lucide-react';

const aboutSections = [
    {
        id: 'vision',
        icon: Target,
        title: 'Vision of the Project',
        content: 'To empower cardiologists with a high-precision, explainable neural framework that automates the early detection of High-Grade Atrial-Ventricular (AV) blocks, ensuring faster intervention and improved patient clinical outcomes.',
    },
    {
        id: 'problem',
        icon: AlertCircle,
        title: 'The Problem',
        content: 'Manual identification of AV dissociation in complex ECG rhythms requires high expertise and is prone to human error, specially in lengthy records. Traditional automated systems often lack the clinical explainability required for high-stakes healthcare decisions.',
    },
    {
        id: 'solution',
        icon: Cpu,
        title: 'The Solution',
        content: 'AtrionNet utilizes a Multi-Task Attention Fusion architecture to simultaneously localize wave landmarks and classify rhythmic patterns. It is specifically engineered for dissociated P-waves detection, providing clear visual evidence of AV dissociation to bridge the gap between AI performance and clinical trust.',
    },
    {
        id: 'features',
        icon: ShieldCheck,
        title: 'Key Features',
        content: 'Automated 12-lead signal classification, precise dissociated P-waves detection, QRS complex localization, explainable AI (XAI) insights with clinical rationale, and automated PDF clinical report generation for laboratory documentation.',
    },
    {
        id: 'developer',
        icon: User,
        title: 'About Developer',
        content: 'This application is developed as part of a final-year research project by a Clinical AI researcher dedicated to advancing computational diagnostics and medical imaging solutions for modern healthcare ecosystems.',
    },
];

const AboutPage = () => (
    <div className="page-shell">
        <div className="instr-shell">
            <header className="instr-header">
                <h1 className="instr-title">About the AtrionNet System</h1>
                <p className="instr-subtitle">
                    A clinical-grade research framework for Explainable ECG Signal Analysis.
                </p>
            </header>

            <div className="instr-steps">
                {aboutSections.map(({ id, icon: Icon, title, content }) => (
                    <div className="instr-step" key={id}>
                        <div className="instr-step-num" style={{ fontSize: '18px' }}>
                             <Icon style={{ width: 20, height: 20 }} />
                        </div>
                        <div className="instr-step-body">
                            <div className="instr-step-title">
                                {title}
                            </div>
                            <p className="instr-step-desc" style={{ fontSize: '14px', marginTop: '4px' }}>
                                {content}
                            </p>
                        </div>
                    </div>
                ))}
            </div>

            <footer className="instr-footer">
                <p className="system-status-label" style={{ color: 'var(--text-muted)' }}>
                    AtrionNet Research Prototype &bull; Version 1.0 &bull; 2026
                </p>
            </footer>
        </div>
    </div>
);

export default AboutPage;
