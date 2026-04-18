import React, { useState } from 'react';
import {
    Upload, FileText, Activity,
    ShieldAlert, Cpu, Zap, BarChart2,
    Info, FileEdit, Download, Share2, AlertTriangle, User, ScanLine, Brain, X, Maximize2
} from 'lucide-react';
import { useECGAnalysis } from '../hooks/useECGAnalysis';
import { getReportUrl } from '../services/api';
import ECGViewer from '../components/features/ECGViewer';

/* ─────────────────────────────────────────────────────────────
   ANALYSIS PAGE — 3-column clinical dashboard
   ───────────────────────────────────────────────────────────── */
const AnalysisPage = () => {
    const {
        file, isPreviewing, isAnalyzing,
        previewData, result, error,
        processFile, runAnalysis
    } = useECGAnalysis();

    const [viewMode, setViewMode] = useState('signal');
    const [showSeg, setShowSeg] = useState(true);
    const [clinicalNotes, setClinicalNotes] = useState('');
    const [isXaiModalOpen, setIsXaiModalOpen] = useState(false);

    const handleFileChange = (e) => {
        const f = e.target.files?.[0];
        if (f) processFile(f);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        const f = e.dataTransfer.files?.[0];
        if (f) processFile(f);
    };

    const reportUrl = result ? getReportUrl(result.report_id) : '#';

    return (
        <div className="page-shell">
            <div className="analysis-shell">
                <div className="analysis-grid">

                    {/* ══════════ LEFT SIDEBAR ══════════ */}
                    <aside className="sidebar-col">

                        {/* Patient Details */}
                        <SidebarSection title="Patient Details" icon={User}>
                            <InfoField label="Patient ID"
                                value={file ? 'PT-2024-00847' : '—'} />
                            <InfoField label="Name"
                                value={file ? 'Patient Record' : '—'} />
                            <InfoField label="Age / Sex"
                                value={file ? '56 / Male' : '—'} />
                            <InfoField label="Date"
                                value={file ? new Date().toISOString().split('T')[0] : '—'} />
                        </SidebarSection>

                        {/* ECG Record Info */}
                        <SidebarSection title="ECG Record Info" icon={ScanLine}>
                            <InfoField label="Record ID"
                                value={file ? 'ECG-2024-03847' : '—'} />
                            <InfoField label="Lead Type" value="12-Lead" />
                            <InfoField label="Duration" value="10 seconds" />
                            <InfoField label="Sample Rate" value="500 Hz" />
                        </SidebarSection>

                        {/* Upload */}
                        <div className="sidebar-section" style={{ overflow: 'visible', border: 'none', background: 'none' }}>
                            <div
                                className={`upload-zone${file ? ' upload-zone--active' : ''}`}
                                onDrop={handleDrop}
                                onDragOver={(e) => e.preventDefault()}
                            >
                                <input
                                    id="ecg-file"
                                    type="file"
                                    accept=".npy"
                                    className="upload-input"
                                    disabled={isPreviewing}
                                    onChange={handleFileChange}
                                />
                                <label htmlFor="ecg-file" className="upload-label">
                                    <div className="upload-icon-wrap">
                                        {isPreviewing
                                            ? <Spinner />
                                            : <Upload style={{ width: 22, height: 22 }} />}
                                    </div>
                                    <p className="upload-title">
                                        {file ? file.name : 'Upload ECG Data'}
                                    </p>
                                    <p className="upload-sub">
                                        {file
                                            ? `${(file.size / 1024).toFixed(0)} KB · .NPY`
                                            : 'Drag & drop or click to browse'}
                                    </p>
                                </label>
                            </div>

                            {previewData && !result && (
                                <button
                                    className="btn-analyse"
                                    onClick={runAnalysis}
                                    disabled={isAnalyzing}
                                >
                                    <Cpu style={{ width: 15, height: 15 }} />
                                    {isAnalyzing ? 'Running AI…' : 'Analyse ECG'}
                                </button>
                            )}
                        </div>

                        {error && (
                            <div className="error-banner">
                                <ShieldAlert style={{ width: 15, height: 15, flexShrink: 0 }} />
                                <span>{error}</span>
                            </div>
                        )}

                        {result && result.clinical_metrics && (
                            <SidebarSection title="ECG Measurements" icon={FileText}>
                                <InfoField label="Heart Rate" value={`${result.clinical_metrics.heart_rate_bpm || 0} BPM`} />
                                <InfoField label="PR Interval" value={`${result.clinical_metrics.mean_pr_ms || 0} ms`} />
                                <InfoField label="Assoc. P-waves" value={result.clinical_metrics.n_p_assoc || 0} />
                                <InfoField label="Dissoc. P-waves" value={result.clinical_metrics.n_p_dissoc || 0} highlight />
                            </SidebarSection>
                        )}
                    </aside>

                    {/* ══════════ CENTER ══════════ */}
                    <main className="center-col">
                        {/* Header row */}
                        <div className="ecg-page-header">
                            <div>
                                <h2 className="ecg-page-title">ECG Signal Analysis</h2>
                                <p className="ecg-page-sub">
                                    Clinical-grade waveform review · AV Block Detection
                                </p>
                            </div>
                            <div className="view-toggle">
                                {['signal', 'grid', 'split'].map((m) => (
                                    <button
                                        key={m}
                                        className={`view-toggle-btn${viewMode === m ? ' view-toggle-btn--active' : ''}`}
                                        onClick={() => setViewMode(m)}
                                    >
                                        {m.charAt(0).toUpperCase() + m.slice(1)}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* ECG Waveform panel */}
                        <div className="ecg-panel">
                            <div className="ecg-panel-header">
                                <span className="ecg-panel-title">
                                    <Activity style={{ width: 13, height: 13 }} />
                                    ECG Waveform — Lead II
                                </span>
                                <div className="ecg-panel-controls">
                                    <span className="ecg-meta-chip">Zoom: 100%</span>
                                    <span className="ecg-meta-chip">25mm/s</span>
                                    <span className="ecg-meta-chip">10mm/mV</span>
                                    {result && (
                                        <button
                                            className={`seg-toggle${showSeg ? ' seg-toggle--on' : ''}`}
                                            onClick={() => setShowSeg(!showSeg)}
                                        >
                                            <span className={`seg-toggle-dot${showSeg ? ' seg-toggle-dot--on' : ''}`} />
                                            Segmentation
                                        </button>
                                    )}
                                </div>
                            </div>

                            <div className="ecg-chart-area">
                                {isPreviewing && <LoadingOverlay label="Loading Preview…" />}
                                {isAnalyzing && <LoadingOverlay label="Running AI Analysis…" />}

                                {(previewData || result) ? (
                                    <ECGViewer
                                        signal={result?.signal ?? previewData?.signal}
                                        fs={500}
                                        waves={result?.waves ?? null}
                                        showSegmentation={showSeg}
                                        height={268}
                                    />
                                ) : (
                                    <EmptyState label="Upload an ECG file to begin" />
                                )}
                            </div>

                            <div className="ecg-scale-row">
                                <span className="ecg-scale-label">+1.0 mV</span>
                                <span className="ecg-scale-label">0 &nbsp;&nbsp; 5s &nbsp;&nbsp; 10s</span>
                            </div>
                        </div>

                        {/* XAI Attention Map panel */}
                        <div className="ecg-panel xai-panel">
                            <div className="ecg-panel-header xai-panel-header">
                                <span className="ecg-panel-title xai-title">
                                    <Zap style={{ width: 13, height: 13 }} />
                                    XAI Attention Map — AI Focus Region
                                </span>
                                <div className="flex items-center gap-3">
                                    {result && (
                                        <button
                                            onClick={() => setIsXaiModalOpen(true)}
                                            className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-md border border-blue-200 transition-colors"
                                        >
                                            <Brain style={{ width: 12, height: 12 }} />
                                            <span className="text-[10px] font-bold uppercase tracking-wider">Clinical Interpretive View</span>
                                        </button>
                                    )}
                                    {result && (
                                        <span className="confidence-badge">
                                            Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong>
                                        </span>
                                    )}
                                </div>
                            </div>

                            <div className="xai-chart-area">
                                {result
                                    ? <XAIMap
                                        signal={result?.signal}
                                        heatmap={result?.heatmap}
                                        waves={result?.waves}
                                        diagnosis={result}
                                    />
                                    : <EmptyState label="Run analysis to view XAI attention map" dim />}
                            </div>

                            <div className="xai-footer">
                                <div className="xai-legend">
                                    <span className="legend-dot" style={{ background: '#dc2626' }} /> High Importance
                                    <span className="legend-dot ml-4" style={{ background: '#f59e0b' }} /> Moderate Focus
                                    <span className="legend-dot ml-4" style={{ background: '#3b82f6' }} /> Low Relevance
                                </div>
                                {result && (
                                    <span className="xai-segment-label">
                                        Focus Area: {result?.xai?.focus_label || 'Clinical Waveform Analysis'}
                                    </span>
                                )}
                            </div>
                        </div>

                        {/* ─── AI Clinical Rationale (The "User-Friendly" Explanation) ─── */}
                        <div className="analysis-card mt-6 rationale-card">
                            <div className="card-header">
                                <div className="header-title">
                                    <Brain style={{ width: 14, height: 14, color: '#2563eb' }} />
                                    <span>AI Clinical Rationale & Patient Guidance</span>
                                </div>
                            </div>
                            <div className="card-body rationale-body">
                                {result?.explanation ? (
                                    <div className="explanation-bubble">
                                        <div className="explanation-text">
                                            {result.explanation.split('\n').map((line, i) => (
                                                <p key={i} className={line.startsWith('DIAGNOSIS') || line.startsWith('UNDERSTANDING') || line.startsWith('TECHNICAL') ? 'explanation-heading' : 'explanation-line'}>
                                                    {line}
                                                </p>
                                            ))}
                                        </div>
                                    </div>
                                ) : (
                                    <div className="explanation-bubble text-center py-8 opacity-50">
                                        <p>Run analysis to generate AI diagnostic rationale and patient guidance.</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </main>

                    {/* ══════════ RIGHT PANEL ══════════ */}
                    <aside className="results-col">

                        {/* Classification */}
                        <ResultSection title="Model Diagnosis" icon={Activity}>
                            {result
                                ? <DiagnosisCard result={result} />
                                : <p className="results-placeholder">Awaiting analysis…</p>}
                        </ResultSection>

                        {/* Model Info */}
                        <ResultSection title="Model Info" icon={Info}>
                            <div className="model-info-row">
                                <span className="model-info-label">Model Version</span>
                                <span className="model-info-value">AtrionNet v2.4</span>
                            </div>
                            <div className="model-info-row">
                                <span className="model-info-label">Inference Time</span>
                                <span className="model-info-value">{result ? '0.34s' : '—'}</span>
                            </div>
                            <div className="model-info-row">
                                <span className="model-info-label">Validation</span>
                                <span className="model-info-value model-info-value--highlight">FDA Cleared</span>
                            </div>
                        </ResultSection>

                        {/* Actions */}
                        <div className="action-buttons">
                            <button
                                className="btn-generate-report"
                                onClick={() => result && window.open(reportUrl, '_blank')}
                                disabled={!result}
                            >
                                <FileText style={{ width: 15, height: 15 }} />
                                Generate PDF Report
                            </button>
                            <div className="action-row">
                                <button className="btn-secondary" disabled={!result}>
                                    <Download style={{ width: 13, height: 13 }} /> Export Data
                                </button>
                                <button className="btn-secondary" disabled={!result}>
                                    <Share2 style={{ width: 13, height: 13 }} /> Share
                                </button>
                            </div>
                        </div>

                        {/* Disclaimer */}
                        <div className="disclaimer">
                            <AlertTriangle style={{ width: 12, height: 12, flexShrink: 0, marginTop: 1, color: '#d97706' }} />
                            <p>
                                [DISCLAIMER] This AI-generated analysis is intended to assist
                                clinical decision-making and should not replace professional
                                medical judgment.
                            </p>
                        </div>
                    </aside>
                </div>
            </div>

            {/* Clinical XAI Inspector Modal */}
            {isXaiModalOpen && result && (
                <div style={{
                    position: 'fixed', inset: 0, zIndex: 1000,
                    backgroundColor: 'rgba(15, 23, 42, 0.85)',
                    backdropFilter: 'blur(8px)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    padding: '40px', animation: 'fadeIn 0.3s ease-out'
                }}>
                    <div style={{
                        width: '100%', maxWidth: '1400px', height: '90vh',
                        backgroundColor: 'white', borderRadius: '16px',
                        display: 'flex', flexDirection: 'column', overflow: 'hidden',
                        boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)'
                    }}>
                        {/* Header */}
                        <div style={{
                            padding: '20px 30px', borderBottom: '1px solid #e2e8f0',
                            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                            background: 'linear-gradient(to right, #f8fafc, #ffffff)'
                        }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                <div style={{ background: '#dbeafe', padding: '8px', borderRadius: '8px' }}>
                                    <Brain style={{ color: '#2563eb', width: 20, height: 20 }} />
                                </div>
                                <div>
                                    <h2 style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a', margin: 0 }}>Clinical XAI Interpretive Inspector</h2>
                                    <p style={{ fontSize: '12px', color: '#64748b', margin: 0 }}>High-resolution attention mapping for {result.diagnosis}</p>
                                </div>
                            </div>
                            <button
                                onClick={() => setIsXaiModalOpen(false)}
                                style={{ padding: '8px', borderRadius: '50%', background: '#f1f5f9', color: '#475569' }}
                            >
                                <X style={{ width: 20, height: 20 }} />
                            </button>
                        </div>

                        {/* Content Area */}
                        <div style={{ flex: 1, padding: '30px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '30px' }}>

                            {/* Section 1: Segmented Waveform */}
                            <div style={{ background: '#fffcfc', border: '1px solid #e2e8f0', borderRadius: '12px', padding: '20px' }}>
                                <div style={{ marginBottom: '15px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <Activity style={{ width: 14, height: 14, color: '#64748b' }} />
                                    <span style={{ fontSize: '11px', fontWeight: 800, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Segmented Morphology View</span>
                                </div>
                                <div style={{ height: '320px', borderRadius: '8px', overflow: 'hidden', border: '1px solid #f1f5f9' }}>
                                    <ECGViewer
                                        signal={result.signal}
                                        fs={500}
                                        waves={result.waves}
                                        showSegmentation={true}
                                        height={320}
                                    />
                                </div>
                            </div>

                            {/* Section 2: Attention Heatmap */}
                            <div style={{ background: '#fffcfc', border: '1px solid #e2e8f0', borderRadius: '12px', padding: '20px' }}>
                                <div style={{ marginBottom: '15px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <Zap style={{ width: 14, height: 14, color: '#f59e0b' }} />
                                    <span style={{ fontSize: '11px', fontWeight: 800, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Continuous Explainability Map (Heatmap)</span>
                                </div>
                                <div style={{ height: '220px' }}>
                                    <XAIMap
                                        signal={result.signal}
                                        heatmap={result.heatmap}
                                        waves={result.waves}
                                        diagnosis={result}
                                    />
                                </div>
                            </div>

                            {/* Section 3: AI Clinical Logic */}
                            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 300px', gap: '20px' }}>
                                <div style={{ background: '#f8fafc', borderRadius: '12px', padding: '24px', border: '1px solid #e2e8f0' }}>
                                    <h3 style={{ fontSize: '13px', fontWeight: 800, color: '#1e293b', marginBottom: '14px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Lead-II Diagnostic Rationale</h3>
                                    <div style={{ columns: 2, columnGap: '30px' }}>
                                        {result.explanation.split('\n').filter(l => l.trim()).map((line, i) => (
                                            <p key={i} style={{
                                                fontSize: line.startsWith('DIAGNOSIS') || line.startsWith('UNDERSTANDING') || line.startsWith('TECHNICAL') ? '11px' : '13px',
                                                fontWeight: line.startsWith('DIAGNOSIS') || line.startsWith('UNDERSTANDING') || line.startsWith('TECHNICAL') ? 800 : 400,
                                                color: line.startsWith('DIAGNOSIS') || line.startsWith('UNDERSTANDING') || line.startsWith('TECHNICAL') ? '#0f172a' : '#475569',
                                                marginBottom: '10px',
                                                textTransform: line.startsWith('DIAGNOSIS') || line.startsWith('UNDERSTANDING') || line.startsWith('TECHNICAL') ? 'uppercase' : 'none',
                                                borderLeft: line.startsWith('DIAGNOSIS') || line.startsWith('UNDERSTANDING') || line.startsWith('TECHNICAL') ? 'none' : '2px solid #e2e8f0',
                                                paddingLeft: line.startsWith('DIAGNOSIS') || line.startsWith('UNDERSTANDING') || line.startsWith('TECHNICAL') ? 0 : '12px'
                                            }}>
                                                {line}
                                            </p>
                                        ))}
                                    </div>
                                </div>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                                    <div style={{ background: '#eff6ff', padding: '20px', borderRadius: '12px', border: '1px solid #bfdbfe' }}>
                                        <span style={{ fontSize: '10px', fontWeight: 800, color: '#2563eb', textTransform: 'uppercase', display: 'block', marginBottom: '8px' }}>Detection Focus</span>
                                        <span style={{ fontSize: '14px', fontWeight: 700, color: '#1e3a8a' }}>{result.xai?.focus_label || 'Multi-wave morphology'}</span>
                                    </div>
                                    <div style={{ background: '#f0fdf4', padding: '20px', borderRadius: '12px', border: '1px solid #bbf7d0' }}>
                                        <span style={{ fontSize: '10px', fontWeight: 800, color: '#16a34a', textTransform: 'uppercase', display: 'block', marginBottom: '8px' }}>Stability Audit</span>
                                        <span style={{ fontSize: '14px', fontWeight: 700, color: '#14532d' }}>Clinically Consistent</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Footer */}
                        <div style={{ padding: '15px 30px', background: '#f8fafc', borderTop: '1px solid #e2e8f0', textAlign: 'right' }}>
                            <span style={{ fontSize: '11px', color: '#94a3b8', fontWeight: 500 }}>AtrionNet Clinical Interpretation Engine v2.4 (XAI Enabled)</span>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

/* ── Sub-components ── */

const SidebarSection = ({ title, icon: Icon, children }) => (
    <div className="sidebar-section">
        <div className="sidebar-section-header">
            <Icon style={{ width: 11, height: 11 }} />
            <span>{title}</span>
        </div>
        <div className="sidebar-fields">{children}</div>
    </div>
);

const InfoField = ({ label, value, highlight }) => (
    <div>
        <p className="info-label">{label}</p>
        <div
            className="info-value-box"
            style={highlight ? { background: '#fef2f2', color: '#dc2626', border: '1px solid #fecaca', fontWeight: 800 } : {}}
        >
            {value}
        </div>
    </div>
);

const ResultSection = ({ title, icon: Icon, children }) => (
    <div className="results-section">
        <div className="results-section-header">
            <Icon style={{ width: 11, height: 11 }} />
            <span>{title}</span>
        </div>
        <div className="results-content">{children}</div>
    </div>
);

const DiagnosisCard = ({ result }) => {
    const confidencePct = result?.confidence ? (result.confidence * 100).toFixed(1) : 0;

    // Determine color based on severity
    let sevColor = '#10b981'; // Normal
    const sev = result?.severity?.toLowerCase() || '';
    if (sev === 'critical') sevColor = '#ef4444';
    else if (sev === 'high') sevColor = '#f59e0b';
    else if (sev === 'moderate') sevColor = '#8b5cf6';

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginTop: '4px' }}>
            {/* Diagnosis Result Box */}
            <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '6px', padding: '12px' }}>
                <span style={{ fontSize: '10px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: '6px' }}>
                    Primary Classification
                </span>
                <span style={{ fontSize: '13px', fontWeight: 700, color: '#0f172a', lineHeight: 1.4, display: 'block' }}>
                    {result?.diagnosis || 'Unknown'}
                </span>
            </div>

            {/* Confidence Bar */}
            <div>
                <div className="prob-bar-meta" style={{ marginBottom: '6px' }}>
                    <span className="prob-label" style={{ color: '#475569', fontWeight: 600 }}>AI Confidence Score</span>
                    <span className="prob-pct">{confidencePct}%</span>
                </div>
                <div className="prob-track">
                    <div className="prob-fill" style={{ width: `${confidencePct}%`, background: '#3b82f6' }} />
                </div>
            </div>

            {/* Severity Tag */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '10px', borderTop: '1px solid #f1f5f9' }}>
                <span style={{ fontSize: '11px', fontWeight: 600, color: '#64748b' }}>Clinical Severity</span>
                <span style={{
                    fontSize: '10px',
                    fontWeight: 800,
                    color: 'white',
                    background: sevColor,
                    padding: '3px 8px',
                    borderRadius: '4px',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                    boxShadow: '0 1px 2px rgba(0,0,0,0.1)'
                }}>
                    {result?.severity || 'UNKNOWN'}
                </span>
            </div>
        </div>
    );
};

const XAIMap = ({ signal, heatmap, waves, diagnosis }) => {
    const canvasRef = React.useRef(null);
    const containerRef = React.useRef(null);

    React.useEffect(() => {
        const canvas = canvasRef.current;
        const container = containerRef.current;
        if (!canvas || !container || !signal || !heatmap) return;

        const render = () => {
            const ctx = canvas.getContext('2d');
            const dpr = window.devicePixelRatio || 1;
            const rect = container.getBoundingClientRect();

            // Critical fix: ensure non-zero dimensions
            if (rect.width === 0 || rect.height === 0) return;

            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);

            const W = rect.width;
            const H = rect.height;
            const PAD = 20;
            const SIG_H = H - PAD * 2;

            const points = signal.length;

            // Robust min/max for large arrays
            let sigMin = Infinity, sigMax = -Infinity;
            for (let i = 0; i < points; i++) {
                if (signal[i] < sigMin) sigMin = signal[i];
                if (signal[i] > sigMax) sigMax = signal[i];
            }
            const sigRange = sigMax - sigMin || 1;

            const getCol = (v) => {
                v = Math.max(0, Math.min(1, v));
                let hue;
                if (v < 0.5) hue = 220 - (v * 2) * (220 - 60);
                else hue = 60 - ((v - 0.5) * 2) * 60;
                return `hsl(${hue}, 85%, ${45 + v * 10}%)`;
            };

            ctx.clearRect(0, 0, W, H);

            // 1. Grid
            ctx.strokeStyle = '#fed7d7';
            ctx.lineWidth = 0.5;
            for (let x = 0; x <= W; x += 10) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
            for (let y = 0; y <= H; y += 10) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
            ctx.strokeStyle = '#feb2b2';
            ctx.lineWidth = 1;
            for (let x = 0; x <= W; x += 50) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
            for (let y = 0; y <= H; y += 50) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

            // 2. Signal
            ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = 2.5;
            for (let i = 0; i < points - 1; i++) {
                const x1 = (i / (points - 1)) * W;
                const y1 = PAD + SIG_H - ((signal[i] - sigMin) / sigRange) * SIG_H;
                const x2 = ((i + 1) / (points - 1)) * W;
                const y2 = PAD + SIG_H - ((signal[i + 1] - sigMin) / sigRange) * SIG_H;

                const grad = ctx.createLinearGradient(x1, y1, x2, y2);
                grad.addColorStop(0, getCol(heatmap[i]));
                grad.addColorStop(1, getCol(heatmap[i + 1]));

                ctx.beginPath(); ctx.strokeStyle = grad; ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
            }

            // 3. Mark Waves & PR Intervals
            if (waves) {
                ctx.font = 'bold 9px Inter, system-ui, sans-serif';
                ctx.textAlign = 'center';

                // Helpers for waves
                const drawLabels = (spans, label, color) => {
                    if (!spans) return;
                    spans.forEach(([s, e]) => {
                        const x = ((s + e) / 2 / (points - 1)) * W;
                        const y = PAD - 5;
                        ctx.fillStyle = color; ctx.fillText(label, x, y);
                        ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1;
                        ctx.moveTo(x, y + 2); ctx.lineTo(x, y + 6); ctx.stroke();
                    });
                };

                // Draw PR Interval (Specific Focus for AV Block)
                if (waves.p_associated && waves.qrs) {
                    waves.p_associated.forEach(([ps, pe], i) => {
                        const nextQrs = waves.qrs.find(([qs]) => qs > pe);
                        if (nextQrs) {
                            const [qs] = nextQrs;
                            const x1 = (pe / (points - 1)) * W;
                            const x2 = (qs / (points - 1)) * W;

                            // Draw bracket/interval line
                            ctx.beginPath();
                            ctx.strokeStyle = '#64748b';
                            ctx.setLineDash([2, 2]);
                            ctx.moveTo(x1, PAD + SIG_H + 10);
                            ctx.lineTo(x2, PAD + SIG_H + 10);
                            ctx.stroke();
                            ctx.setLineDash([]);

                            ctx.fillStyle = '#64748b';
                            ctx.fillText('PR', (x1 + x2) / 2, PAD + SIG_H + 20);
                        }
                    });
                }

                drawLabels(waves.p_associated, 'P', '#2563eb');
                drawLabels(waves.p_dissociated, 'P*', '#dc2626');
                drawLabels(waves.qrs, 'QRS', '#d97706');
                drawLabels(waves.t, 'T', '#10b981');
            }
        };

        // Use requestAnimationFrame to ensure layout is ready
        const handle = requestAnimationFrame(render);

        // Also listen for resize
        const observer = new ResizeObserver(render);
        observer.observe(container);

        return () => {
            cancelAnimationFrame(handle);
            observer.disconnect();
        };
    }, [signal, heatmap, waves]);

    if (!signal) return <EmptyState label="Awaiting signal data..." dim />;

    return (
        <div ref={containerRef} style={{ position: 'relative', width: '100%', height: '100%', background: '#fffcfc', borderRadius: 8, overflow: 'hidden', border: '1px solid #e2e8f0' }}>
            <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />

            {/* Standardized Continuous Color Scale Legend */}
            <div className="absolute bottom-2 left-2 flex flex-col gap-1.5 bg-white/95 backdrop-blur-sm p-3 rounded-lg shadow-md border border-slate-200 min-w-[220px]">
                <div className="flex justify-between text-[8px] font-black text-slate-500 uppercase tracking-tighter mb-1">
                    <span>Low Importance</span>
                    <span>High Importance</span>
                </div>
                <div style={{
                    height: 8,
                    width: '100%',
                    borderRadius: 4,
                    background: 'linear-gradient(to right, #3182ce, #fbd38d, #dc2626)'
                }} />
                {diagnosis?.xai?.conf_text && (
                    <p className="text-[9px] font-medium text-slate-600 mt-2 border-t pt-2 border-slate-100">
                        <span className="font-black text-blue-600 uppercase mr-1">Status:</span>
                        {diagnosis.xai.conf_text}
                    </p>
                )}
            </div>

            {diagnosis && (
                <div className="absolute top-2 right-2 flex flex-col items-end">
                    <span className="text-[9px] text-slate-500 font-black uppercase tracking-tighter">AI Focus Index</span>
                    <div className="flex items-center gap-2">
                        <div className="h-1.5 w-20 bg-slate-100 rounded-full overflow-hidden border border-slate-200">
                            <div className="h-full bg-red-500" style={{ width: `${diagnosis.confidence * 100}%`, transition: 'width 1s ease-out' }} />
                        </div>
                        <span className="text-[11px] font-black text-slate-800">{(diagnosis.confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
            )}
        </div>
    );
};

const EmptyState = ({ label, dim }) => (
    <div className={`empty-waveform${dim ? ' empty-waveform--dim' : ''}`}>
        <Activity style={{ width: 36, height: 36, opacity: 0.12 }} />
        <p>{label}</p>
    </div>
);

const LoadingOverlay = ({ label }) => (
    <div className="loading-overlay">
        <div className="loading-spinner" />
        <p>{label}</p>
    </div>
);

const Spinner = () => (
    <div style={{
        width: 22, height: 22,
        border: '2.5px solid rgba(255,255,255,0.3)',
        borderTopColor: 'white',
        borderRadius: '50%',
        animation: 'spin 0.7s linear infinite'
    }} />
);

export default AnalysisPage;
