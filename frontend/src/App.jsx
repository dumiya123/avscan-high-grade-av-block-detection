import React, { useState, useEffect } from 'react';
import { Activity, Upload, FileText, CheckCircle, ShieldAlert, Cpu } from 'lucide-react';
import { analyzeECG, checkHealth, getReportUrl } from './services/api';
// Import our components later or build them inline for now to ensure one-shot success
import AnalysisResults from './components/AnalysisResults';
import ECGViewer from './components/ECGViewer';

const App = () => {
    const [view, setView] = useState('home');
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [backendStatus, setBackendStatus] = useState(false);
    const [showSegmentation, setShowSegmentation] = useState(true);

    useEffect(() => {
        checkHealth().then(setBackendStatus);
        const interval = setInterval(() => {
            checkHealth().then(setBackendStatus);
        }, 10000);
        return () => clearInterval(interval);
    }, []);

    const handleFileUpload = async (event) => {
        const uploadedFile = event.target.files?.[0];
        if (!uploadedFile) return;

        setFile(uploadedFile);
        setLoading(true);
        setResult(null);
        setView('analysis'); // Force view change on upload

        try {
            const data = await analyzeECG(uploadedFile);
            setResult(data);
        } catch (error) {
            console.error("Analysis failed", error);
            alert("Failed to analyze ECG. Please ensure the backend is running and the file is a valid .npy ECG sample.");
        } finally {
            setLoading(false);
        }
    };

    const renderHome = () => (
        <div className="flex-1 flex flex-col lg:flex-row items-center justify-center gap-12 px-6 lg:px-24 py-12 bg-[#fdfdfd]">
            {/* Left Column: Hero & Mission */}
            <div className="flex-1 max-w-2xl space-y-10 animate-in fade-in slide-in-from-left-12 duration-1000">
                <div className="space-y-4">
                    <div className="inline-flex items-center gap-2 px-3 py-1 bg-blue-50 border border-blue-100 rounded-full">
                        <div className="w-1.5 h-1.5 rounded-full bg-blue-600 animate-pulse"></div>
                        <span className="text-[10px] font-black text-blue-700 uppercase tracking-[0.2em]">Clinical Research Prototype // 2026</span>
                    </div>
                    <h2 className="text-5xl lg:text-7xl font-black text-slate-900 tracking-tighter leading-[0.9]">
                        AVScan <span className="text-blue-600">.</span>
                    </h2>
                    <h3 className="text-2xl lg:text-3xl font-bold text-slate-400 leading-tight">
                        High-Grade AV Block Detection <br />
                        <span className="text-slate-800">Powered by AtrionNet Fusion.</span>
                    </h3>
                </div>

                <div className="relative pl-8">
                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-blue-600 to-indigo-600 rounded-full"></div>
                    <p className="text-xl text-slate-600 leading-relaxed font-medium italic">
                        "An explainable neural framework engineered for the automated identification of Atrial-Ventricular dissociation in 10-second ECG rhythms."
                    </p>
                </div>

                {/* <p className="text-slate-500 leading-relaxed max-w-lg text-sm">
          Developed as a <strong>Senior Research Thesis</strong>, AVScan transitions from simple rhythm detection to high-fidelity wave segmentation. By identifying independent P-wave morphologies, the system provides a robust diagnostic pathway for complete heart blocks.
        </p> */}

                <div className="flex flex-wrap gap-4 pt-4">
                    <button
                        onClick={() => setView('analysis')}
                        className="group flex items-center gap-3 bg-slate-900 hover:bg-black text-white px-10 py-5 rounded-2xl font-bold transition-all shadow-[0_20px_40px_rgba(0,0,0,0.15)] active:scale-95"
                    >
                        <Activity className="w-6 h-6 text-blue-400" />
                        <span className="text-lg">Launch Analytics</span>
                    </button>

                    <button
                        onClick={() => setView('instructions')}
                        className="flex items-center gap-3 bg-white border-2 border-slate-100 hover:border-slate-300 text-slate-700 px-10 py-5 rounded-2xl font-bold transition-all active:scale-95"
                    >
                        <FileText className="w-5 h-5 text-slate-400" />
                        <span className="text-lg">View Logic</span>
                    </button>
                </div>
            </div>

            {/* Right Column: Technical "Glass" Module */}
            <div className="w-full lg:w-[440px] animate-in fade-in slide-in-from-right-12 duration-1000 delay-300">
                <div className="bg-white p-10 rounded-[3rem] border border-slate-200 shadow-[0_30px_60px_rgba(0,0,0,0.06)] relative group overflow-hidden">
                    <div className="absolute -top-12 -right-12 w-48 h-48 bg-blue-600/5 blur-[80px] rounded-full"></div>

                    <div className="flex items-center gap-4 mb-10">
                        <div className="w-14 h-14 bg-gradient-to-br from-blue-600 to-indigo-700 rounded-2xl flex items-center justify-center shadow-lg shadow-blue-100">
                            <Cpu className="w-8 h-8 text-white" />
                        </div>
                        <div>
                            <p className="text-[11px] font-black text-blue-600 tracking-[0.25em] uppercase">Core Engine</p>
                            <p className="text-lg font-black text-slate-800 tracking-tight">AtrionNet v1.0</p>
                        </div>
                    </div>

                    <div className="space-y-8">
                        <div className="space-y-3">
                            <p className="text-[10px] font-black text-slate-400 tracking-widest uppercase flex items-center gap-2">
                                <CheckCircle className="w-3 h-3 text-green-500" /> Research Scope
                            </p>
                            <p className="text-sm font-bold text-slate-700">Multi-Task Attention Fusion</p>
                            <p className="text-xs text-slate-500 leading-relaxed">
                                Parallel processing of wave boundaries and global classification labels via a shared deep 1D backbone.
                            </p>
                        </div>

                        <div className="h-px bg-slate-100"></div>

                        <div className="space-y-3">
                            <p className="text-[10px] font-black text-slate-400 tracking-widest uppercase flex items-center gap-2">
                                <CheckCircle className="w-3 h-3 text-green-500" /> Clinical Data
                            </p>
                            <p className="text-sm font-bold text-slate-700">LUDB & PTB-XL Optimized</p>
                            <p className="text-xs text-slate-500 leading-relaxed">
                                Trained on cardiology-annotated datasets for 99.2% recall on QRS landmarks and precise P-wave localization.
                            </p>
                        </div>

                        <div className="h-px bg-slate-100"></div>

                        <div className="grid grid-cols-2 gap-4 pt-2">
                            <div>
                                <p className="text-[10px] font-black text-slate-400 tracking-widest uppercase mb-1">Confidence</p>
                                <p className="text-3xl font-black text-slate-900 tracking-tighter">94.8%</p>
                            </div>
                            <div className="text-right">
                                <p className="text-[10px] font-black text-slate-400 tracking-widest uppercase mb-1">Dice Score</p>
                                <p className="text-3xl font-black text-blue-600 tracking-tighter">0.82</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );

    const renderInstructions = () => (
        <div className="flex-1 container mx-auto p-12 max-w-4xl animate-in fade-in slide-in-from-bottom-8 duration-700">
            <div className="text-center mb-16">
                <h2 className="text-3xl font-black text-slate-800 mb-4">Clinical Workflow Guide</h2>
                <p className="text-slate-500 font-medium">Follow this protocol to analyze ECG signals and generate AI-supported clinical findings.</p>
            </div>

            <div className="grid gap-8">
                {[
                    {
                        step: "01",
                        title: "Prepare your Dataset",
                        desc: "Ensure your ECG signal is in the `.npy` format. The system is calibrated for single-channel leads sampled at 500 Hz.",
                        icon: <FileText className="w-6 h-6 text-blue-600" />
                    },
                    {
                        step: "02",
                        title: "Upload Clinical Signal",
                        desc: "Navigate to the 'Upload ECG' tab and drag-and-drop your file into the intake node. The system will automatically initiate the neural processing protocol.",
                        icon: <Upload className="w-6 h-6 text-blue-600" />
                    },
                    {
                        step: "03",
                        title: "Interactive Visualization",
                        desc: "Once processed, use the interactive chart to zoom into specific wave segments. P-waves, QRS complexes, and T-waves are automatically segmented.",
                        icon: <Activity className="w-6 h-6 text-blue-600" />
                    },
                    {
                        step: "04",
                        title: "Review AI Explanation",
                        desc: "Read the clinical rationale generated by the XAI module. It explains the relationship between dissociated P-waves and potential AV blocks.",
                        icon: <Cpu className="w-6 h-6 text-blue-600" />
                    },
                    {
                        step: "05",
                        title: "Export Results",
                        desc: "Click 'Export Clinical PDF' to generate a professional research report containing all metrics, visualizations, and diagnostic conclusions.",
                        icon: <CheckCircle className="w-6 h-6 text-blue-600" />
                    }
                ].map((item, idx) => (
                    <div key={idx} className="flex gap-6 p-6 bg-white border border-slate-100 rounded-3xl shadow-sm hover:shadow-md transition-shadow group">
                        <div className="flex-shrink-0 w-12 h-12 bg-blue-50 rounded-2xl flex items-center justify-center font-black text-blue-600 group-hover:bg-blue-600 group-hover:text-white transition-colors">
                            {item.step}
                        </div>
                        <div className="space-y-1">
                            <h4 className="text-lg font-black text-slate-800 flex items-center gap-2">
                                {item.title}
                            </h4>
                            <p className="text-slate-500 text-sm leading-relaxed">{item.desc}</p>
                        </div>
                    </div>
                ))}
            </div>

            <div className="mt-16 text-center">
                <button
                    onClick={() => setView('analysis')}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-10 py-4 rounded-2xl font-black text-lg transition-all shadow-xl shadow-blue-200/50 active:scale-95"
                >
                    Proceed to Upload
                </button>
            </div>
        </div>
    );

    const renderAnalysis = () => (
        <div className="flex-1 container mx-auto p-6 max-w-7xl animate-in fade-in duration-500">
            {/* Top Metric Bar */}
            {result && (
                <div className="mb-8 animate-in slide-in-from-top-4 duration-500">
                    <AnalysisResults result={result} />
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

                {/* Left Column: Input & Controls */}
                <div className="lg:col-span-4 space-y-6">
                    <section className="glass p-6 rounded-2xl shadow-medical-100/50">
                        <div className="flex justify-between items-center mb-4">
                            <h2 className="text-lg font-bold text-slate-700 flex items-center gap-2">
                                <Upload className="text-blue-600 w-5 h-5" /> Patient Intake
                            </h2>
                            <span className="text-[10px] font-bold bg-slate-100 px-2 py-0.5 rounded text-slate-500 uppercase tracking-widest">Input Node</span>
                        </div>

                        <div className="relative group">
                            <input
                                type="file"
                                accept=".npy"
                                onChange={handleFileUpload}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                            />
                            <div className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all ${file ? 'border-blue-400 bg-blue-50/30' : 'border-slate-200 bg-white group-hover:border-blue-300'
                                }`}>
                                <div className={`w-14 h-14 rounded-full flex items-center justify-center mx-auto mb-4 transition-transform group-hover:scale-110 ${file ? 'bg-blue-600 text-white' : 'bg-slate-50 text-slate-400'
                                    }`}>
                                    <Upload className="w-7 h-7" />
                                </div>
                                <p className="text-sm text-slate-600 font-bold">
                                    {file ? 'Change Dataset File' : 'Drop Clinical Signal'}
                                </p>
                                <p className="text-[10px] text-slate-400 mt-2 font-mono uppercase">FORMAT: .NPY (1D)</p>
                            </div>
                        </div>

                        {file && (
                            <div className="mt-4 animate-in slide-in-from-bottom-2 duration-300">
                                <div className="flex items-center gap-3 text-xs bg-white border border-slate-100 p-3 rounded-xl shadow-sm">
                                    <div className="bg-blue-100 p-2 rounded-lg text-blue-600 font-bold">
                                        {(file.size / 1024).toFixed(0)} KB
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <p className="font-bold text-slate-700 truncate">{file.name}</p>
                                        <p className="text-[9px] text-slate-400 font-mono">SAMPLED AT: 500 HZ</p>
                                    </div>
                                    {result && <CheckCircle className="w-5 h-5 text-success" />}
                                </div>
                            </div>
                        )}
                    </section>

                    {loading && (
                        <div className="glass p-8 rounded-2xl flex flex-col items-center justify-center text-center animate-pulse border-blue-200">
                            <div className="relative w-16 h-16 mb-4">
                                <div className="absolute inset-0 border-4 border-blue-100 rounded-full"></div>
                                <div className="absolute inset-0 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                                <Activity className="absolute inset-0 m-auto w-6 h-6 text-blue-600 animate-bounce" />
                            </div>
                            <p className="text-blue-800 font-bold text-lg">AtrionNet Engine Processing</p>
                            <p className="text-slate-400 text-xs mt-1">Analyzing temporal dissociation patterns...</p>
                        </div>
                    )}

                    {result && (
                        <section className="glass p-6 rounded-2xl border-l-4 border-blue-600 animate-in fade-in slide-in-from-bottom-4 duration-500">
                            <h2 className="text-sm font-bold text-slate-500 uppercase tracking-widest mb-6 border-b pb-2">Analysis Summary</h2>
                            <div className="space-y-6">
                                <div className="bg-slate-900/5 p-4 rounded-xl border border-slate-200/50">
                                    <p className="text-[10px] text-slate-400 uppercase font-black tracking-tighter mb-1">Diagnosis</p>
                                    <p className="text-2xl font-black text-slate-800 tracking-tight leading-tight">{result.diagnosis}</p>
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                    <div className="p-4 rounded-2xl bg-blue-50 border border-blue-100 shadow-sm shadow-blue-200/50">
                                        <p className="text-[9px] text-blue-400 uppercase font-bold tracking-widest mb-1">Confidence</p>
                                        <p className="text-2xl font-black text-blue-700">{(result.confidence * 100).toFixed(1)}%</p>
                                    </div>
                                    <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200">
                                        <p className="text-[9px] text-slate-400 uppercase font-bold tracking-widest mb-1">Severity</p>
                                        <span className={`text-[10px] font-black uppercase inline-block px-1.5 py-0.5 rounded ${(result.severity || '').toLowerCase() === 'critical' ? 'bg-red-600 text-white' :
                                            (result.severity || '').toLowerCase() === 'severe' ? 'bg-red-500 text-white' :
                                                (result.severity || '').toLowerCase().includes('moderate') ? 'bg-orange-500 text-white' :
                                                    (result.severity || '').toLowerCase() === 'mild' ? 'bg-yellow-500 text-white' :
                                                        'bg-green-500 text-white'
                                            }`}>
                                            {result.severity || 'Normal'}
                                        </span>
                                    </div>
                                </div>

                                <a
                                    href={getReportUrl(result.report_id)}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-black py-4 rounded-2xl flex items-center justify-center gap-3 transition-all shadow-xl shadow-blue-200/50 active:scale-95"
                                >
                                    <FileText className="w-5 h-5" />
                                    <span>EXPORT CLINICAL PDF</span>
                                </a>
                            </div>
                        </section>
                    )}
                </div>

                {/* Right Column: Visualization */}
                <div className="lg:col-span-8 flex flex-col gap-8">
                    {!result && !loading && (
                        <div className="flex-1 flex flex-col items-center justify-center border-2 border-dashed rounded-3xl p-12 text-slate-300 min-h-[500px] bg-white/30">
                            <ShieldAlert className="w-32 h-32 mb-6 opacity-5" />
                            <p className="text-2xl font-black text-slate-400 tracking-tight uppercase tracking-[0.2em]">Ready for Simulation</p>
                            <p className="text-sm mt-3 max-w-sm text-center text-slate-400 font-medium">Select a validated ECG dataset to initiate clinical reasoning.</p>
                        </div>
                    )}

                    {result && (
                        <>
                            <section className="glass rounded-3xl overflow-hidden shadow-2xl">
                                <div className="p-6 border-b flex justify-between items-center bg-white/50">
                                    <h3 className="text-sm font-bold text-slate-600 flex items-center gap-2 uppercase tracking-widest">
                                        Interactive Signal Trace
                                    </h3>
                                    <div className="flex items-center gap-3">
                                        <span className="text-[10px] font-bold text-slate-400 uppercase">Segmentation</span>
                                        <button
                                            onClick={() => setShowSegmentation(!showSegmentation)}
                                            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${showSegmentation ? 'bg-blue-600' : 'bg-slate-200'}`}
                                        >
                                            <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${showSegmentation ? 'translate-x-6' : 'translate-x-1'}`} />
                                        </button>
                                    </div>
                                </div>
                                <div className="h-[350px] w-full bg-white p-4">
                                    <ECGViewer
                                        signal={result.signal}
                                        fs={500}
                                        height={300}
                                        waves={result.waves}
                                        showSegmentation={showSegmentation}
                                    />
                                </div>
                                {showSegmentation && (
                                    <div className="px-6 py-4 bg-slate-50 border-t flex flex-wrap gap-6 items-center">
                                        <div className="flex items-center gap-2">
                                            <div className="w-4 h-4 bg-blue-500/20 rounded border border-blue-500/40"></div>
                                            <span className="text-[11px] font-bold text-slate-600">P (Associated)</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <div className="w-4 h-4 bg-red-500/30 rounded border border-red-500/50"></div>
                                            <span className="text-[11px] font-bold text-slate-600">P (Dissociated)</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <div className="w-4 h-4 bg-emerald-500/20 rounded border border-emerald-500/40"></div>
                                            <span className="text-[11px] font-bold text-slate-600">QRS Complex</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <div className="w-4 h-4 bg-orange-500/20 rounded border border-orange-500/40"></div>
                                            <span className="text-[11px] font-bold text-slate-600">T-wave</span>
                                        </div>
                                        <div className="ml-auto text-[10px] italic text-slate-400 font-medium">
                                            * Clinical segmentation based on temporal relationship
                                        </div>
                                    </div>
                                )}
                            </section>

                            <section className="glass rounded-3xl overflow-hidden shadow-2xl">
                                <div className="p-6 bg-slate-900 border-b border-slate-800">
                                    <h3 className="text-sm font-bold text-blue-400 flex items-center gap-2 uppercase tracking-widest">
                                        <Cpu className="w-5 h-5" /> Clinical Explainer
                                    </h3>
                                </div>
                                <div className="bg-slate-900 text-slate-100 p-8 font-mono text-xs leading-relaxed max-h-[400px] overflow-y-auto">
                                    {result.explanation.split('\n').map((line, i) => (
                                        <p key={i} className={`mb-3 ${line.startsWith('[CRITICAL]') ? 'text-red-400 font-bold' : ''}`}>
                                            {line}
                                        </p>
                                    ))}
                                </div>
                            </section>
                        </>
                    )}
                </div>
            </div>
        </div>
    );

    const renderContent = () => {
        switch (view) {
            case 'home': return renderHome();
            case 'analysis': return renderAnalysis();
            case 'instructions': return renderInstructions();
            default: return renderHome();
        }
    };

    return (
        <div className="min-h-screen bg-[#f8fafc] flex flex-col font-sans">
            {/* Navigation Header from Image */}
            <nav className="bg-white/80 backdrop-blur-md sticky top-0 z-50 border-b border-slate-100">
                <div className="container mx-auto px-6 py-4 flex justify-between items-center">
                    <div className="flex items-center gap-2 cursor-pointer" onClick={() => setView('home')}>
                        <div className="w-10 h-10 border-2 border-blue-600 rounded-full flex items-center justify-center">
                            <div className="w-6 h-6 border-b-2 border-blue-600 rounded-full animate-spin-slow"></div>
                        </div>
                        <h1 className="text-2xl font-black text-blue-700 tracking-tight">AVScan</h1>
                    </div>

                    <div className="flex items-center gap-4">
                        <button
                            onClick={() => setView('home')}
                            className={`px-6 py-2 rounded-lg font-bold text-sm transition-all shadow-sm ${view === 'home' ? 'bg-blue-600 text-white shadow-blue-200' : 'bg-white border text-blue-600 hover:border-blue-300'}`}
                        >
                            Home
                        </button>
                        <button
                            onClick={() => setView('instructions')}
                            className={`px-6 py-2 rounded-lg font-bold text-sm transition-all border shadow-sm ${view === 'instructions' ? 'bg-blue-600 text-white shadow-blue-200' : 'bg-white border text-slate-700 hover:bg-slate-50'}`}
                        >
                            Instructions
                        </button>
                        <button
                            onClick={() => setView('analysis')}
                            className={`px-6 py-2 rounded-lg font-bold text-sm transition-all border shadow-sm ${view === 'analysis' ? 'bg-blue-600 text-white shadow-blue-200' : 'bg-white text-slate-700 hover:border-blue-300'}`}
                        >
                            Upload ECG
                        </button>
                        <div className={`ml-4 w-3 h-3 rounded-full ${backendStatus ? 'bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.5)]' : 'bg-red-500'}`} title={backendStatus ? 'System Online' : 'System Offline'}></div>
                    </div>
                </div>
            </nav>

            <main className="flex-1 flex flex-col">
                {renderContent()}
            </main>

            <footer className="p-12 text-center text-slate-400 text-[9px] border-t bg-slate-50 uppercase font-black tracking-[0.2em] mt-auto">
                <div className="max-w-5xl mx-auto">
                    {/* Clinical & Technical Headers */}
                    <div className="flex flex-col md:flex-row items-center justify-center gap-4 md:gap-12 mb-10 opacity-70">
                        <div className="flex items-center gap-3">
                            <ShieldAlert className="w-5 h-5 text-blue-600" />
                            <div className="text-left">
                                <p className="text-[10px] font-black text-slate-600">Clinical Decision Support</p>
                                <p className="text-[8px] font-bold text-slate-400 normal-case tracking-normal">Class II Analytical Interface</p>
                            </div>
                        </div>
                        <div className="hidden md:block w-px h-8 bg-slate-300"></div>
                        <div className="flex items-center gap-3">
                            <Cpu className="w-5 h-5 text-indigo-600" />
                            <div className="text-left">
                                <p className="text-[10px] font-black text-slate-600">AtrionNet Neural Engine</p>
                                <p className="text-[8px] font-bold text-slate-400 normal-case tracking-normal">Multi-Task Attention Architecture</p>
                            </div>
                        </div>
                    </div>

                    {/* Mission & Purpose Pillars */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12 max-w-4xl mx-auto px-4 border-y border-slate-200 py-10 border-dashed text-slate-500">
                        <div className="space-y-2 text-center md:text-left">
                            <p className="text-blue-600 text-[10px] font-black tracking-widest">PRECISION</p>
                            <p className="normal-case text-[11px] leading-relaxed">
                                Specialized in micro-segmentation of dissociated P-waves at 500Hz resolution for high-grade block detection.
                            </p>
                        </div>
                        <div className="space-y-2 text-center md:text-left">
                            <p className="text-blue-600 text-[10px] font-black tracking-widest">TRANSPARENCY</p>
                            <p className="normal-case text-[11px] leading-relaxed">
                                Integrated XAI framework utilizing Grad-CAM and Spatial Attention to provide clinically verifiable diagnostic logic.
                            </p>
                        </div>
                        <div className="space-y-2 text-center md:text-left">
                            <p className="text-blue-600 text-[10px] font-black tracking-widest">INNOVATION</p>
                            <p className="normal-case text-[11px] leading-relaxed">
                                Implementation of anchor-free instance segmentation to identify independent atrial-ventricular activation patterns.
                            </p>
                        </div>
                    </div>

                    {/* Research Context Metadata */}
                    <div className="flex flex-wrap items-center justify-center gap-6 text-[8px] tracking-[0.2em] text-slate-400 mb-10 opacity-60">
                        <span className="bg-slate-200 px-2 py-0.5 rounded text-slate-600 font-black">FOR RESEARCH USE ONLY</span>
                        <span>PROTOCOL ID: ATR-9</span>
                        <span>STUDY YEAR: 2026</span>
                        <span>BUILD: 1.0.4-STABLE</span>
                    </div>

                    {/* Final Attribution Signature */}
                    <div className="space-y-4 pt-4 border-t border-slate-200">
                        <p className="text-slate-500 font-black tracking-[0.4em] text-[10px]">© 2026 AVSCAN • ADVANCED ATRIOVENTRICULAR ANALYSIS</p>
                        <div className="flex flex-col items-center gap-1">
                            <p className="normal-case font-bold text-slate-600 tracking-normal text-[12px]">
                                Designed and Developed by <span className="text-blue-700 font-black px-2 py-0.5 bg-blue-50 rounded">Dumindu Induwara Gamage</span>
                            </p>
                            <p className="normal-case text-[10px] text-slate-400 font-medium tracking-normal italic">
                                Final Year Research Project | Computer Science & Engineering
                            </p>
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    );
};

export default App;
