import React, { useState, useEffect } from 'react';
import { Activity, Upload, FileText, CheckCircle, ShieldAlert, Cpu } from 'lucide-react';
import { analyzeECG, checkHealth, getReportUrl } from './services/api';
import type { ECGAnalysisResult } from './types';
// Import our components later or build them inline for now to ensure one-shot success
import AnalysisResults from './components/AnalysisResults';
import ECGViewer from './components/ECGViewer';

const App: React.FC = () => {
  const [view, setView] = useState<'home' | 'analysis' | 'instructions'>('home');
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ECGAnalysisResult | null>(null);
  const [backendStatus, setBackendStatus] = useState<boolean>(false);

  useEffect(() => {
    checkHealth().then(setBackendStatus);
    const interval = setInterval(() => {
      checkHealth().then(setBackendStatus);
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
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
    <div className="flex-1 flex flex-col lg:flex-row items-center justify-center gap-16 px-6 lg:px-24">
      <div className="flex-1 max-w-2xl space-y-8 animate-in fade-in slide-in-from-left-8 duration-700">
        <h2 className="text-4xl lg:text-5xl font-black text-slate-800 leading-tight">
          Reveal insights with deep learning.
        </h2>
        <h3 className="text-3xl lg:text-4xl font-black text-slate-900 leading-[1.2]">
          AtrionNet: A Novel Explainable AI-Focused Approach for Dissociated P-Wave Detection in ECG Signals to Support Diagnosis of High-Grade Atrioventricular Block.
        </h3>
        <div className="space-y-4 pt-8">
          <p className="text-slate-500 font-bold">Ready to start?</p>
          <button
            onClick={() => setView('analysis')}
            className="group flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg font-bold transition-all shadow-lg active:scale-95"
          >
            <FileText className="w-5 h-5" />
            <span>Get Started</span>
          </button>
        </div>
      </div>

      <div className="w-full lg:w-96 animate-in fade-in slide-in-from-right-8 duration-700 delay-200">
        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-8 rounded-3xl border border-blue-100/50 shadow-2xl relative overflow-hidden group">
          <div className="absolute top-0 right-0 w-32 h-32 bg-yellow-200/20 blur-3xl rounded-full -mr-16 -mt-16 group-hover:bg-yellow-200/40 transition-colors"></div>
          <p className="text-blue-600 font-bold text-center mb-4">Anchor-Free 1D Instance Segmentation</p>
          <p className="text-slate-600 text-sm leading-relaxed text-center">
            Detects multiple P-waves per RR interval — overcoming semantic constraints in traditional models.
          </p>
        </div>
      </div>
    </div>
  );

  const renderInstructions = () => (
    <div className="flex-1 container mx-auto p-12 max-w-4xl animate-in fade-in slide-in-from-bottom-8 duration-700">
      <div className="text-center mb-16">
        <h2 className="text-3xl font-black text-slate-800 mb-4">How to use AVSegNet</h2>
        <p className="text-slate-500 font-medium">Follow these simple steps to analyze ECG signals with our Explainable AI system.</p>
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
                </div>
                <div className="h-[350px] w-full bg-white p-4">
                  <ECGViewer
                    signal={result.signal}
                    fs={500}
                    height={300}
                  />
                </div>
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
            <h1 className="text-2xl font-black text-blue-700 tracking-tight">AVSegNet</h1>
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

      <footer className="p-8 text-center text-slate-400 text-[10px] border-t bg-white uppercase font-bold tracking-widest mt-auto">
        <div className="flex items-center justify-center gap-4 mb-4 opacity-50">
          <ShieldAlert className="w-4 h-4" />
          <span>Clinical Research Node // Protocol ATR-9 // 2026</span>
        </div>
        <p>© 2026 AtrionNet AI Systems</p>
      </footer>
    </div>
  );
};

export default App;
