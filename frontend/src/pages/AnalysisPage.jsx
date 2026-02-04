import React, { useState } from 'react';
import { Upload, FileText, Activity, ShieldAlert, Cpu, CheckCircle } from 'lucide-react';
import { useECGAnalysis } from "../hooks/useECGAnalysis";
import { getReportUrl } from "../services/api";
import { cn } from "../utils/cn";

import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";
import Button from "../components/ui/Button";
import AnalysisMetrics from "../components/features/AnalysisMetrics";
import ECGViewer from "../components/features/ECGViewer";

/**
 * AnalysisPage Component
 * Two-window workflow: Preview window + Analysis window
 */
const AnalysisPage = () => {
    const {
        file,
        isPreviewing,
        isAnalyzing,
        previewData,
        result,
        error,
        processFile,
        runAnalysis
    } = useECGAnalysis();

    const [showSegmentation, setShowSegmentation] = useState(true);

    const handleFileChange = (e) => {
        const uploadedFile = e.target.files?.[0];
        if (uploadedFile) processFile(uploadedFile);
    };

    return (
        <div className="flex-1 container mx-auto p-6 max-w-7xl animate-in fade-in duration-500">
            {/* Metrics Overview Bar */}
            {result && (
                <div className="mb-8 animate-in slide-in-from-top-4 duration-500">
                    <AnalysisMetrics result={result} />
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Left Control Column */}
                <div className="lg:col-span-4 space-y-6">
                    <InputSection
                        file={file}
                        onFileChange={handleFileChange}
                        isLoading={isPreviewing}
                    />

                    {/* Analyse ECG Button */}
                    {previewData && !result && (
                        <div className="animate-in slide-in-from-bottom-4 duration-500">
                            <Button
                                size="lg"
                                variant="primary"
                                className="w-full py-6 text-sm font-black uppercase tracking-widest shadow-xl"
                                icon={Cpu}
                                onClick={runAnalysis}
                                disabled={isAnalyzing}
                            >
                                {isAnalyzing ? 'Processing AI...' : 'Analyse ECG'}
                            </Button>
                        </div>
                    )}

                    {result && (
                        <DiagnosisCard
                            result={result}
                            reportUrl={getReportUrl(result.report_id)}
                        />
                    )}

                    {error && <ErrorCard message={error} />}
                </div>

                {/* Right Visualization Column - Two Windows */}
                <div className="lg:col-span-8 space-y-8">
                    {/* Window 1: Preview */}
                    <div className="space-y-3">
                        <div className="flex items-center gap-2 px-2">
                            <div className="w-2 h-2 rounded-full bg-slate-400" />
                            <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest">
                                Window 01 / Signal Preview
                            </h3>
                        </div>
                        <VisualizationWindow
                            title="Raw ECG Signal"
                            data={previewData}
                            isLoading={isPreviewing}
                            showSegmentation={false}
                        />
                    </div>

                    {/* Window 2: Analysis Result */}
                    <div className="space-y-3">
                        <div className="flex items-center gap-2 px-2">
                            <div className={cn(
                                "w-2 h-2 rounded-full",
                                result ? "bg-blue-600 animate-pulse" : "bg-slate-300"
                            )} />
                            <h3 className={cn(
                                "text-[10px] font-black uppercase tracking-widest",
                                result ? "text-blue-600" : "text-slate-300"
                            )}>
                                Window 02 / AI Analysis Result
                            </h3>
                        </div>
                        <VisualizationWindow
                            title="Segmented ECG with Wave Detection"
                            data={result}
                            isLoading={isAnalyzing}
                            showSegmentation={showSegmentation}
                            onToggleSegmentation={() => setShowSegmentation(!showSegmentation)}
                            isAnalysisWindow={true}
                        />
                    </div>

                    {result && <ExplanationSection explanation={result.explanation} />}
                </div>
            </div>
        </div>
    );
};

/* --- UI Components --- */

const InputSection = ({ file, onFileChange, isLoading }) => (
    <Card variant="default" padding="md">
        <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-bold text-slate-700 flex items-center gap-2">
                <Upload className="text-blue-600 w-5 h-5" /> Upload ECG
            </h2>
            <Badge>Input</Badge>
        </div>

        <div className="relative group">
            <input
                type="file"
                accept=".npy"
                onChange={onFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                disabled={isLoading}
            />
            <div className={cn(
                "border-2 border-dashed rounded-2xl p-8 text-center transition-all",
                file ? 'border-blue-400 bg-blue-50/30' : 'border-slate-200 bg-white group-hover:border-blue-300'
            )}>
                <div className={cn(
                    "w-14 h-14 rounded-full flex items-center justify-center mx-auto mb-4 transition-transform group-hover:scale-110",
                    file ? 'bg-blue-600 text-white' : 'bg-slate-50 text-slate-400'
                )}>
                    {isLoading ? (
                        <div className="w-6 h-6 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    ) : (
                        <Upload className="w-7 h-7" />
                    )}
                </div>
                <p className="text-sm text-slate-600 font-bold">
                    {file ? 'File Loaded' : 'Drop ECG File'}
                </p>
                <p className="text-[10px] text-slate-400 mt-2 font-mono uppercase tracking-widest">
                    Format: .NPY (500Hz)
                </p>
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
                    <CheckCircle className="w-5 h-5 text-green-500" />
                </div>
            </div>
        )}
    </Card>
);

const DiagnosisCard = ({ result, reportUrl }) => (
    <Card padding="md" className="border-l-4 border-blue-600 space-y-6 animate-in slide-in-from-bottom-6 duration-700">
        <Badge>Analysis Summary</Badge>

        <div className="bg-slate-900/5 p-4 rounded-xl border border-slate-200/50">
            <p className="text-[10px] text-slate-400 uppercase font-black tracking-tighter mb-1">Diagnosis</p>
            <p className="text-2xl font-black text-slate-800 tracking-tight leading-tight">{result.diagnosis}</p>
        </div>

        <div className="grid grid-cols-2 gap-4">
            <Card padding="sm" className="bg-blue-50 border-blue-100">
                <p className="text-[9px] text-blue-400 uppercase font-black mb-1">Confidence</p>
                <p className="text-2xl font-black text-blue-700">{(result.confidence * 100).toFixed(1)}%</p>
            </Card>

            <Card padding="sm" className="bg-slate-50 border-slate-200">
                <p className="text-[9px] text-slate-400 uppercase font-black mb-1">Severity</p>
                <Badge variant={result.severity?.toLowerCase() === 'critical' ? 'danger' : 'warning'}>
                    {result.severity || 'Normal'}
                </Badge>
            </Card>
        </div>

        <Button size="md" className="w-full py-4 text-xs tracking-widest" icon={FileText} onClick={() => window.open(reportUrl, '_blank')}>
            EXPORT CLINICAL PDF
        </Button>
    </Card>
);

const ErrorCard = ({ message }) => (
    <Card padding="md" className="bg-red-50 border-red-200">
        <p className="text-sm font-bold text-red-600 flex items-center gap-2">
            <ShieldAlert className="w-4 h-4" /> System Error
        </p>
        <p className="text-xs text-red-500 mt-2">{message}</p>
    </Card>
);

const VisualizationWindow = ({
    title,
    data,
    isLoading,
    showSegmentation = false,
    onToggleSegmentation,
    isAnalysisWindow = false
}) => (
    <section className={cn(
        "bg-white rounded-3xl overflow-hidden shadow-2xl border transition-all duration-500",
        data ? 'border-slate-100 opacity-100' : 'border-slate-100 opacity-40'
    )}>
        <header className="p-6 border-b flex justify-between items-center bg-slate-50/50">
            <h3 className="text-sm font-black text-slate-600 flex items-center gap-2 uppercase tracking-widest">
                {title}
            </h3>
            {isAnalysisWindow && data && (
                <div className="flex items-center gap-3">
                    <span className="text-[10px] font-bold text-slate-400 uppercase">Segmentation</span>
                    <button
                        onClick={onToggleSegmentation}
                        className={cn(
                            "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
                            showSegmentation ? 'bg-blue-600' : 'bg-slate-200'
                        )}
                    >
                        <span className={cn(
                            "inline-block h-4 w-4 transform rounded-full bg-white transition-transform",
                            showSegmentation ? 'translate-x-6' : 'translate-x-1'
                        )} />
                    </button>
                </div>
            )}
        </header>

        <div className="h-[350px] w-full p-4 relative">
            {isLoading && (
                <div className="absolute inset-0 bg-white/80 z-20 flex flex-col items-center justify-center backdrop-blur-sm">
                    <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4" />
                    <p className="text-blue-600 font-black text-xs uppercase tracking-widest">
                        {isAnalysisWindow ? 'Running AI Analysis...' : 'Loading Preview...'}
                    </p>
                </div>
            )}

            {data ? (
                <ECGViewer
                    signal={data.signal}
                    fs={500}
                    waves={isAnalysisWindow ? data.waves : null}
                    showSegmentation={isAnalysisWindow ? showSegmentation : false}
                />
            ) : (
                <div className="h-full flex flex-col items-center justify-center text-slate-300">
                    <Activity className="w-16 h-16 opacity-10 mb-4" />
                    <p className="text-sm font-bold uppercase tracking-widest">
                        {isAnalysisWindow ? 'Awaiting Analysis' : 'No Data'}
                    </p>
                </div>
            )}
        </div>

        {isAnalysisWindow && data && showSegmentation && <SegmentationLegend />}
    </section>
);

const SegmentationLegend = () => (
    <footer className="px-6 py-4 bg-white border-t flex flex-wrap gap-6 items-center">
        {[
            { color: 'bg-blue-500', label: 'P (Associated)' },
            { color: 'bg-red-500', label: 'P (Dissociated)' },
            { color: 'bg-emerald-500', label: 'QRS Complex' },
            { color: 'bg-orange-500', label: 'T-wave' },
        ].map(item => (
            <div key={item.label} className="flex items-center gap-2">
                <div className={cn(item.color, "w-3 h-3 rounded-full opacity-60 border border-current")} />
                <span className="text-[10px] font-black text-slate-500 uppercase">{item.label}</span>
            </div>
        ))}
        <p className="ml-auto text-[10px] italic text-slate-400">
            * Dynamic temporal relationship analysis
        </p>
    </footer>
);

const ExplanationSection = ({ explanation }) => (
    <Card variant="default" padding="none" className="bg-slate-900 shadow-2xl animate-in fade-in duration-1000">
        <header className="p-6 border-b border-slate-800 bg-slate-900/50">
            <h3 className="text-xs font-black text-blue-400 flex items-center gap-2 uppercase tracking-[0.2em]">
                <Cpu className="w-4 h-4" /> Clinical Rationale
            </h3>
        </header>
        <div className="p-8 font-mono text-[11px] leading-relaxed max-h-[400px] overflow-y-auto text-slate-300">
            {explanation.split('\n').map((line, i) => (
                <p key={i} className={cn("mb-3", line.startsWith('[CRITICAL]') && "text-red-400 font-bold")}>
                    {line.startsWith('>') ? <span className="text-blue-500">→</span> : null} {line}
                </p>
            ))}
        </div>
    </Card>
);

export default AnalysisPage;
