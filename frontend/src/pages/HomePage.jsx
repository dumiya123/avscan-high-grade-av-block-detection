import React from 'react';
import { Activity, FileText, Cpu } from 'lucide-react';
import { cn } from "../utils/cn";
import Button from '../components/ui/Button';
import Card from '../components/ui/Card';
import Badge from '../components/ui/Badge';

/**
 * HomePage Component
 * Purpose: Professional landing page highlighting the research context and core engine.
 */
const HomePage = ({ onNavigate }) => {
    return (
        <div className="flex-1 flex flex-col lg:flex-row items-center justify-center gap-12 px-6 lg:px-24 py-12 bg-[#fdfdfd]">

            {/* Hero Section */}
            <section className="flex-1 max-w-2xl space-y-10 animate-in fade-in slide-in-from-left-12 duration-1000">
                <div className="space-y-4">
                    <Badge variant="primary" className="px-3 py-1">
                        Clinical Research Prototype // 2026
                    </Badge>

                    <h2 className="text-5xl lg:text-7xl font-black text-slate-900 tracking-tighter leading-[0.9]">
                        AVScan <span className="text-blue-600">.</span>
                    </h2>

                    <h3 className="text-2xl lg:text-3xl font-bold text-slate-400 leading-tight">
                        High-Grade AV Block Detection <br />
                        <span className="text-slate-800">Powered by AtrionNet Fusion.</span>
                    </h3>
                </div>

                <div className="relative pl-8">
                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-blue-600 to-indigo-600 rounded-full" />
                    <p className="text-xl text-slate-600 leading-relaxed font-medium italic">
                        "An explainable neural framework engineered for the automated identification of Atrial-Ventricular dissociation in 10-second ECG rhythms."
                    </p>
                </div>

                <div className="flex flex-wrap gap-4 pt-4">
                    <Button
                        size="lg"
                        icon={Activity}
                        onClick={() => onNavigate('analysis')}
                    >
                        Launch Analytics
                    </Button>

                    <Button
                        variant="secondary"
                        size="lg"
                        icon={FileText}
                        onClick={() => onNavigate('instructions')}
                    >
                        View Logic
                    </Button>
                </div>
            </section>

            {/* Technical Overview Card */}
            <aside className="w-full lg:w-[440px] animate-in fade-in slide-in-from-right-12 duration-1000 delay-300">
                <Card padding="lg" variant="default" className="relative group overflow-hidden border-2 border-slate-50 shadow-xl">
                    <div className="absolute -top-12 -right-12 w-48 h-48 bg-blue-600/5 blur-[80px] rounded-full" />

                    <header className="flex items-center gap-4 mb-10">
                        <div className="w-14 h-14 bg-gradient-to-br from-blue-600 to-indigo-700 rounded-2xl flex items-center justify-center shadow-lg shadow-blue-100 text-white">
                            <Cpu className="w-8 h-8" />
                        </div>
                        <div>
                            <p className="text-[11px] font-black text-blue-600 tracking-[0.25em] uppercase">Core Engine</p>
                            <p className="text-lg font-black text-slate-800 tracking-tight">AtrionNet v1.0</p>
                        </div>
                    </header>

                    <div className="space-y-8">
                        <FeatureMetric
                            title="Multi-Task Attention Fusion"
                            description="Parallel processing of wave boundaries and global classification labels via a shared deep 1D backbone."
                        />
                        <div className="h-px bg-slate-100" />
                        <FeatureMetric
                            title="LUDB & PTB-XL Optimized"
                            description="Trained on cardiology-annotated datasets for 99.2% recall on QRS landmarks and precise P-wave localization."
                        />
                        <div className="h-px bg-slate-100" />

                        <footer className="grid grid-cols-2 gap-4 pt-2">
                            <StatBlock label="Confidence" value="94.8%" highlight />
                            <StatBlock label="Dice Score" value="0.82" />
                        </footer>
                    </div>
                </Card>
            </aside>
        </div>
    );
};

/* Internal UI Helpers */

const FeatureMetric = ({ title, description }) => (
    <div className="space-y-3">
        <Badge variant="success" className="bg-transparent text-slate-400 p-0">
            Research Scope
        </Badge>
        <p className="text-sm font-bold text-slate-700">{title}</p>
        <p className="text-xs text-slate-500 leading-relaxed">{description}</p>
    </div>
);

const StatBlock = ({ label, value, highlight }) => (
    <div className={highlight ? "" : "text-right"}>
        <p className="text-[10px] font-black text-slate-400 tracking-widest uppercase mb-1">{label}</p>
        <p className={cn("text-3xl font-black tracking-tighter", highlight ? "text-slate-900" : "text-blue-600")}>
            {value}
        </p>
    </div>
);

export default HomePage;
