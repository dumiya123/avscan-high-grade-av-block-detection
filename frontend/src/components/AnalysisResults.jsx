import React from 'react';
import { Clock, Activity, Hash, Layers } from 'lucide-react';

const AnalysisResults = ({ result }) => {
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="glass p-4 rounded-xl flex items-center gap-4">
                <div className="bg-medical-100 p-3 rounded-lg">
                    <Clock className="text-medical-600 w-6 h-6" />
                </div>
                <div>
                    <p className="text-xs text-slate-500 font-bold uppercase">Avg PR Interval</p>
                    <p className="text-lg font-bold text-slate-800">{result.intervals.avg_pr?.toFixed(1) || '--'} <span className="text-xs font-normal">ms</span></p>
                </div>
            </div>

            <div className="glass p-4 rounded-xl flex items-center gap-4">
                <div className="bg-medical-100 p-3 rounded-lg">
                    <Activity className="text-medical-600 w-6 h-6" />
                </div>
                <div>
                    <p className="text-xs text-slate-500 font-bold uppercase">Heart Rate</p>
                    <p className="text-lg font-bold text-slate-800">{result.intervals.hr?.toFixed(0) || '--'} <span className="text-xs font-normal">bpm</span></p>
                </div>
            </div>

            <div className="glass p-4 rounded-xl flex items-center gap-4">
                <div className="bg-medical-100 p-3 rounded-lg">
                    <Hash className="text-medical-600 w-6 h-6" />
                </div>
                <div>
                    <p className="text-xs text-slate-500 font-bold uppercase">P : QRS Ratio</p>
                    <p className="text-lg font-bold text-slate-800">{result.intervals.p_qrs_ratio.toFixed(2)}</p>
                </div>
            </div>

            <div className="glass p-4 rounded-xl flex items-center gap-4">
                <div className="bg-medical-100 p-3 rounded-lg">
                    <Hash className="text-medical-600 w-6 h-6" />
                </div>
                <div>
                    <p className="text-xs text-slate-500 font-bold uppercase">Detections</p>
                    <p className="text-lg font-bold text-slate-800">{result.intervals.pr.length} <span className="text-xs font-normal">beats</span></p>
                </div>
            </div>
        </div>
    );
};

export default AnalysisResults;
