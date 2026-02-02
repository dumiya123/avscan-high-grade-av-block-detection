import React from 'react';
import { Clock, Activity, Hash, Layers } from 'lucide-react';
import { cn } from "../../utils/cn";
import Card from '../ui/Card';

/**
 * Component to display high-level temporal metrics from ECG analysis.
 */
const AnalysisMetrics = ({ result }) => {
    const metrics = [
        {
            label: 'Avg PR Interval',
            value: `${result.intervals.avg_pr?.toFixed(1) || '--'} ms`,
            icon: Clock,
            color: 'text-blue-600',
            bg: 'bg-blue-50'
        },
        {
            label: 'Heart Rate',
            value: `${result.intervals.hr?.toFixed(0) || '--'} bpm`,
            icon: Activity,
            color: 'text-indigo-600',
            bg: 'bg-indigo-50'
        },
        {
            label: 'P : QRS Ratio',
            value: result.intervals.p_qrs_ratio.toFixed(2),
            icon: Hash,
            color: 'text-emerald-600',
            bg: 'bg-emerald-50'
        },
        {
            label: 'Detections',
            value: `${result.intervals.pr.length} beats`,
            icon: Layers,
            color: 'text-violet-600',
            bg: 'bg-violet-50'
        }
    ];

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {metrics.map((metric) => (
                <Card key={metric.label} padding="sm" className="flex items-center gap-4">
                    <div className={cn(metric.bg, "p-3 rounded-xl")}>
                        <metric.icon className={cn(metric.color, "w-6 h-6")} />
                    </div>
                    <div>
                        <p className="text-[10px] text-slate-400 font-black uppercase tracking-wider">
                            {metric.label}
                        </p>
                        <p className="text-lg font-bold text-slate-800">
                            {metric.value}
                        </p>
                    </div>
                </Card>
            ))}
        </div>
    );
};



export default AnalysisMetrics;
