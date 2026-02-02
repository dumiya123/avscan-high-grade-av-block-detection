import React from 'react';
import Plot from 'react-plotly.js';

interface ECGViewerProps {
    signal: number[];
    fs: number;
    height?: number | string;
    waves?: {
        p_associated: [number, number][];
        p_dissociated: [number, number][];
        qrs: [number, number][];
        t: [number, number][];
    };
    showSegmentation?: boolean;
}

const ECGViewer: React.FC<ECGViewerProps> = ({ signal, fs, height = 300, waves, showSegmentation = true }) => {
    // Generate time axis
    // If signal length is 1000, we assume it's subsampled from a 10s recording (effective fs = 100)
    const effectiveFs = signal.length === 1000 ? 100 : fs;
    const time = signal.map((_, i) => i / effectiveFs);

    const shapes: any[] = showSegmentation && waves ? [] : [];

    if (showSegmentation && waves) {
        const categories = [
            { data: waves.p_associated, color: 'rgb(59, 130, 246)', label: 'P-wave (Assoc)' },
            { data: waves.p_dissociated, color: 'rgb(239, 68, 68)', label: 'P-wave (Dissoc)' },
            { data: waves.qrs, color: 'rgb(16, 185, 129)', label: 'QRS' },
            { data: waves.t, color: 'rgb(245, 158, 11)', label: 'T-wave' }
        ];

        categories.forEach(({ data, color }) => {
            if (!data) return;
            data.forEach(([start, end]) => {
                const x0 = start / effectiveFs;
                const x1 = end / effectiveFs;

                shapes.push({
                    type: 'rect',
                    xref: 'x',
                    yref: 'paper',
                    x0, x1,
                    y0: 0,
                    y1: 1,
                    fillcolor: color.replace('rgb', 'rgba').replace(')', ', 0.3)'),
                    line: {
                        color: color.replace('rgb', 'rgba').replace(')', ', 0.5)'),
                        width: 1
                    },
                    layer: 'below'
                });
            });
        });
    }

    return (
        <div className="w-full h-full">
            <Plot
                data={[
                    {
                        x: time,
                        y: signal,
                        type: 'scatter',
                        mode: 'lines',
                        line: {
                            color: '#000000',
                            width: 1.2,
                        },
                        name: 'ECG Signal',
                        hoverinfo: 'x+y',
                    },
                ]}
                layout={{
                    autosize: true,
                    height: typeof height === 'number' ? height : undefined,
                    margin: { l: 40, r: 20, t: 10, b: 40 },
                    paper_bgcolor: 'white',
                    plot_bgcolor: 'white',
                    xaxis: {
                        title: 'Time (seconds)',
                        gridcolor: '#f1f5f9',
                        showgrid: true,
                        zeroline: false,
                        range: [0, Math.min(2, time[time.length - 1])],
                    },
                    yaxis: {
                        title: 'Amplitude (mV)',
                        gridcolor: '#f1f5f9',
                        showgrid: true,
                        zeroline: false,
                        range: [-2, 2],
                    },
                    shapes: shapes,
                    dragmode: 'pan',
                    hovermode: 'closest',
                    showlegend: false
                }}
                config={{
                    responsive: true,
                    displayModeBar: false,
                    scrollZoom: true,
                }}
                className="w-full h-full"
            />
        </div>
    );
};

export default ECGViewer;
