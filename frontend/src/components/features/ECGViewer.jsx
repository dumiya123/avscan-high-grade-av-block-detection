import React from 'react';
import Plot from 'react-plotly.js';
import { cn } from "../../utils/cn";

/**
 * Clinical-grade ECG signal visualization component.
 * Features standard ECG paper grid and wave segmentation overlays.
 */
const ECGViewer = ({
    signal,
    fs,
    height = 300,
    waves,
    showSegmentation = true,
    className
}) => {
    // Determine effective frequency (handles subsampled UI signals)
    const effectiveFs = signal.length === 1000 ? 100 : fs;
    const time = signal.map((_, i) => i / effectiveFs);

    const shapes = [];

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
        <div className={cn("w-full h-full bg-white", className)}>
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
                    margin: { l: 50, r: 20, t: 30, b: 50 },
                    paper_bgcolor: 'white',
                    plot_bgcolor: '#fff5f5', // Authentic ECG paper pink
                    xaxis: {
                        title: 'Time (seconds)',
                        range: [0, Math.min(2, time[time.length - 1])],
                        dtick: 0.2, // Large squares (0.2s)
                        gridcolor: '#feb2b2',
                        gridwidth: 1.5,
                        showgrid: true,
                        zeroline: false,
                        minor: {
                            dtick: 0.04, // Small squares (0.04s)
                            showgrid: true,
                            gridcolor: '#fed7d7',
                            gridwidth: 0.5,
                        }
                    },
                    yaxis: {
                        title: 'Amplitude (mV)',
                        range: [-2, 2],
                        dtick: 0.5, // Large squares (0.5mV)
                        gridcolor: '#feb2b2',
                        gridwidth: 1.5,
                        showgrid: true,
                        zeroline: true,
                        zerolinecolor: '#feb2b2',
                        zerolinewidth: 2,
                        minor: {
                            dtick: 0.1, // Small squares (0.1mV)
                            showgrid: true,
                            gridcolor: '#fed7d7',
                            gridwidth: 0.5,
                        }
                    },
                    shapes: shapes,
                    dragmode: 'pan',
                    hovermode: 'closest',
                    showlegend: false
                }}
                config={{
                    responsive: true,
                    displayModeBar: true,
                    scrollZoom: true,
                }}
                className="w-full h-full"
            />
        </div>
    );
};

export default ECGViewer;
