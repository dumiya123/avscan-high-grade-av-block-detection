import React from 'react';
import Plot from 'react-plotly.js';

interface ECGViewerProps {
    signal: number[];
    fs: number;
    height?: number | string;
}

const ECGViewer: React.FC<ECGViewerProps> = ({ signal, fs, height = 300 }) => {
    // Create time axis
    const time = signal.map((_, i) => i / fs);

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
                            color: '#0284c7',
                            width: 1.5,
                            shape: 'spline',
                        },
                        name: 'ECG Signal',
                        hoverinfo: 'x+y',
                    },
                ]}
                layout={{
                    autosize: true,
                    height: typeof height === 'number' ? height : undefined,
                    margin: { l: 40, r: 20, t: 10, b: 40 },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    xaxis: {
                        title: 'Time (seconds)',
                        gridcolor: '#e2e8f0',
                        zeroline: false,
                        range: [0, 2], // Show first 2 seconds initially
                    },
                    yaxis: {
                        title: 'Amplitude (mV)',
                        gridcolor: '#e2e8f0',
                        zeroline: false,
                    },
                    dragmode: 'pan',
                    hovermode: 'closest',
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
