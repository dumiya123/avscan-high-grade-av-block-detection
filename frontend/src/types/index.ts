export interface ECGAnalysisResult {
    diagnosis: string;
    confidence: number;
    severity: 'low' | 'moderate' | 'high' | 'moderate-severe' | 'critical';
    intervals: {
        pr: number[];
        rr: number[];
        p_qrs_ratio: number;
        avg_pr: number;
        hr: number;
    };
    report_id: string;
    explanation: string;
    signal: number[];
    waves: {
        p_associated: [number, number][];
        p_dissociated: [number, number][];
        qrs: [number, number][];
        t: [number, number][];
    };
}

export interface ECGData {
    signal: number[];
    fs: number;
}
