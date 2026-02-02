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
}

export interface ECGData {
    signal: number[];
    fs: number;
}
