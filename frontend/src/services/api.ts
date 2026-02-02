import axios from 'axios';
import type { ECGAnalysisResult } from '../types';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
});

export const analyzeECG = async (file: File): Promise<ECGAnalysisResult> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post<ECGAnalysisResult>('/analyze', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    return response.data;
};

export const getReportUrl = (report_id: string): string => {
    return `${API_BASE_URL}/report/${report_id}`;
};

export const checkHealth = async (): Promise<boolean> => {
    try {
        const response = await api.get('/health');
        return response.data.status === 'online';
    } catch (error) {
        return false;
    }
};
