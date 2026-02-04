import axios from 'axios';

// Use environment variable for API base URL, fallback to local for dev
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
});

export const previewECG = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/preview', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    return response.data;
};

export const analyzeECG = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/analyze', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    return response.data;
};

export const getReportUrl = (report_id) => {
    return `${API_BASE_URL}/report/${report_id}`;
};

export const checkHealth = async () => {
    try {
        const response = await api.get('/health');
        return response.data.status === 'online';
    } catch (error) {
        return false;
    }
};
