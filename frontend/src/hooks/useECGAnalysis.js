import { useState, useCallback } from "react";
import { analyzeECG } from "../services/api";

/**
 * Custom hook to handle ECG file analysis, including state management
 * for uploading, loading, results, and error handling.
 */
export function useECGAnalysis() {
    const [file, setFile] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const processFile = useCallback(async (uploadedFile) => {
        if (!uploadedFile) return;

        setFile(uploadedFile);
        setIsLoading(true);
        setResult(null);
        setError(null);

        try {
            const data = await analyzeECG(uploadedFile);
            setResult(data);
            return data;
        } catch (err) {
            const message = "Failed to analyze ECG. Please ensure the backend is running and the file is a valid .npy sample.";
            console.error("[ECGAnalysis Error]:", err);
            setError(message);
            throw new Error(message);
        } finally {
            setIsLoading(false);
        }
    }, []);

    const reset = useCallback(() => {
        setFile(null);
        setResult(null);
        setError(null);
        setIsLoading(false);
    }, []);

    return {
        file,
        isLoading,
        result,
        error,
        processFile,
        reset
    };
}
