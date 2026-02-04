import { useState, useCallback } from "react";
import { analyzeECG, previewECG } from "../services/api";

/**
 * Custom hook to handle ECG file analysis with dual-window support.
 * Manages preview state and analysis state separately for two-window workflow.
 */
export function useECGAnalysis() {
    const [file, setFile] = useState(null);
    const [isPreviewing, setIsPreviewing] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [previewData, setPreviewData] = useState(null);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    // Step 1: Load file and generate preview
    const processFile = useCallback(async (uploadedFile) => {
        if (!uploadedFile) return;

        setFile(uploadedFile);
        setIsPreviewing(true);
        setPreviewData(null);
        setResult(null);
        setError(null);

        try {
            const data = await previewECG(uploadedFile);
            setPreviewData(data);
            return data;
        } catch (err) {
            const message = "Failed to load preview. Please check the file format.";
            console.error("[Preview Error]:", err);
            setError(message);
        } finally {
            setIsPreviewing(false);
        }
    }, []);

    // Step 2: Run AI analysis
    const runAnalysis = useCallback(async () => {
        if (!file) return;

        setIsAnalyzing(true);
        setError(null);

        try {
            const data = await analyzeECG(file);
            setResult(data);
            return data;
        } catch (err) {
            const message = "AI analysis failed. Please ensure the backend is running.";
            console.error("[Analysis Error]:", err);
            setError(message);
        } finally {
            setIsAnalyzing(false);
        }
    }, [file]);

    const reset = useCallback(() => {
        setFile(null);
        setPreviewData(null);
        setResult(null);
        setError(null);
        setIsPreviewing(false);
        setIsAnalyzing(false);
    }, []);

    return {
        file,
        isLoading: isPreviewing || isAnalyzing,
        isPreviewing,
        isAnalyzing,
        previewData,
        result,
        error,
        processFile,
        runAnalysis,
        reset
    };
}
