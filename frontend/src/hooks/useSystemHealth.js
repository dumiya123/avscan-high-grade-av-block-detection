import { useState, useEffect } from "react";
import { checkHealth } from "../services/api";

/**
 * Hook to monitor the backend system health.
 * Provides real-time connectivity status to the UI.
 */
export function useSystemHealth(intervalMs = 10000) {
    const [isOnline, setIsOnline] = useState(false);

    useEffect(() => {
        // Initial check
        checkHealth().then(setIsOnline);

        // Periodic check
        const interval = setInterval(() => {
            checkHealth().then(setIsOnline);
        }, intervalMs);

        return () => clearInterval(interval);
    }, [intervalMs]);

    return isOnline;
}
