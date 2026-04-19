import { useState, useEffect } from 'react';

/**
 * useResponsive — hook to detect viewport size.
 * Breakpoints match index.css
 */
export const useResponsive = () => {
    const [state, setState] = useState({
        isMobile: window.innerWidth <= 600,
        isTablet: window.innerWidth > 600 && window.innerWidth <= 960,
        isDesktop: window.innerWidth > 960,
        width: window.innerWidth
    });

    useEffect(() => {
        const handleResize = () => {
            setState({
                isMobile: window.innerWidth <= 600,
                isTablet: window.innerWidth > 600 && window.innerWidth <= 960,
                isDesktop: window.innerWidth > 960,
                width: window.innerWidth
            });
        };

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    return state;
};
