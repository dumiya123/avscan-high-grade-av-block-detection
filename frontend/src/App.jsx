import React, { useState, Suspense, lazy } from 'react';
import Navbar from './components/layout/Navbar';
import './App.css';

// Lazy load pages for better performance
const HomePage = lazy(() => import('./pages/HomePage'));
const InstructionsPage = lazy(() => import('./pages/InstructionsPage'));
const AnalysisPage = lazy(() => import('./pages/AnalysisPage'));
const AboutPage = lazy(() => import('./pages/AboutPage'));

// Fallback loader for Suspense
const PageLoader = () => (
    <div className="flex items-center justify-center min-h-[calc(100vh-64px)] w-full">
        <div className="w-8 h-8 border-4 border-slate-200 border-t-blue-600 rounded-full animate-spin"></div>
    </div>
);

const App = () => {
    const [view, setView] = useState('home');

    const renderContent = () => {
        switch (view) {
            case 'home': return <HomePage onNavigate={setView} />;
            case 'instructions': return <InstructionsPage onNavigate={setView} />;
            case 'analysis': return <AnalysisPage />;
            case 'about': return <AboutPage />;
            default: return <HomePage onNavigate={setView} />;
        }
    };

    return (
        <>
            <Navbar currentView={view} onNavigate={setView} />
            <Suspense fallback={<PageLoader />}>
                {renderContent()}
            </Suspense>
        </>
    );
};

export default App;
