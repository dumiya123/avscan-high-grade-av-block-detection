import React, { useState } from 'react';

// Layout Components
import Navbar from './components/layout/Navbar';
import Footer from './components/layout/Footer';

// Page Views
import HomePage from './pages/HomePage';
import InstructionsPage from './pages/InstructionsPage';
import AnalysisPage from './pages/AnalysisPage';

/**
 * Main Application Component (Entry Point)
 * Responsible for high-level view orchestration and layout wrapping.
 * 
 * Follows industry-standard structural principles:
 * - Decoupled child components
 * - Layout/Page separation
 * - Clean state orchestration
 */
const App = () => {
    const [view, setView] = useState('home');

    const renderContent = () => {
        switch (view) {
            case 'home':
                return <HomePage onNavigate={setView} />;
            case 'instructions':
                return <InstructionsPage onNavigate={setView} />;
            case 'analysis':
                return <AnalysisPage />;
            default:
                return <HomePage onNavigate={setView} />;
        }
    };

    return (
        <div className="min-h-screen bg-[#f8fafc] flex flex-col font-sans selection:bg-blue-100 selection:text-blue-900">
            {/* Global Navigation */}
            <Navbar currentView={view} onNavigate={setView} />

            {/* Dynamic Viewport */}
            <main className="flex-1 flex flex-col">
                {renderContent()}
            </main>

            {/* Research Branding Footer */}
            <Footer />
        </div>
    );
};

export default App;
