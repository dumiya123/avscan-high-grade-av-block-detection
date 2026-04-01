import React, { useState } from 'react';
import Navbar from './components/layout/Navbar';
import HomePage from './pages/HomePage';
import InstructionsPage from './pages/InstructionsPage';
import AnalysisPage from './pages/AnalysisPage';
import AboutPage from './pages/AboutPage';
import './App.css';

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
            {renderContent()}
        </>
    );
};

export default App;
