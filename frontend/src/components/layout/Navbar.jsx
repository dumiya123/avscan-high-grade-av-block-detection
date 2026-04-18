import React from 'react';
import { useSystemHealth } from "../../hooks/useSystemHealth";
import { cn } from "../../utils/cn";
import { Activity } from 'lucide-react';

/**
 * Navbar — clinical-grade top navigation bar.
 * Matches the wireframe: logo left, nav tabs centre-right, status indicator right.
 */
const Navbar = ({ currentView, onNavigate }) => {
    const isOnline = useSystemHealth();

    const navItems = [
        { id: 'home', label: 'Home' },
        { id: 'instructions', label: 'Instructions' },
        { id: 'analysis', label: 'Analysis' },
        { id: 'about', label: 'About' },
    ];

    return (
        <nav className="navbar">
            <div className="navbar-inner">

                {/* Brand */}
                <button className="navbar-brand" onClick={() => onNavigate('home')}>
                    <div className="navbar-logo">
                        <Activity className="w-4 h-4 text-white" />
                    </div>
                    <span className="navbar-brand-name">AtrionNet</span>
                </button>

                {/* Navigation tabs */}
                <div className="navbar-tabs">
                    {navItems.map((item) => (
                        <button
                            key={item.id}
                            onClick={() => onNavigate(item.id)}
                            className={cn(
                                'navbar-tab',
                                currentView === item.id && 'navbar-tab--active'
                            )}
                        >
                            {item.label}
                        </button>
                    ))}
                </div>

                {/* Right side — status */}
                <div className="navbar-right">
                    <span className="system-status-label">
                        {isOnline ? 'System Online' : 'System Offline'}
                    </span>
                    <span
                        className={cn(
                            'status-dot',
                            isOnline ? 'status-dot--online' : 'status-dot--offline'
                        )}
                        title={isOnline ? 'Backend Connected' : 'Backend Offline'}
                    />
                    <span className="navbar-version">v1.0</span>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
