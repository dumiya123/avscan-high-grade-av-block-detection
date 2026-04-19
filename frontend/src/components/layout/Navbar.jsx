import React, { useState } from 'react';
import { useSystemHealth } from "../../hooks/useSystemHealth";
import { useResponsive } from "../../hooks/useResponsive";
import { cn } from "../../utils/cn";
import { Activity, Menu, X } from 'lucide-react';

/**
 * Navbar — clinical-grade top navigation bar.
 * Matches the wireframe: logo left, nav tabs centre-right, status indicator right.
 * Responsive: Hamburger menu on mobile.
 */
const Navbar = ({ currentView, onNavigate }) => {
    const isOnline = useSystemHealth();
    const { isMobile } = useResponsive();
    const [isMenuOpen, setIsMenuOpen] = useState(false);

    const navItems = [
        { id: 'home', label: 'Home' },
        { id: 'instructions', label: 'Instructions' },
        { id: 'analysis', label: 'Analysis' },
        { id: 'about', label: 'About' },
    ];

    const handleNavigate = (id) => {
        onNavigate(id);
        setIsMenuOpen(false);
    };

    return (
        <nav className={cn("navbar", isMenuOpen && "navbar--open")}>
            <div className="navbar-inner">

                {/* Brand */}
                <button className="navbar-brand" onClick={() => handleNavigate('home')}>
                    <div className="navbar-logo">
                        <Activity className="w-4 h-4 text-white" />
                    </div>
                    <span className="navbar-brand-name">AtrionNet</span>
                </button>

                {/* Mobile Menu Toggle */}
                {isMobile && (
                    <button 
                        className="navbar-mobile-toggle"
                        onClick={() => setIsMenuOpen(!isMenuOpen)}
                        aria-label="Toggle menu"
                    >
                        {isMenuOpen ? <X size={20} /> : <Menu size={20} />}
                    </button>
                )}

                {/* Navigation tabs - Desktop */}
                {!isMobile && (
                    <div className="navbar-tabs">
                        {navItems.map((item) => (
                            <button
                                key={item.id}
                                onClick={() => handleNavigate(item.id)}
                                className={cn(
                                    'navbar-tab',
                                    currentView === item.id && 'navbar-tab--active'
                                )}
                            >
                                {item.label}
                            </button>
                        ))}
                    </div>
                )}

                {/* Right side — status - Desktop */}
                {!isMobile && (
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
                )}
            </div>

            {/* Mobile Navigation Overlay */}
            {isMobile && isMenuOpen && (
                <div className="navbar-mobile-menu">
                    <div className="mobile-nav-items">
                        {navItems.map((item) => (
                            <button
                                key={item.id}
                                onClick={() => handleNavigate(item.id)}
                                className={cn(
                                    'mobile-nav-item',
                                    currentView === item.id && 'mobile-nav-item--active'
                                )}
                            >
                                {item.label}
                            </button>
                        ))}
                    </div>
                    <div className="mobile-nav-footer">
                        <div className="mobile-status">
                            <span
                                className={cn(
                                    'status-dot',
                                    isOnline ? 'status-dot--online' : 'status-dot--offline'
                                )}
                            />
                            <span className="system-status-label">
                                {isOnline ? 'System Online' : 'System Offline'}
                            </span>
                        </div>
                        <span className="navbar-version">v1.0</span>
                    </div>
                </div>
            )}
        </nav>
    );
};

export default Navbar;
