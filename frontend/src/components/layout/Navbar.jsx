import React from 'react';
import { useSystemHealth } from "../../hooks/useSystemHealth";
import { cn } from "../../utils/cn";

/**
 * Navbar component extracted from the main app logic.
 * Handles primary navigation and global system status visibility.
 */
const Navbar = ({ currentView, onNavigate }) => {
    const isOnline = useSystemHealth();

    const navItems = [
        { id: 'home', label: 'Home' },
        { id: 'instructions', label: 'Instructions' },
        { id: 'analysis', label: 'Upload ECG' },
    ];

    return (
        <nav className="bg-white/80 backdrop-blur-md sticky top-0 z-50 border-b border-slate-100">
            <div className="container mx-auto px-6 py-4 flex justify-between items-center">
                {/* Brand Logo */}
                <div
                    className="flex items-center gap-2 cursor-pointer"
                    onClick={() => onNavigate('home')}
                >
                    <div className="w-10 h-10 border-2 border-blue-600 rounded-full flex items-center justify-center">
                        <div className="w-6 h-6 border-b-2 border-blue-600 rounded-full animate-spin-slow"></div>
                    </div>
                    <h1 className="text-2xl font-black text-blue-700 tracking-tight">AVScan</h1>
                </div>

                {/* Navigation Links */}
                <div className="flex items-center gap-4">
                    {navItems.map((item) => (
                        <button
                            key={item.id}
                            onClick={() => onNavigate(item.id)}
                            className={cn(
                                "px-6 py-2 rounded-lg font-bold text-sm transition-all shadow-sm border",
                                currentView === item.id
                                    ? "bg-blue-600 text-white shadow-blue-200 border-blue-600"
                                    : "bg-white text-slate-700 border-transparent hover:border-slate-200"
                            )}
                        >
                            {item.label}
                        </button>
                    ))}

                    {/* System Health Indicator */}
                    <div
                        className={cn(
                            "ml-4 w-3 h-3 rounded-full transition-shadow",
                            isOnline
                                ? "bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.5)]"
                                : "bg-red-500"
                        )}
                        title={isOnline ? 'System Online' : 'System Offline'}
                    />
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
