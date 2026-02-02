import React from 'react';
import { cn } from "../../utils/cn";

/**
 * Badge component for small status indicators or labels.
 */
const Badge = ({ children, variant = 'default', className }) => {
    const variants = {
        default: 'bg-slate-100 text-slate-500',
        primary: 'bg-blue-50 text-blue-700 border border-blue-100',
        success: 'bg-green-500 text-white',
        danger: 'bg-red-600 text-white',
        warning: 'bg-orange-500 text-white',
        info: 'bg-indigo-50 text-indigo-700'
    };

    return (
        <span className={cn(
            'inline-flex items-center px-2 py-0.5 rounded text-[10px] font-black uppercase tracking-widest',
            variants[variant],
            className
        )}>
            {children}
        </span>
    );
};

export default Badge;
