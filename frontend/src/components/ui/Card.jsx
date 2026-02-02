import React from 'react';
import { cn } from "../../utils/cn";

/**
 * Standardized Card component with glassmorphism support.
 */
const Card = ({ children, className, variant = 'default', padding = 'md' }) => {
    const paddings = {
        none: 'p-0',
        sm: 'p-4',
        md: 'p-6',
        lg: 'p-10'
    };

    const variants = {
        default: 'bg-white border border-slate-200 shadow-[0_10px_30px_rgba(0,0,0,0.04)]',
        glass: 'glass border border-white/20 backdrop-blur-md',
        interactive: 'bg-white border border-slate-100 shadow-sm hover:shadow-md transition-shadow'
    };

    return (
        <div className={cn(
            'rounded-3xl overflow-hidden',
            variants[variant],
            paddings[padding],
            className
        )}>
            {children}
        </div>
    );
};

export default Card;
