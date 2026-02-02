import React from 'react';
import { cn } from "../../utils/cn";

/**
 * Professional, polymorphic Button component.
 * Supports multiple variants and sizes with consistent interactive states.
 */
const Button = ({
    children,
    variant = 'primary',
    size = 'md',
    className,
    icon: Icon,
    isLoading,
    ...props
}) => {
    const variants = {
        primary: 'bg-slate-900 text-white hover:bg-black shadow-[0_10px_20px_rgba(0,0,0,0.1)] active:scale-95',
        secondary: 'bg-white border-2 border-slate-100 text-slate-700 hover:border-slate-300 active:scale-95',
        outline: 'bg-transparent border-2 border-blue-600 text-blue-600 hover:bg-blue-50 active:scale-95',
        ghost: 'bg-transparent hover:bg-slate-50 text-slate-600',
        danger: 'bg-red-600 text-white hover:bg-red-700',
        link: 'bg-transparent text-blue-600 hover:underline p-0 h-auto'
    };

    const sizes = {
        sm: 'px-4 py-2 text-xs rounded-xl',
        md: 'px-6 py-3 text-sm rounded-2xl font-bold',
        lg: 'px-10 py-5 text-lg rounded-2xl font-bold'
    };

    return (
        <button
            className={cn(
                'inline-flex items-center justify-center gap-3 transition-all disabled:opacity-50 disabled:pointer-events-none',
                variants[variant],
                sizes[size],
                className
            )}
            disabled={isLoading}
            {...props}
        >
            {isLoading ? (
                <div className="w-5 h-5 border-2 border-current border-t-transparent rounded-full animate-spin" />
            ) : Icon && <Icon className={cn("w-5 h-5", size === 'lg' ? 'w-6 h-6' : 'w-5 h-5')} />}
            {children}
        </button>
    );
};

export default Button;
