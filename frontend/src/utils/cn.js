import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Utility for merging tailwind classes safely using clsx and tailwind-merge.
 * This prevents class conflicts and allows for conditional styling in a clean way.
 */
export function cn(...inputs) {
    return twMerge(clsx(inputs));
}
