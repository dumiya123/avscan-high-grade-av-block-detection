import React from 'react';
import { ShieldAlert, Cpu } from 'lucide-react';
import Badge from '../ui/Badge';

/**
 * Footer component focused on research branding and clinical context.
 */
const Footer = () => {
    const pillars = [
        {
            label: 'PRECISION',
            content: 'Specialized in micro-segmentation of dissociated P-waves at 500Hz resolution for high-grade block detection.'
        },
        {
            label: 'TRANSPARENCY',
            content: 'Integrated XAI framework utilizing Grad-CAM and Spatial Attention to provide clinically verifiable diagnostic logic.'
        },
        {
            label: 'INNOVATION',
            content: 'Implementation of anchor-free instance segmentation to identify independent atrial-ventricular activation patterns.'
        }
    ];

    return (
        <footer className="p-12 text-center text-slate-400 text-[9px] border-t bg-slate-50 uppercase font-black tracking-[0.2em] mt-auto">
            <div className="max-w-5xl mx-auto">

                {/* Clinical & Technical Headers */}
                <div className="flex flex-col md:flex-row items-center justify-center gap-4 md:gap-12 mb-10 opacity-70">
                    <div className="flex items-center gap-3">
                        <ShieldAlert className="w-5 h-5 text-blue-600" />
                        <div className="text-left">
                            <p className="text-[10px] font-black text-slate-600">Clinical Decision Support</p>
                            <p className="text-[8px] font-bold text-slate-400 normal-case tracking-normal">Class II Analytical Interface</p>
                        </div>
                    </div>
                    <div className="hidden md:block w-px h-8 bg-slate-300"></div>
                    <div className="flex items-center gap-3">
                        <Cpu className="w-5 h-5 text-indigo-600" />
                        <div className="text-left">
                            <p className="text-[10px] font-black text-slate-600">AtrionNet Neural Engine</p>
                            <p className="text-[8px] font-bold text-slate-400 normal-case tracking-normal">Multi-Task Attention Architecture</p>
                        </div>
                    </div>
                </div>

                {/* Mission & Purpose Pillars */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12 max-w-4xl mx-auto px-4 border-y border-slate-200 py-10 border-dashed text-slate-500">
                    {pillars.map((pillar) => (
                        <div key={pillar.label} className="space-y-2 text-center md:text-left">
                            <p className="text-blue-600 text-[10px] font-black tracking-widest">{pillar.label}</p>
                            <p className="normal-case text-[11px] leading-relaxed italic font-medium">
                                {pillar.content}
                            </p>
                        </div>
                    ))}
                </div>

                {/* Research Context Metadata */}
                <div className="flex flex-wrap items-center justify-center gap-6 text-[8px] tracking-[0.2em] text-slate-400 mb-10 opacity-60">
                    <Badge>FOR RESEARCH USE ONLY</Badge>
                    <span>PROTOCOL ID: ATR-9</span>
                    <span>STUDY YEAR: 2026</span>
                    <span>BUILD: 1.0.4-STABLE</span>
                </div>

                {/* Final Attribution Signature */}
                <div className="space-y-4 pt-4 border-t border-slate-200">
                    <p className="text-slate-500 font-black tracking-[0.4em] text-[10px]">© 2026 AVSCAN • ADVANCED ATRIOVENTRICULAR ANALYSIS</p>
                    <div className="flex flex-col items-center gap-1">
                        <p className="normal-case font-bold text-slate-600 tracking-normal text-[12px]">
                            Designed and Developed by <span className="text-blue-700 font-black px-2 py-0.5 bg-blue-50 rounded">Dumindu Induwara Gamage</span>
                        </p>
                        <p className="normal-case text-[10px] text-slate-400 font-medium tracking-normal italic">
                            Final Year Research Project | Computer Science & Engineering
                        </p>
                    </div>
                </div>
            </div>
        </footer>
    );
};

export default Footer;
