"""
Clinical Report Generator for AV Block Detection System
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

class ClinicalReport:
    """Generates clinical reports from model predictions"""
    
    def __init__(self):
        self.severity_levels = {
            'Normal': 'Low',
            '1st Degree': 'Low',
            '2nd Deg (Type I)': 'Moderate',
            '2nd Deg (Type II)': 'High',
            '3rd Degree': 'Critical',
            'VT': 'Critical'
        }
        
        self.recommendations = {
            'Normal': "Routine follow-up. No immediate intervention required.",
            '1st Degree': "Monitor PR interval. Check for underlying causes (medications, electrolyte imbalance).",
            '2nd Deg (Type I)': "Usually benign. Monitor for progression to higher grade block.",
            '2nd Deg (Type II)': "Risk of progression to complete heart block. Pacemaker evaluation recommended.",
            '3rd Degree': "Urgent cardiology consultation required. Permanent pacemaker likely indicated.",
            'VT': "Medical Emergency. Immediate intervention required."
        }

    def assess_severity(self, diagnosis):
        """Determine severity level based on diagnosis"""
        return self.severity_levels.get(diagnosis, 'Unknown')

    def get_recommendation(self, diagnosis):
        """Get clinical recommendation"""
        return self.recommendations.get(diagnosis, "Clinical correlation recommended.")

    def clean_line(self, line):
        """Clean markdown and special chars from a single line"""
        chars_to_remove = ['*', '#', '=', '_', '- ']
        for char in chars_to_remove:
            line = line.replace(char, '')
        return line.strip()

    def create_report(self, result, output_path):
        """
        Generate a professional clinical report in a single-page structured format
        """
        diagnosis = result['diagnosis']['av_block_type']
        confidence = result['diagnosis']['confidence']
        severity = result['diagnosis']['severity'] # Use refined severity from result
        urgency = result['diagnosis']['urgency']   # Use refined urgency from result
        explanation_raw = result['xai']['explanation']
        
        # Create figure
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        
        # Tracking pointer (starts at 0.95 and goes down)
        y = 0.95
        
        # 1. HEADER
        plt.text(0.5, y, "ATRIONNET CLINICAL ANALYSIS REPORT", ha='center', fontsize=18, weight='bold')
        y -= 0.02
        plt.text(0.5, y, "ECG-Based Arrhythmia Detection System", ha='center', fontsize=10, color='gray')
        y -= 0.02
        plt.plot([0.1, 0.9], [y, y], color='black', linewidth=1.5)
        y -= 0.04
        
        # 2. PATIENT & PHYSICIAN INFO (Table Layout)
        plt.text(0.1, y, "PATIENT INFORMATION", fontsize=10, weight='bold')
        plt.text(0.5, y, "EXAMINATION DETAILS", fontsize=10, weight='bold')
        y -= 0.02
        
        # Info Grid
        plt.text(0.1, y, f"Patient ID: ANON-12345", fontsize=9)
        plt.text(0.5, y, f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=9)
        y -= 0.015
        plt.text(0.1, y, f"Ref. Physician: Dr. Cardiologist", fontsize=9)
        plt.text(0.5, y, f"Report ID: ATR-{datetime.now().strftime('%m%H%S')}", fontsize=9)
        y -= 0.02
        plt.plot([0.1, 0.9], [y, y], color='lightgray', linewidth=0.5)
        y -= 0.04
        
        # 3. CLINICAL FINDINGS (Bold Section)
        plt.text(0.1, y, "PRIMARY DIAGNOSIS", fontsize=12, weight='bold', color='#1a1a1a')
        y -= 0.03
        
        # Color mapping for background shading
        colors = {
            'None': '#f8f9fa', 
            'Mild': '#f0f9ff', 
            'Moderate': '#fffbf0', 
            'Moderate-Severe': '#fff5f0',
            'Severe': '#fff0f0', 
            'Critical': '#fff0f0'
        }
        plt.gca().add_patch(plt.Rectangle((0.1, y-0.04), 0.8, 0.06, color=colors.get(severity, '#f8f9fa'), zorder=0))
        
        plt.text(0.15, y-0.01, diagnosis.upper(), fontsize=14, weight='bold', color='#c0392b' if severity in ['Severe', 'Critical', 'Moderate-Severe'] else '#2c3e50')
        y -= 0.06
        
        # Confidence Metrics
        plt.text(0.1, y, f"Detection Confidence: {confidence:.2%}", fontsize=10)
        plt.text(0.5, y, f"Clinical Severity: {severity.upper()}", fontsize=10, weight='bold', color='red' if severity in ['Severe', 'Critical', 'Moderate-Severe'] else 'black')
        y -= 0.04
        
        # 4. QUANTITATIVE METRICS (Tabular)
        plt.text(0.1, y, "QUANTITATIVE METRICS", fontsize=11, weight='bold')
        y -= 0.01
        plt.plot([0.1, 0.4], [y, y], color='black', linewidth=0.8)
        y -= 0.03
        
        # Recalculate HR using QRS centers if available (fs=500)
        qrs = result['waves']['QRS']
        hr_text = "N/A"
        if len(qrs) > 1:
            avg_rr_samples = np.mean(np.diff(qrs))
            hr = (60 * 500) / avg_rr_samples
            hr_text = f"{int(hr)} bpm"

        metrics = [
            ("Heart Rate", hr_text),
            ("P:QRS Ratio", f"{result['intervals']['p_qrs_ratio']:.2f}"),
            ("Avg PR Interval", f"{result['intervals'].get('pr', []) and np.mean(result['intervals']['pr']):.0f} ms" if result['intervals'].get('pr') else "N/A"),
            ("P-waves (Assoc)", f"{len(result['waves']['P_associated'])}"),
            ("P-waves (Dissoc)", f"{len(result['waves']['P_dissociated'])}"),
            ("Seg. Consistency", f"{result['seg_confidence']:.1%}")
        ]
        
        # 3 per row
        for i, (label, val) in enumerate(metrics):
            col = i % 3
            if i > 0 and col == 0: y -= 0.04
            x_pos = 0.1 + (col * 0.28)
            plt.text(x_pos, y, label, fontsize=8, color='gray')
            plt.text(x_pos, y-0.015, val, fontsize=11, weight='bold')
        
        y -= 0.06
        
        # 5. DETAILED ANALYSIS (Clean text)
        plt.text(0.1, y, "DETAILED SYSTEM ANALYSIS", fontsize=11, weight='bold')
        y -= 0.01
        plt.plot([0.1, 0.4], [y, y], color='black', linewidth=0.8)
        y -= 0.03
        
        # Parse explanation lines
        raw_lines = explanation_raw.split('\n')
        import textwrap
        wrapper = textwrap.TextWrapper(width=100)
        
        start_printing = False
        for line in raw_lines:
            line = self.clean_line(line)
            if "Detection Analysis" in line: start_printing = True
            if not start_printing or not line or "Analysis Report" in line or "===" in line: continue
            if "Recommendations:" in line: break # Stop before recommendations part as we handle it below
            
            wrapped = wrapper.wrap(line)
            for w_line in wrapped:
                if y < 0.25: break # Prevent overflow
                plt.text(0.1, y, w_line, fontsize=9.5, linespacing=1.2)
                y -= 0.018
            y -= 0.005 # Spacer between paragraphs
            
        # 6. RECOMMENDATIONS (Bottom Sticky Section)
        y = 0.22 # Fixed bottom position to ensure it's always visible
        plt.plot([0.1, 0.9], [y+0.02, y+0.02], color='gray', linewidth=0.5)
        plt.text(0.1, y, "CLINICAL RECOMMENDATIONS & URGENCY", fontsize=11, weight='bold')
        y -= 0.03
        
        plt.text(0.1, y, f"URGENCY PROTOCOL: {urgency.upper()}", fontsize=10, weight='bold', color='#c0392b' if 'Emergency' in urgency or 'Urgent' in urgency else '#2c3e50')
        y -= 0.025
        
        # Simple recommendation summary
        rec_sum = self.get_recommendation(diagnosis)
        wrapped_rec = textwrap.wrap(rec_sum, width=105)
        for r_line in wrapped_rec:
            plt.text(0.1, y, f"• {r_line}", fontsize=10, style='italic')
            y -= 0.018
            
        # 7. FOOTER
        plt.text(0.5, 0.05, "ELECTRONICALLY GENERATED BY ATRIONNET AI - NO SIGNATURE REQUIRED", ha='center', fontsize=8, color='gray')
        plt.text(0.5, 0.035, "This report is for clinical research purposes and should be verified by a board-certified cardiologist.", ha='center', fontsize=7, color='gray')
        
        plt.axis('off')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        return output_path

