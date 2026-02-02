"""
AtrionNet Diagnostic Engine: Temporal Relationship Analyzer
This module applies clinical reasoning to the the model's segmentation output.

Logic:
1.  **Interval Extraction**: Converts pixel-wise masks into time-stamped intervals.
2.  **Rhythm Math**: Calculates PR (Atrial-Ventricular delay), RR (Heart Rate), and P:QRS ratios.
3.  **Heuristic Classifiers**: Applies clinical decision trees to confirm high-grade AV blocks.
"""

import numpy as np
from typing import Dict, List, Tuple
import torch


class TemporalAnalyzer:
    """
    Expert System for Clinical ECG Interpretation.
    
    Why this exists:
    While the neural network is great at pattern recognition, clinical medicine 
    relies on specific interval criteria (e.g., "PR > 200ms"). This class 
    validates the AI's output against these established medical standards.
    """
    
    def __init__(self, fs: int = 500):
        self.fs = fs
    
    def extract_waves(self, seg_mask: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
        """
        Mask-to-Interval Transformation.
        
        Logic:
        Iterates through the 1D mask and finds contiguous blocks of the same 
        class label. It uses `np.diff` to find the exact onset and offset 
        of every wave the model detected.
        """
        waves = {
            'P_associated': [], 'P_dissociated': [],
            'QRS': [], 'T': []
        }
        
        class_map = {
            1: 'P_associated', 2: 'P_dissociated',
            3: 'QRS', 4: 'T'
        }
        
        for class_id, wave_type in class_map.items():
            mask = (seg_mask == class_id).astype(int)
            
            # Find transitions from 0 to 1 (start) and 1 to 0 (end)
            diff = np.diff(np.concatenate([[0], mask, [0]]))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            for start, end in zip(starts, ends):
                waves[wave_type].append((int(start), int(end)))
        
        return waves
    
    def calculate_intervals(self, waves: Dict[str, List[Tuple[int, int]]]) -> Dict[str, any]:
        """
        Quantitative Rhythm Measurement.
        
        Metrics:
        - **PR Interval**: Captures the delay between atrial and ventricular contraction.
        - **RR Interval**: Used to calculate Heart Rate and rhythm regularity.
        - **P:QRS Ratio**: Identifies missing beats (e.g., 2:1 or 3:1 block).
        """
        intervals = {
            'pr': [], 'rr': [], 'qt': [], 'p_qrs_ratio': 0.0
        }
        
        # We use the mathematical center (peak approximation) of waves for measurements
        p_associated = [(s + e) // 2 for s, e in waves['P_associated']]
        p_dissociated = [(s + e) // 2 for s, e in waves['P_dissociated']]
        qrs = [(s + e) // 2 for s, e in waves['QRS']]
        t_waves = [(s + e) // 2 for s, e in waves['T']]
        
        # Calculate PR (P-wave start to QRS start approximation)
        for p_idx in p_associated:
            following_qrs = [q for q in qrs if q > p_idx]
            if following_qrs:
                pr_ms = ((following_qrs[0] - p_idx) / self.fs) * 1000
                intervals['pr'].append(pr_ms)
        
        # Calculate RR Variability (Critical for Mobitz Type I detection)
        if len(qrs) > 1:
            rr_ms = (np.diff(qrs) / self.fs) * 1000
            intervals['rr'] = rr_ms.tolist()
        
        # Calculate Atrial-Ventricular Coupling Efficiency
        total_p = len(p_associated) + len(p_dissociated)
        if len(qrs) > 0:
            intervals['p_qrs_ratio'] = total_p / len(qrs)
        
        return intervals
    
    def detect_av_block_type(self, waves: Dict[str, List[Tuple[int, int]]], 
                            intervals: Dict[str, any]) -> Tuple[int, str, float]:
        """
        The Diagnostic Decision Tree.
        
        This is where the AI's 'Dissociation Detection' is combined with 
        deterministic clinical rules.
        """
        p_associated = waves['P_associated']
        p_dissociated = waves['P_dissociated']
        qrs = waves['QRS']
        
        total_p = len(p_associated) + len(p_dissociated)
        total_qrs = len(qrs)
        
        if total_p == 0 or total_qrs == 0:
            return 0, "Normal", 0.5
        
        # Key Diagnostic Feature: What % of P-waves have 'abandoned' their QRS?
        dissociation_ratio = len(p_dissociated) / total_p if total_p > 0 else 0
        p_qrs_ratio = intervals['p_qrs_ratio']
        
        pr_intervals = intervals['pr']
        rr_intervals = intervals['rr']
        
        avg_pr = np.mean(pr_intervals) if pr_intervals else 0
        pr_variability = np.std(pr_intervals) if len(pr_intervals) > 1 else 0
        rr_variability = np.std(rr_intervals) if len(rr_intervals) > 1 else 0
        
        # 1. THIRD DEGREE (Complete Block): Total dissociation, high P:QRS.
        if dissociation_ratio > 0.7 and p_qrs_ratio > 1.3:
            confidence = min(0.9, dissociation_ratio * 0.8 + (p_qrs_ratio - 1) * 0.2)
            return 4, "3rd degree (Complete Heart Block)", confidence
        
        # 2. SECOND DEGREE TYPE I: Progressive "tiring" of the conduction (Wenckebach).
        if pr_variability > 30 and len(pr_intervals) > 2:
            if self._is_progressive_prolongation(pr_intervals):
                confidence = min(0.85, pr_variability / 50)
                return 2, "2nd degree Type I (Wenckebach)", confidence
        
        # 3. SECOND DEGREE TYPE II: Constant PR until a beat is suddenly dropped.
        if dissociation_ratio > 0.3 and pr_variability < 20:
            confidence = min(0.8, dissociation_ratio * 1.5)
            return 3, "2nd degree Type II (Mobitz II)", confidence
        
        # 4. FIRST DEGREE: 1:1 conduction, but the delay is just too long.
        if avg_pr > 200:
            confidence = min(0.9, (avg_pr - 200) / 100)
            return 1, "1st degree AV block", confidence
        
        return 0, "Normal sinus rhythm", 0.8
    
    def _is_progressive_prolongation(self, pr_intervals: List[float]) -> bool:
        """
        Check if PR intervals show progressive prolongation (Wenckebach pattern)
        
        Args:
            pr_intervals: List of PR intervals in ms
            
        Returns:
            True if progressive prolongation detected
        """
        if len(pr_intervals) < 3:
            return False
        
        # Check if there's an increasing trend
        increasing_count = 0
        for i in range(len(pr_intervals) - 1):
            if pr_intervals[i+1] > pr_intervals[i]:
                increasing_count += 1
        
        # At least 60% should be increasing
        return (increasing_count / (len(pr_intervals) - 1)) > 0.6
    
    def generate_findings(self, waves: Dict[str, List[Tuple[int, int]]], 
                         intervals: Dict[str, any],
                         av_block_type: Tuple[int, str, float]) -> List[str]:
        """
        Generate clinical findings as text
        
        Args:
            waves: Dictionary of wave intervals
            intervals: Dictionary of intervals
            av_block_type: Detected AV block type
            
        Returns:
            List of clinical findings
        """
        findings = []
        
        # Wave counts
        total_p = len(waves['P_associated']) + len(waves['P_dissociated'])
        findings.append(f"Detected {total_p} P-waves ({len(waves['P_associated'])} associated, {len(waves['P_dissociated'])} dissociated)")
        findings.append(f"Detected {len(waves['QRS'])} QRS complexes")
        findings.append(f"Detected {len(waves['T'])} T-waves")
        
        # P:QRS ratio
        findings.append(f"P:QRS ratio = {intervals['p_qrs_ratio']:.2f}")
        
        # PR interval
        if intervals['pr']:
            avg_pr = np.mean(intervals['pr'])
            findings.append(f"Average PR interval = {avg_pr:.1f} ms")
            
            if avg_pr > 200:
                findings.append("⚠️ Prolonged PR interval (>200 ms)")
        
        # RR interval
        if intervals['rr']:
            avg_rr = np.mean(intervals['rr'])
            hr = 60000 / avg_rr  # Heart rate
            findings.append(f"Average heart rate = {hr:.0f} bpm")
        
        # Diagnosis
        _, diagnosis, confidence = av_block_type
        findings.append(f"\n🔍 Diagnosis: {diagnosis} (confidence: {confidence:.1%})")
        
        return findings
    
    def analyze(self, seg_mask: np.ndarray) -> Dict:
        """
        Complete temporal analysis pipeline
        
        Args:
            seg_mask: Segmentation mask
            
        Returns:
            Dictionary with all analysis results
        """
        # Extract waves
        waves = self.extract_waves(seg_mask)
        
        # Calculate intervals
        intervals = self.calculate_intervals(waves)
        
        # Detect AV block
        av_block_type = self.detect_av_block_type(waves, intervals)
        
        # Generate findings
        findings = self.generate_findings(waves, intervals, av_block_type)
        
        return {
            'waves': waves,
            'intervals': intervals,
            'av_block_type': av_block_type,
            'findings': findings
        }


if __name__ == "__main__":
    # Test temporal analyzer
    print("Testing Temporal Analyzer...")
    
    # Create synthetic segmentation mask
    seg_mask = np.zeros(5000)
    
    # Add some P-waves (associated)
    seg_mask[100:120] = 1
    seg_mask[800:820] = 1
    seg_mask[1500:1520] = 1
    
    # Add some P-waves (dissociated)
    seg_mask[400:420] = 2
    seg_mask[1200:1220] = 2
    
    # Add QRS complexes
    seg_mask[200:250] = 3
    seg_mask[900:950] = 3
    seg_mask[1600:1650] = 3
    
    # Add T-waves
    seg_mask[350:400] = 4
    seg_mask[1050:1100] = 4
    seg_mask[1750:1800] = 4
    
    # Analyze
    analyzer = TemporalAnalyzer(fs=500)
    results = analyzer.analyze(seg_mask)
    
    print("\n📊 Analysis Results:")
    for finding in results['findings']:
        print(f"  {finding}")
