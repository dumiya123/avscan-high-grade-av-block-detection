"""
AtrionNet XAI: Clinical Explanation Generator (NLG)
This module translates the model's raw numbers into human-readable narratives.

Logic:
1.  **Clinical Mapping**: Matches class IDs to medical descriptions/severity.
2.  **Evidence Synthesis**: Combines wave counts and intervals into 'Finding' statements.
3.  **Advisory Logic**: Generates clinical recommendations based on diagnosis.
"""

import numpy as np
from typing import Dict, List


class ClinicalExplainer:
    """
    Explainable AI (XAI) NLG Engine.
    
    Why: A diagnosis is useless if a clinician doesn't trust it. This class 
    provides the "The AI decided this because..." narrative, citing specific 
    evidence like PR prolongation or AV dissociation.
    """
    
    def __init__(self):
        # The Clinician's Knowledge Base: Severity and Urgency mapping
        self.av_block_descriptions = {
            0: {'name': 'Normal Sinus Rhythm', 'description': 'Regular rhythm...', 'severity': 'None', 'urgency': 'Routine'},
            1: {'name': '1st Degree AV Block', 'description': 'Delayed conduction...', 'severity': 'Mild', 'urgency': 'Monitor'},
            2: {'name': '2nd Degree Type I', 'description': 'Wenckebach progression...', 'severity': 'Moderate', 'urgency': 'Evaluate'},
            3: {'name': '2nd Degree Type II', 'description': 'Sudden dropped beats...', 'severity': 'Serious', 'urgency': 'Urgent'},
            4: {'name': '3rd Degree Block', 'description': 'Complete AV dissociation...', 'severity': 'Severe', 'urgency': 'Emergency'},
            5: {'name': 'VT/AV Dissociation', 'description': 'Life-threatening rhythm...', 'severity': 'Critical', 'urgency': 'Emergency'}
        }
    
    def explain_segmentation(self, waves: Dict[str, List[int]], 
                           seg_confidence: float) -> str:
        """
        Translating the Object Detection Mask into a Narrative Summary.
        """
        total_p = len(waves['P_associated']) + len(waves['P_dissociated'])
        
        explanation = f"**Wave Detection Analysis:**\n\n"
        explanation += f"The model identified {total_p} P-waves in this ECG recording:\n"
        explanation += f"- {len(waves['P_associated'])} P-waves are **associated** with following QRS complexes\n"
        explanation += f"- {len(waves['P_dissociated'])} P-waves are **dissociated** (not followed by QRS)\n"
        explanation += f"- {len(waves['QRS'])} QRS complexes detected\n"
        
        if len(waves['P_dissociated']) > 0:
            explanation += f"[CRITICAL] Critical Finding: Dissociated P-waves detected, indicating potential AV block.\n\n"
        
        return explanation
    
    def explain_classification(self, av_block_class: int, confidence: float,
                               intervals: Dict[str, any]) -> str:
        """
        Generating the 'Supporting Evidence' Section.
        
        Logic:
        Takes the numerical intervals (PR, Rate) and converts them into 
        qualitative statements (e.g., "prolonged", "tachycardic").
        """
        block_info = self.av_block_descriptions[av_block_class]
        
        explanation = f"**Diagnosis: {block_info['name']}**\n\n"
        explanation += f"**Clinical Significance:**\n"
        explanation += f"- Severity: {block_info['severity']}\n"
        explanation += f"- Urgency: {block_info['urgency']}\n"
        explanation += f"- Confidence: {confidence:.1%}\n\n"
        
        explanation += f"**Supporting Evidence:**\n"
        
        if intervals['pr']:
            avg_pr = np.mean(intervals['pr'])
            explanation += f"- Average PR interval: {avg_pr:.1f} ms"
            explanation += " (prolonged)" if avg_pr > 200 else " (normal)"
            explanation += "\n"
        
        if intervals['p_qrs_ratio']:
            explanation += f"- P:QRS ratio: {intervals['p_qrs_ratio']:.2f}"
            explanation += " (suggests conduction block)" if intervals['p_qrs_ratio'] > 1.2 else " (normal)"
            explanation += "\n"
        
        return explanation
    
    def generate_full_explanation(self, 
                                 waves: Dict[str, List[int]],
                                 intervals: Dict[str, any],
                                 av_block_type: tuple,
                                 seg_confidence: float,
                                 attention_focus: str = None) -> str:
        """
        Generate complete clinical explanation
        
        Args:
            waves: Detected waves
            intervals: Calculated intervals
            av_block_type: (class_id, class_name, confidence)
            seg_confidence: Segmentation confidence
            attention_focus: Description of where model focused attention
            
        Returns:
            Complete explanation text
        """
        class_id, class_name, clf_confidence = av_block_type
        
        explanation = "# ECG Analysis Report\n\n"
        explanation += "=" * 60 + "\n\n"
        
        # Segmentation explanation
        explanation += self.explain_segmentation(waves, seg_confidence)
        explanation += "\n" + "-" * 60 + "\n\n"
        
        # Classification explanation
        explanation += self.explain_classification(class_id, clf_confidence, intervals)
        explanation += "\n" + "-" * 60 + "\n\n"
        
        # Model attention
        if attention_focus:
            explanation += f"**Model Attention:**\n\n"
            explanation += f"{attention_focus}\n\n"
            explanation += "-" * 60 + "\n\n"
        
        # Recommendations
        explanation += self.generate_recommendations(class_id)
        
        return explanation
    
    def generate_recommendations(self, av_block_class: int) -> str:
        """
        Generate clinical recommendations based on diagnosis
        
        Args:
            av_block_class: AV block class
            
        Returns:
            Recommendations text
        """
        recommendations = {
            0: [
                "Continue routine monitoring",
                "No immediate intervention required"
            ],
            1: [
                "Monitor for progression",
                "Review medications that may prolong PR interval",
                "Consider cardiology consultation if symptomatic"
            ],
            2: [
                "Cardiology consultation recommended",
                "Monitor for progression to higher-degree block",
                "Consider Holter monitoring"
            ],
            3: [
                "**Urgent cardiology consultation**",
                "High risk of progression to complete heart block",
                "Pacemaker evaluation may be indicated",
                "Avoid medications that slow AV conduction"
            ],
            4: [
                "**EMERGENCY - Immediate intervention required**",
                "Temporary pacing may be needed",
                "Permanent pacemaker strongly indicated",
                "Admit for continuous monitoring"
            ],
            5: [
                "**CRITICAL - Life-threatening arrhythmia**",
                "Immediate ACLS protocol",
                "Defibrillation may be required",
                "ICU admission"
            ]
        }
        
        recs = recommendations.get(av_block_class, ["Consult cardiology for further evaluation"])
        
        text = "**Recommendations:**\n\n"
        for i, rec in enumerate(recs, 1):
            text += f"{i}. {rec}\n"
        
        return text


if __name__ == "__main__":
    # Test explainer
    explainer = ClinicalExplainer()
    
    # Sample data
    waves = {
        'P_associated': [100, 800, 1500],
        'P_dissociated': [400, 1200],
        'QRS': [200, 900, 1600],
        'T': [350, 1050, 1750]
    }
    
    intervals = {
        'pr': [150, 160, 170],
        'rr': [700, 700, 700],
        'qt': [350, 360, 355],
        'p_qrs_ratio': 1.67
    }
    
    av_block_type = (4, "3rd degree", 0.92)
    
    explanation = explainer.generate_full_explanation(
        waves, intervals, av_block_type, seg_confidence=0.85
    )
    
    print(explanation)
