"""
End-to-end inference predictor for AV Block Detection
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.ecg_unet import ECGUNet
from src.data.preprocessing import bandpass_filter, normalize_signal
from src.analysis.temporal_analyzer import TemporalAnalyzer
from src.xai.gradcam import GradCAM
from src.xai.explainer import ClinicalExplainer
from src.utils import get_device, load_checkpoint
from src.reports.report_generator import ClinicalReport
import matplotlib.pyplot as plt


class AVBlockPredictor:
    """
    Complete inference pipeline for AV block detection with XAI
    """
    
    def __init__(self, checkpoint: str, fs: int = 500):
        """
        Args:
            checkpoint: Path to model checkpoint
            fs: Sampling frequency
        """
        self.fs = fs
        self.device = get_device()
        
        # Load model
        print("Loading model...")
        self.model = ECGUNet(
            in_channels=1,
            num_seg_classes=5,
            num_clf_classes=6,
            use_attention=True
        ).to(self.device)
        
        load_checkpoint(checkpoint, self.model, device=self.device)
        self.model.eval()
        
        # Initialize components
        self.temporal_analyzer = TemporalAnalyzer(fs=fs)
        self.explainer = ClinicalExplainer()
        self.reporter = ClinicalReport()
        
        # Setup Grad-CAM
        target_layer = self.model.enc4.conv
        self.gradcam = GradCAM(self.model, target_layer)
        
        print("Predictor ready!")
    
    def preprocess(self, ecg_signal: np.ndarray) -> torch.Tensor:
        """
        Preprocess raw ECG signal
        
        Args:
            ecg_signal: Raw ECG signal
            
        Returns:
            Preprocessed tensor (1, 1, seq_len)
        """
        # Apply bandpass filter
        filtered = bandpass_filter(ecg_signal, fs=self.fs)
        
        # Normalize
        normalized = normalize_signal(filtered)
        
        # Convert to tensor
        tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0)
        
        # Pad or crop to 5000 samples
        if tensor.shape[2] < 5000:
            pad_length = 5000 - tensor.shape[2]
            tensor = torch.nn.functional.pad(tensor, (0, pad_length))
        elif tensor.shape[2] > 5000:
            tensor = tensor[:, :, :5000]
        
        return tensor
    
    def predict(self, ecg_signal: np.ndarray, generate_report: bool = True) -> dict:
        """
        Run complete inference pipeline
        
        Args:
            ecg_signal: Raw ECG signal (numpy array)
            generate_report: Whether to generate full clinical report
            
        Returns:
            Dictionary with all results
        """
        # Preprocess
        input_tensor = self.preprocess(ecg_signal).to(self.device)
        
        # Run model
        with torch.no_grad():
            seg_pred, clf_pred = self.model(input_tensor)
        
        # Get predictions
        seg_mask = torch.argmax(seg_pred, dim=1)[0].cpu().numpy()
        clf_probs = torch.softmax(clf_pred, dim=1)[0].cpu().numpy()
        clf_class = torch.argmax(clf_pred, dim=1)[0].item()
        
        # Temporal analysis
        temporal_results = self.temporal_analyzer.analyze(seg_mask)
        
        # Grad-CAM
        gradcam_heatmap = self.gradcam.generate_heatmap(
            input_tensor, target_class=clf_class, task='classification'
        )
        
        # Get attention maps
        attention_maps = self.model.get_attention_maps(input_tensor)
        
        # Calculate confidence
        seg_confidence = self._calculate_seg_confidence(seg_pred[0])
        clf_confidence = clf_probs[clf_class]
        
        # Get refined classification from temporal analysis
        refined_class, refined_label, refined_confidence = temporal_results['av_block_type']
        
        # Prepare results - Prioritize refined analysis for clinical fields
        results = {
            'segmentation': seg_mask,
            'seg_confidence': seg_confidence,
            'waves': temporal_results['waves'],
            'intervals': temporal_results['intervals'],
            'diagnosis': {
                'av_block_type': refined_label,
                'av_block_class': refined_class,
                'nn_class': clf_class,
                'confidence': float(max(clf_confidence, refined_confidence)),
                'severity': self.explainer.av_block_descriptions[refined_class]['severity'],
                'urgency': self.explainer.av_block_descriptions[refined_class]['urgency']
            },
            'xai': {
                'gradcam': gradcam_heatmap,
                'attention_maps': attention_maps,
                'explanation': None
            }
        }
        
        # Generate explanation
        if generate_report:
            print("Generating explanation...")
            explanation = self.explainer.generate_full_explanation(
                waves=temporal_results['waves'],
                intervals=temporal_results['intervals'],
                av_block_type=(clf_class, results['diagnosis']['av_block_type'], clf_confidence),
                seg_confidence=seg_confidence
            )
            results['xai']['explanation'] = explanation
        
        return results
    
    def _calculate_seg_confidence(self, seg_logits: torch.Tensor) -> float:
        """
        Calculate segmentation confidence
        
        Args:
            seg_logits: Segmentation logits (num_classes, seq_len)
            
        Returns:
            Average confidence score
        """
        probs = torch.softmax(seg_logits, dim=0)
        max_probs = torch.max(probs, dim=0)[0]
        return max_probs.mean().item()
    
    def visualize_prediction(self, ecg_signal: np.ndarray, results: dict, 
                           save_path: str = None):
        """
        Create comprehensive visualization of prediction
        
        Args:
            ecg_signal: Original ECG signal
            results: Prediction results
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 1, hspace=0.3)
        
        time = np.arange(len(ecg_signal)) / self.fs
        
        # 1. ECG with segmentation
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(time, ecg_signal, 'k-', linewidth=0.8, alpha=0.7, label='ECG')
        
        # Overlay segmentation
        seg_mask = results['segmentation']
        class_colors = {
            1: ('blue', 'P-assoc'),
            2: ('red', 'P-dissoc'),
            3: ('green', 'QRS'),
            4: ('orange', 'T-wave')
        }
        
        for class_id, (color, label) in class_colors.items():
            mask = (seg_mask == class_id)
            if mask.any():
                ax1.fill_between(time, ecg_signal.min(), ecg_signal.max(),
                               where=mask, alpha=0.3, color=color, label=label)
        
        ax1.set_ylabel('Amplitude (mV)')
        ax1.set_title('ECG with Segmentation')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Grad-CAM heatmap
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(time, ecg_signal, 'k-', linewidth=0.8, alpha=0.5)
        
        gradcam = results['xai']['gradcam']
        im = ax2.imshow(gradcam.reshape(1, -1), cmap='jet', aspect='auto',
                       extent=[time[0], time[-1], ecg_signal.min(), ecg_signal.max()],
                       alpha=0.5)
        
        ax2.set_ylabel('Amplitude (mV)')
        ax2.set_title('Grad-CAM Heatmap (Model Attention)')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(im, ax=ax2, label='Attention')
        
        # 3. Temporal intervals
        ax3 = fig.add_subplot(gs[2])
        
        if results['intervals']['pr']:
            pr_intervals = results['intervals']['pr']
            ax3.plot(pr_intervals, 'bo-', label=f'PR intervals (avg: {np.mean(pr_intervals):.1f} ms)')
            ax3.axhline(y=200, color='r', linestyle='--', label='Normal limit (200 ms)')
        
        ax3.set_xlabel('Beat number')
        ax3.set_ylabel('Interval (ms)')
        ax3.set_title('PR Intervals Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Diagnosis summary
        ax4 = fig.add_subplot(gs[3])
        ax4.axis('off')
        
        diagnosis_text = f"""
        DIAGNOSIS: {results['diagnosis']['av_block_type']}
        Confidence: {results['diagnosis']['confidence']:.1%}
        Severity: {results['diagnosis']['severity']}
        Urgency: {results['diagnosis']['urgency']}
        
        Wave Counts:
        - P-waves (associated): {len(results['waves']['P_associated'])}
        - P-waves (dissociated): {len(results['waves']['P_dissociated'])}
        - QRS complexes: {len(results['waves']['QRS'])}
        - T-waves: {len(results['waves']['T'])}
        
        P:QRS Ratio: {results['intervals']['p_qrs_ratio']:.2f}
        """
        
        ax4.text(0.1, 0.5, diagnosis_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.suptitle('AV Block Detection - Complete Analysis', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        return fig
    
    def save_report(self, results: dict, output_path: str):
        """
        Save results to file using professional formatting
        
        Args:
            results: Prediction results
            output_path: Output file path (.txt or .pdf)
        """
        output_path = Path(output_path)
        
        if output_path.suffix == '.txt':
            # Save as text
            with open(output_path, 'w') as f:
                f.write(results['xai']['explanation'])
            print(f"Report saved to {output_path}")
        
        elif output_path.suffix == '.pdf':
            # Use professional report generator
            self.reporter.create_report(results, str(output_path))
            print(f"Professional Clinical Report saved to {output_path}")
        
        else:
            raise ValueError(f"Unsupported format: {output_path.suffix}")


if __name__ == "__main__":
    # Test predictor
    print("Testing AV Block Predictor...")
    
    # Create synthetic ECG
    t = np.arange(5000) / 500
    ecg_signal = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.random.randn(5000)
    
    # Note: This requires a trained model checkpoint
    # predictor = AVBlockPredictor(checkpoint='checkpoints/best_model.pth')
    # results = predictor.predict(ecg_signal)
    # predictor.visualize_prediction(ecg_signal, results, save_path='test_prediction.png')
    
    print("✅ Predictor module ready (requires trained checkpoint)")
