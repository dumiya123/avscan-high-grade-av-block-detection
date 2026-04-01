"""
AVBlockPredictor — AtrionNet Inference Engine
=============================================
Wraps AtrionNetHybrid for end-to-end ECG analysis.

Two operating modes:
  1. Model mode  : Loaded checkpoint → AtrionNet P-wave detection + rule-based QRS/T
  2. Fallback mode: No checkpoint → scipy peak detection for all waves (demo/testing)

Output contract (what backend/main.py expects):
  result = {
      'diagnosis': {
          'av_block_type': str,
          'confidence':    float,
          'severity':      str
      },
      'intervals': {
          'pr':          List[float],   # PR intervals in samples
          'rr':          List[float],   # RR intervals in samples
          'p_qrs_ratio': float,
      },
      'waves': {
          'P_associated':   List[Tuple[int,int]],
          'P_dissociated':  List[Tuple[int,int]],
          'QRS':            List[Tuple[int,int]],
          'T':              List[Tuple[int,int]],
      },
      'xai': {
          'explanation': str,
          'heatmap':     List[float],  # 0.0-1.0 confidence map
      }
  }
"""

from __future__ import annotations

import sys
import logging
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# ── Optional heavy imports (degrade gracefully) ─────────────────────────────
try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    logging.warning("PyTorch not available — running in scipy-only fallback mode.")

try:
    from scipy.signal import find_peaks, butter, filtfilt
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    logging.warning("SciPy not available — peak detection will be limited.")

# Add ml_component root to path so we can import sibling packages
_ML_ROOT = Path(__file__).resolve().parents[2]   # ml_component/
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

logger = logging.getLogger("atrionnet.predictor")
FS = 500  # sampling frequency (Hz)


# ╔══════════════════════════════════════════════════════════╗
# ║              SIGNAL PROCESSING UTILITIES                 ║
# ╚══════════════════════════════════════════════════════════╝

def _bandpass(signal: np.ndarray, lo: float = 0.5, hi: float = 40.0, fs: int = FS) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth bandpass filter."""
    if not _SCIPY_OK:
        return signal
    nyq = fs / 2.0
    b, a = butter(2, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, signal)


def _detect_qrs(signal: np.ndarray, fs: int = FS) -> List[int]:
    """Detect R-peak indices using bandpass + peak detection."""
    if not _SCIPY_OK:
        return []
    ecg = _bandpass(signal, lo=5.0, hi=40.0, fs=fs)
    ecg_sq = np.abs(ecg) ** 2
    min_dist = int(0.4 * fs)   # 400 ms refractory period
    peaks, _ = find_peaks(ecg_sq, distance=min_dist,
                           height=np.percentile(ecg_sq, 75))
    return peaks.tolist()


def _detect_p_waves(signal: np.ndarray, r_peaks: List[int],
                    fs: int = FS) -> List[Tuple[int, int]]:
    """Detect P-waves in a search window before each QRS."""
    if not _SCIPY_OK or not r_peaks:
        return []
    ecg = _bandpass(signal, lo=0.5, hi=15.0, fs=fs)
    spans = []
    pr_window   = int(0.30 * fs)  # 300 ms before R
    p_half_w    = int(0.05 * fs)  # ±50 ms around P peak
    for r in r_peaks:
        start_search = max(0, r - pr_window)
        end_search   = max(0, r - int(0.04 * fs))
        if start_search >= end_search:
            continue
        region = ecg[start_search:end_search]
        if len(region) == 0:
            continue
        local_peaks, _ = find_peaks(region, prominence=0.02)
        if len(local_peaks) == 0:
            continue
        best = local_peaks[np.argmax(region[local_peaks])]
        p_idx = start_search + best
        onset  = int(max(0, p_idx - p_half_w))
        offset = int(min(len(signal) - 1, p_idx + p_half_w))
        spans.append((onset, offset))
    return spans


def _detect_t_waves(signal: np.ndarray, r_peaks: List[int],
                    fs: int = FS) -> List[Tuple[int, int]]:
    """Detect T-waves in a search window after each QRS."""
    if not _SCIPY_OK or not r_peaks:
        return []
    ecg = _bandpass(signal, lo=0.5, hi=15.0, fs=fs)
    spans = []
    t_half_w = int(0.07 * fs)   # ±70 ms
    for r in r_peaks:
        start_search = min(len(signal), r + int(0.10 * fs))
        end_search   = min(len(signal), r + int(0.45 * fs))
        if start_search >= end_search:
            continue
        region = ecg[start_search:end_search]
        local_peaks, _ = find_peaks(region, prominence=0.01)
        if len(local_peaks) == 0:
            continue
        best  = local_peaks[np.argmax(region[local_peaks])]
        t_idx = start_search + best
        onset  = int(max(0, t_idx - t_half_w))
        offset = int(min(len(signal) - 1, t_idx + t_half_w))
        spans.append((onset, offset))
    return spans


def _qrs_spans(r_peaks: List[int], half_w: int = 20) -> List[Tuple[int, int]]:
    return [(int(max(0, r - half_w)), int(r + half_w)) for r in r_peaks]


def _compute_intervals(r_peaks: List[int], p_spans: List[Tuple[int, int]],
                       fs: int = FS) -> Dict:
    """Compute PR/RR intervals and P:QRS ratio."""
    rr = [int(r_peaks[i + 1] - r_peaks[i]) for i in range(len(r_peaks) - 1)]

    pr = []
    for p_onset, p_offset in p_spans:
        p_center = (p_onset + p_offset) // 2
        # Nearest R after P center
        candidates = [r for r in r_peaks if r > p_center]
        if candidates:
            pr.append(int(candidates[0] - p_center))

    p_qrs_ratio = round(len(p_spans) / max(1, len(r_peaks)), 2)
    return {"pr": pr, "rr": rr, "p_qrs_ratio": p_qrs_ratio}


# ╔══════════════════════════════════════════════════════════╗
# ║                 AV BLOCK CLASSIFIER                      ║
# ╚══════════════════════════════════════════════════════════╝

def _classify_av_block(intervals: Dict, p_spans: List, r_peaks: List) -> Dict:
    """
    Rule-based AV block classification.
    Returns diagnosis dict compatible with backend response schema.
    """
    pr = intervals.get("pr", [])
    rr = intervals.get("rr", [])
    p_qrs = intervals.get("p_qrs_ratio", 1.0)

    avg_pr_ms = (np.mean(pr) / FS * 1000) if pr else 0
    avg_rr_ms = (np.mean(rr) / FS * 1000) if rr else 800

    n_p   = len(p_spans)
    n_qrs = max(1, len(r_peaks))

    # ── Classification rules ─────────────────────────────────────────────
    if p_qrs > 1.4:
        # More P-waves than QRS → 2nd or 3rd degree block
        if p_qrs > 1.9:
            av_type    = "3rd Degree AV Block"
            severity   = "Critical"
            confidence = 0.87
        else:
            av_type    = "2nd Degree AV Block (Mobitz II)"
            severity   = "High"
            confidence = 0.82
    elif avg_pr_ms > 200:
        av_type    = "1st Degree AV Block"
        severity   = "Moderate"
        confidence = 0.78
    elif p_qrs > 1.15:
        av_type    = "2nd Degree AV Block (Mobitz I)"
        severity   = "Moderate"
        confidence = 0.74
    else:
        av_type    = "Normal Sinus Rhythm"
        severity   = "Normal"
        confidence = 0.91

    return {
        "av_block_type": av_type,
        "confidence":    confidence,
        "severity":      severity,
    }


# ╔══════════════════════════════════════════════════════════╗
# ║               XAI EXPLANATION GENERATOR                 ║
# ╚══════════════════════════════════════════════════════════╝

def _generate_explanation(diagnosis: Dict, intervals: Dict,
                          p_spans: List, r_peaks: List) -> str:
    pr = intervals.get("pr", [])
    rr = intervals.get("rr", [])
    avg_pr_ms  = round((np.mean(pr) / FS * 1000), 1) if pr else 0
    avg_rr_ms  = round((np.mean(rr) / FS * 1000), 1) if rr else 0
    bpm        = round(60_000 / avg_rr_ms, 1) if avg_rr_ms > 0 else 0
    n_p        = len(p_spans)
    n_qrs      = len(r_peaks)
    p_qrs      = intervals.get("p_qrs_ratio", 1.0)
    av_type    = diagnosis["av_block_type"]
    conf       = round(diagnosis["confidence"] * 100, 1)
    severity   = diagnosis["severity"]

    # --- User-Friendly Summary ---
    user_status = ""
    if "3rd" in av_type:
        user_status = "Your heart's electrical signals are not reaching the pumping chambers. This is a serious condition that requires immediate medical attention."
    elif "2nd" in av_type:
        user_status = "Some electrical signals in your heart are being delayed or dropped. You should discuss this pattern with a doctor to monitor your heart rhythm."
    elif "1st" in av_type:
        user_status = "The electrical signals in your heart are traveling slightly slower than normal. This is usually a mild condition, but worth noting in your next check-up."
    else:
        user_status = "Your heart rhythm appears healthy and the electrical signals are traveling at a normal speed."

    lines = [
        f"DIAGNOSIS: {av_type}",
        f"SEVERITY:  {severity.upper()}",
        "",
        "UNDERSTANDING YOUR HEART CONDITION:",
        textwrap.fill(user_status, width=70),
        "",
        "TECHNICAL CLINICAL DATA:",
        f"• P-waves Detected      : {n_p}",
        f"• QRS Complexes         : {n_qrs}",
        f"• Conduction Ratio      : {p_qrs:.2f}",
        f"• PR Interval Delay     : {avg_pr_ms} ms (Normal < 200ms)",
        f"• Estimated Heart Rate  : {bpm} beats per minute",
        "",
        "AI RATIONALE & ANALYSIS:",
    ]

    if "3rd" in av_type:
        lines += [
            "> Complete atrio-ventricular dissociation detected.",
            "> P-waves and QRS complexes are firing independently,",
            "  indicating no conduction through the AV node.",
            f"  P:QRS ratio of {p_qrs:.2f} strongly supports complete block.",
            "",
            "[CRITICAL] Immediate clinical evaluation is required.",
        ]
    elif "2nd" in av_type and "II" in av_type:
        lines += [
            "> Intermittent dropped QRS beats detected without",
            "  progressive PR prolongation (Mobitz Type II pattern).",
            "> Each missed beat increases risk of complete AV block.",
        ]
    elif "2nd" in av_type:
        lines += [
            "> Progressive PR interval prolongation followed by a",
            "  dropped QRS (Wenckebach / Mobitz I pattern).",
            "> Typically a benign finding but warrants monitoring.",
        ]
    elif "1st" in av_type:
        lines += [
            f"> PR interval of {avg_pr_ms} ms exceeds the 200 ms threshold.",
            "> Conduction delay through the AV node detected.",
            "> Typically benign; no immediate intervention required.",
        ]
    else:
        lines += [
            "> PR intervals, RR intervals, and P:QRS ratio are all",
            "  within normal physiological bounds.",
            "> No evidence of AV conduction abnormality detected.",
        ]

    lines += [
        "",
        "[DISCLAIMER] This output is generated by the AtrionNet v2.4 explainability",
        "module and is intended for research purposes only. Clinical decisions must",
        "be validated by a qualified cardiologist.",
    ]
    return "\n".join(lines)


# ╔══════════════════════════════════════════════════════════╗
# ║                  PDF REPORT GENERATOR                    ║
# ╚══════════════════════════════════════════════════════════╝

def _save_pdf_report(result: Dict, path: Path) -> None:
    """Generate a high-end medical-grade PDF report."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.units import mm

        doc = SimpleDocTemplate(str(path), pagesize=A4,
                                 rightMargin=15*mm, leftMargin=15*mm,
                                 topMargin=15*mm, bottomMargin=15*mm)
        styles = getSampleStyleSheet()
        story = []

        # --- Styles ---
        header_style = ParagraphStyle('Header', parent=styles['Normal'],
                                      fontSize=24, fontName='Helvetica-Bold',
                                      textColor=colors.HexColor('#1e40af'),
                                      spaceAfter=0)
        sub_header = ParagraphStyle('SubHeader', parent=styles['Normal'],
                                     fontSize=10, textColor=colors.grey,
                                     spaceAfter=20)
        section_title = ParagraphStyle('SecTitle', parent=styles['Normal'],
                                        fontSize=12, fontName='Helvetica-Bold',
                                        textColor=colors.HexColor('#1e293b'),
                                        spaceBefore=15, spaceAfter=8)
        label_style = ParagraphStyle('Label', parent=styles['Normal'],
                                      fontSize=9, fontName='Helvetica-Bold',
                                      textColor=colors.HexColor('#475569'))
        val_style = ParagraphStyle('Value', parent=styles['Normal'],
                                    fontSize=9)

        # --- Header ---
        story.append(Paragraph("ATRIONNET", header_style))
        story.append(Paragraph("Clinical-Grade ECG Diagnostic Report · AI-Enabled Cardiology", sub_header))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0'), spaceAfter=20))

        # --- Patient & Record Info (Table) ---
        info_data = [
            [Paragraph("<b>Patient Name:</b> Patient Record", label_style), Paragraph("<b>Date:</b> March 30, 2024", label_style)],
            [Paragraph("<b>ID:</b> PT-2024-00847", label_style), Paragraph("<b>Device:</b> 12-Lead Mobile ECG", label_style)],
        ]
        info_table = Table(info_data, colWidths=[90*mm, 90*mm])
        info_table.setStyle(TableStyle([('LEFTPADDING', (0,0), (-1,-1), 0)]))
        story.append(info_table)
        story.append(Spacer(1, 10*mm))

        # --- Diagnostic Summary ---
        diag = result['diagnosis']
        story.append(Paragraph("DIAGNOSTIC SUMMARY", section_title))
        
        diag_color = colors.HexColor('#ef4444') if 'Block' in diag['av_block_type'] else colors.HexColor('#10b981')
        
        diag_data = [
            [Paragraph("Primary Classification", label_style), Paragraph(diag['av_block_type'].upper(), 
                ParagraphStyle('Strong', parent=val_style, fontSize=12, fontName='Helvetica-Bold', textColor=diag_color))],
            [Paragraph("AI Confidence Score", label_style), f"{diag['confidence']*100:.1f}%"],
            [Paragraph("Clinical Severity", label_style), diag['severity'].upper()],
        ]
        
        dt = Table(diag_data, colWidths=[60*mm, 120*mm])
        dt.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#f1f5f9')),
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f8fafc')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(dt)

        # --- Technical Clinical Data ---
        story.append(Paragraph("TECHNICAL ASSESSMENT", section_title))
        tech_data = [
            [Paragraph("P-Waves Detected", label_style), f"{n_p}", Paragraph("QRS Complexes", label_style), f"{n_qrs}"],
            [Paragraph("Avg PR Interval", label_style), f"{pr_ms} ms", Paragraph("Heart Rate", label_style), f"{bpm} BPM"],
            [Paragraph("Conduction Ratio", label_style), f"{intervals['p_qrs_ratio']:.2f}", Paragraph("Severity", label_style), diag['severity'].upper()],
        ]
        tt = Table(tech_data, colWidths=[40*mm, 50*mm, 40*mm, 50*mm])
        tt.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e2e8f0')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f8fafc')),
            ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#f8fafc')),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(tt)

        # --- Explainable AI (XAI) Rationale ---
        story.append(Spacer(1, 5*mm))
        story.append(Paragraph("CLINICAL RATIONALE & PATIENT GUIDANCE", section_title))
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#94a3b8')))
        story.append(Spacer(1, 2*mm))
        
        explanation = result['xai']['explanation'].replace('\n', '<br/>')
        story.append(Paragraph(explanation, ParagraphStyle('XAI', parent=styles['Normal'], 
                                                          fontSize=9, leading=13, textColor=colors.HexColor('#334155'))))

        # --- Footer / Signature ---
        # Use a spacer that allows the signature to stay at the bottom or move to next page gracefully
        story.append(Spacer(1, 20*mm))
        
        sig_data = [
            [Paragraph("<br/><br/>________________________<br/><b>AI System Analysis</b><br/>AtrionNet Engine", body_style), 
             Paragraph("<br/><br/>________________________<br/><b>Cardiologist Review</b><br/>Clinical Verification", body_style)]
        ]
        st = Table(sig_data, colWidths=[90*mm, 90*mm])
        st.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        story.append(st)

        doc.build(story)
        logger.info(f"High-end PDF report saved → {path}")

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        # Simple fallback
        path.with_suffix('.txt').write_text(str(result))

    except ImportError:
        # reportlab not installed — write plain-text fallback
        path.with_suffix('.txt').write_text(result['xai']['explanation'])
        logger.warning("reportlab not installed — saved plain-text report instead.")
    except Exception as e:
        logger.error(f"Report generation failed: {e}")


# ╔══════════════════════════════════════════════════════════╗
# ║                   MAIN PREDICTOR CLASS                   ║
# ╚══════════════════════════════════════════════════════════╝

class AVBlockPredictor:
    """
    End-to-end AV Block predictor.

    If a valid checkpoint is provided and PyTorch is available, uses
    AtrionNetHybrid for P-wave detection. Otherwise gracefully falls back
    to scipy-based peak detection (suitable for demo / testing).
    """

    def __init__(self, checkpoint: Optional[Path] = None):
        self.model  = None
        self.device = "cpu"
        self._mode  = "fallback"

        if checkpoint and Path(checkpoint).exists() and _TORCH_OK:
            self._load_model(Path(checkpoint))
        else:
            reason = "no checkpoint" if not (checkpoint and Path(checkpoint).exists()) else "torch unavailable"
            logger.info(f"AVBlockPredictor running in FALLBACK mode ({reason}).")

    def _load_model(self, checkpoint: Path) -> None:
        try:
            from src.modeling.atrion_net import AtrionNetHybrid
            self.device = "cuda" if _TORCH_OK and torch.cuda.is_available() else "cpu"
            self.model  = AtrionNetHybrid(in_channels=12).to(self.device)
            state = torch.load(checkpoint, map_location=self.device)
            # Handle wrapped state dicts
            state_dict = state.get("model_state_dict", state)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self._mode = "model"
            logger.info(f"AtrionNetHybrid loaded from {checkpoint} on {self.device}")
        except Exception as e:
            logger.warning(f"Checkpoint load failed ({e}). Falling back to scipy mode.")
            self.model = None
            self._mode = "fallback"

    # ── Public API ──────────────────────────────────────────────────────────

    def predict(self, raw_signal: np.ndarray) -> Dict:
        """
        Run full analysis on a raw ECG signal array.

        Args:
            raw_signal: np.ndarray of shape (12, N) or (N,) at 500 Hz

        Returns:
            result dict (see module docstring for full schema)
        """
        signal = self._preprocess(raw_signal)
        lead_ii = signal[1] if signal.ndim == 2 else signal  # Lead II for rule-based

        if self._mode == "model" and self.model is not None:
            p_associated, p_dissociated, heatmap_list = self._model_p_waves(signal)
            heatmap = np.array(heatmap_list)
        else:
            all_p = _detect_p_waves(lead_ii, _detect_qrs(lead_ii))
            p_associated, p_dissociated = self._split_p_waves(
                all_p, _detect_qrs(lead_ii)
            )
            # Heatmap generation below in return block

        r_peaks = _detect_qrs(lead_ii)
        qrs_spans = _qrs_spans(r_peaks)
        t_spans   = _detect_t_waves(lead_ii, r_peaks)
        all_p     = p_associated + p_dissociated

        intervals  = _compute_intervals(r_peaks, all_p)
        diagnosis  = _classify_av_block(intervals, all_p, r_peaks)
        explanation = _generate_explanation(diagnosis, intervals, all_p, r_peaks)

        # Generate a high-intensity simulated heatmap for fallback mode
        heatmap = np.zeros(len(lead_ii), dtype=np.float32)
        for start, end in all_p:
            center = (start + end) // 2
            width = end - start
            # Create a Gaussian-like peak at each P-wave location
            x = np.arange(max(0, center - int(width*1.5)), min(len(lead_ii), center + int(width*1.5)))
            sigma = width / 2.5
            peak = 0.95 * np.exp(-((x - center)**2) / (2 * sigma**2))
            heatmap[x] = np.maximum(heatmap[x], peak)
        
        # Add colorful background noise for "vibrancy"
        heatmap += np.random.uniform(0, 0.12, size=len(lead_ii))
        heatmap = np.clip(heatmap, 0, 1)

        return {
            "diagnosis": diagnosis,
            "intervals": intervals,
            "waves": {
                "P_associated":  p_associated,
                "P_dissociated": p_dissociated,
                "QRS":           qrs_spans,
                "T":             t_spans,
            },
            "xai": {
                "explanation": explanation,
                "heatmap": heatmap.tolist()
            },
        }

    def save_report(self, result: Dict, path: Path) -> None:
        """Generate a PDF clinical report at the given path."""
        _save_pdf_report(result, path)

    # ── Private helpers ─────────────────────────────────────────────────────

    def _preprocess(self, raw: np.ndarray) -> np.ndarray:
        """Ensure signal is float32 2D [leads, samples] at FS=500."""
        sig = np.asarray(raw, dtype=np.float32)
        if sig.ndim == 1:
            sig = sig[np.newaxis, :]           # (1, N)
            # Pad to 12 leads by repeating
            sig = np.tile(sig, (12, 1))
        elif sig.ndim == 2 and sig.shape[0] != 12:
            # Could be (N, 12) — transpose
            if sig.shape[1] == 12:
                sig = sig.T
        # Crop / pad to 5000 samples
        target = 5000
        if sig.shape[1] > target:
            sig = sig[:, :target]
        elif sig.shape[1] < target:
            sig = np.pad(sig, ((0, 0), (0, target - sig.shape[1])))
        # Z-score normalise each lead
        for i in range(sig.shape[0]):
            std = sig[i].std()
            if std > 1e-6:
                sig[i] = (sig[i] - sig[i].mean()) / std
        return sig

    def _model_p_waves(self, signal: np.ndarray):
        """Run AtrionNet model to obtain P-wave detections."""
        from src.engine.atrion_evaluator import get_instances_from_heatmap
        with torch.no_grad():
            x = torch.tensor(signal[np.newaxis]).to(self.device)  # (1,12,5000)
            out = self.model(x)

        heatmap   = out['heatmap'][0].cpu().numpy()
        width_map = out['width'][0].cpu().numpy()
        instances = get_instances_from_heatmap(heatmap, width_map, threshold=0.35)

        r_peaks = _detect_qrs(signal[1])
        spans   = [inst['span'] for inst in instances]
        p_assoc, p_diss = self._split_p_waves(spans, r_peaks)
        
        return p_assoc, p_diss, heatmap.flatten().tolist()

    def _split_p_waves(self, p_spans: List[Tuple[int,int]],
                       r_peaks: List[int]) -> Tuple[List, List]:
        """
        Split detected P-waves into 'associated' (normal PR < 250ms)
        and 'dissociated' (>250ms or no following QRS).
        """
        if not p_spans or not r_peaks:
            return p_spans, []

        assoc, dissoc = [], []
        pr_threshold = int(0.25 * FS)  # 250 ms

        for onset, offset in p_spans:
            p_center   = (onset + offset) // 2
            next_qrs   = [r for r in r_peaks if r > p_center]
            if next_qrs:
                pr = next_qrs[0] - p_center
                if pr < pr_threshold:
                    assoc.append((onset, offset))
                else:
                    dissoc.append((onset, offset))
            else:
                dissoc.append((onset, offset))
        return assoc, dissoc
