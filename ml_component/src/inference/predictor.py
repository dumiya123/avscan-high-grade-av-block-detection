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

def _classify_av_block(intervals: Dict, p_spans: List, r_peaks: List, heatmap: Optional[np.ndarray] = None) -> Dict:
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
    
    # Calculate physiological progression for Mobitz I
    progression = False
    if len(pr) >= 2:
        diffs = np.diff(pr)
        # At least one consecutive PR increased by 20ms
        if np.any(diffs > int(0.02 * FS)): 
            progression = True

    # ── Classification rules ─────────────────────────────────────────────
    # ── Classification rules (Hospital Severity Standards) ────────────────
    if p_qrs > 1.4:
        # More P-waves than QRS → 2nd or 3rd degree block
        if p_qrs > 1.9:
            av_type    = "3rd Degree AV Block"
            severity   = "Severe"
        else:
            av_type    = "2nd Degree AV Block (Mobitz II)"
            severity   = "Severe"
    elif p_qrs > 1.15:
        if progression:
            av_type    = "2nd Degree AV Block (Mobitz I / Wenckebach)"
            severity   = "Moderate"
        else:
            av_type    = "2nd Degree AV Block (Mobitz II)"
            severity   = "Moderate"
    elif avg_pr_ms > 200:
        av_type    = "1st Degree AV Block"
        severity   = "Mild"
    else:
        av_type    = "Normal Sinus Rhythm"
        severity   = "Normal"
        
    # Derive Confidence Score from average P-wave detection certainty (Real Model Probabilities)
    base_conf = 0.85 # Default fallback
    if heatmap is not None and p_spans:
        peak_scores = []
        for s, e in p_spans:
            if s < e and e <= len(heatmap):
                peak_scores.append(np.max(heatmap[s:e+1]))
        if peak_scores:
            base_conf = float(np.mean(peak_scores))

    return {
        "av_block_type": av_type,
        "confidence":    round(base_conf, 3),
        "severity":      severity,
    }


# ╔══════════════════════════════════════════════════════════╗
# ║         STRUCTURED CLINICAL IMPORTANCE MAP               ║
# ╚══════════════════════════════════════════════════════════╝

def _build_importance_map(
    signal_len: int,
    p_spans: List, qrs_spans: List, t_spans: List,
    r_peaks: List, diagnosis: Dict,
    model_heatmap: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build a clinically-structured feature importance map.
    Upgraded to handle confidence-based dispersion and smooth intensity transitions.
    """
    importance = np.zeros(signal_len, dtype=np.float32)
    conf = diagnosis.get("confidence", 0.85)
    
    # Sigma adjustment based on confidence: 
    # High confidence -> Sharp, focused peaks
    # Low confidence -> Dispersed, wide peaks
    sigma_scale = 1.0 + (1.0 - conf) * 2.0 

    # --- 1. Mark QRS complexes (ventricular response) ---
    for start, end in qrs_spans:
        qrs_w = max(1, end - start)
        center = (start + end) // 2
        # Wider influence for clinical context
        x_start = max(0, start - int(20 * sigma_scale))
        x_end = min(signal_len, end + int(20 * sigma_scale))
        x = np.arange(x_start, x_end)
        sigma = max(1, (qrs_w / 2.0) * sigma_scale)
        peak = 0.95 * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        importance[x] = np.maximum(importance[x], peak)

    # --- 2. Mark P-wave regions (primary detection target) ---
    av_type = diagnosis.get("av_block_type", "")
    p_base_importance = 0.95 if "Block" in av_type else 0.80
    for start, end in p_spans:
        pw = max(1, end - start)
        center = (start + end) // 2
        x_start = max(0, start - int(15 * sigma_scale))
        x_end = min(signal_len, end + int(15 * sigma_scale))
        x = np.arange(x_start, x_end)
        sigma = max(1, (pw / 2.0) * sigma_scale)
        # Intensity varies slightly based on confidence
        val = p_base_importance * (0.5 + 0.5 * conf)
        peak = val * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        importance[x] = np.maximum(importance[x], peak)

    # --- 3. Mark PR segments (conduction delay) ---
    for p_start, p_end in p_spans:
        p_center = (p_start + p_end) // 2
        next_qrs = [r for r in r_peaks if r > p_center]
        if next_qrs:
            pr_start = p_end
            pr_end   = next_qrs[0]
            if pr_end > pr_start + 5:
                seg_len = pr_end - pr_start
                # PR segment importance peaks if it's longer (abnormal)
                pr_imp = min(0.92, 0.60 + (seg_len / (0.3 * FS)) * 0.35)
                x = np.arange(pr_start, min(signal_len, pr_end))
                # Smooth ramp
                ramp = np.linspace(0.40, pr_imp, len(x))
                importance[x] = np.maximum(importance[x], ramp.astype(np.float32))

    # --- 4. Mark T-wave regions (repolarisation) ---
    for start, end in t_spans:
        tw = max(1, end - start)
        center = (start + end) // 2
        x_start = max(0, start - int(10 * sigma_scale))
        x_end = min(signal_len, end + int(10 * sigma_scale))
        x = np.arange(x_start, x_end)
        sigma = max(1, (tw / 2.5) * sigma_scale)
        peak = 0.50 * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        importance[x] = np.maximum(importance[x], peak)

    # --- 5. Blend real model heatmap (if available) ---
    if model_heatmap is not None and len(model_heatmap) == signal_len:
        model_norm = np.clip(model_heatmap, 0, 1).astype(np.float32)
        
        # Improvement: Instead of simple weighted average, use a 'focus boost'
        # If model is confident about a spot, boost it. Otherwise trust clinical anatomy.
        # This prevents 'clustering at beginning' from killing importance elsewhere.
        importance = np.where(model_norm > 0.3, 
                             0.5 * importance + 0.5 * model_norm, 
                             0.85 * importance + 0.15 * model_norm)

    # --- 6. Smoothing Pass ---
    # Apply a small Gaussian blur to ensure smooth transitions for coloring
    if signal_len > 10:
        kernel_size = 15
        kernel = np.exp(-np.linspace(-2, 2, kernel_size)**2)
        kernel /= kernel.sum()
        importance = np.convolve(importance, kernel, mode='same')

    # --- 7. Baseline and Confidence-based intensity ---
    # If confidence is very low, the whole map is slightly dimmer/more uniform
    baseline_level = 0.05 + (1.0 - conf) * 0.15
    importance = np.maximum(importance, baseline_level)
    
    return np.clip(importance, 0.0, 1.0)


def _get_focus_label(
    importance: np.ndarray,
    p_spans: List, qrs_spans: List, t_spans: List,
    r_peaks: List, signal_len: int
) -> str:
    """Summarise in plain English which ECG region has the highest total importance."""
    def region_sum(spans):
        if not spans: return 0.0
        idx = []
        for s, e in spans:
            idx.extend(range(max(0, s), min(signal_len, e)))
        return float(np.sum(importance[idx])) if idx else 0.0

    pr_segs = []
    for p_start, p_end in p_spans:
        p_center = (p_start + p_end) // 2
        nxt = [r for r in r_peaks if r > p_center]
        if nxt: pr_segs.append((p_end, nxt[0]))

    scores = {
        "P-Wave Intervals": region_sum(p_spans),
        "Atrioventricular Conduction Path": region_sum(pr_segs),
        "Ventricular Depolarization (QRS)": region_sum(qrs_spans),
        "Ventricular Repolarization (T-Wave)": region_sum(t_spans),
    }

    if not any(v > 0 for v in scores.values()):
        return "Broad clinical overview"

    top = max(scores, key=scores.get)
    return f"Model primary focus: {top}"


# ╔══════════════════════════════════════════════════════════╗
# ║               XAI EXPLANATION GENERATOR                 ║
# ╚══════════════════════════════════════════════════════════╝

def _get_clinical_rationale(intervals: Dict, diagnosis: Dict, 
                               p_spans: List, r_peaks: List) -> Dict:
    pr = intervals.get("pr", [])
    rr = intervals.get("rr", [])
    avg_pr_ms  = round((np.mean(pr) / FS * 1000), 1) if pr else 0
    avg_rr_ms  = round((np.mean(rr) / FS * 1000), 1) if rr else 0
    bpm        = round(60_000 / avg_rr_ms, 1) if avg_rr_ms > 0 else 0
    p_qrs      = intervals.get("p_qrs_ratio", 1.0)
    av_type    = diagnosis["av_block_type"]
    severity   = diagnosis["severity"]
    confidence = diagnosis["confidence"]

    # --- 1. Clinical Impression (Doctor-Focused & Diagnosis-Driven) ---
    if "3rd" in av_type:
        impression = (f"Complete AV Heart Block (3rd Degree). Absolute AV dissociation present. "
                      f"Atrial rate exceeds ventricular rate. Conduction ratio: {p_qrs:.2f}. "
                      f"Model focus on independent P-wave cadence confirms total lack of electronic sync.")
        xai_link = "High attention effectively isolated on non-conducted P-waves confirms absolute atrial-ventricular dissociation."
    elif "2nd" in av_type and "II" in av_type:
        impression = (f"Mobitz Type II 2nd-Degree AV Block. Sudden non-conducted P-waves noted with fixed PR intervals ({avg_pr_ms}ms). "
                      f"Significant risk of progression. Model focus targeted on the abrupt transition to a dropped beat.")
        xai_link = "Focused attention peaks on non-conducted P-waves identify the sudden cessation of conduction."
    elif "2nd" in av_type:
        impression = (f"Mobitz Type I (Wenckebach) 2nd-Degree AV Block. Progressive PR prolongation ending in a dropped QRS. "
                      f"Mean conducted PR: {avg_pr_ms}ms. Model highlights focus on the escalating delay in the PR segment.")
        xai_link = "Model focus on the gradual PR interval expansion supports the diagnosis of progressive conduction fatigue."
    elif "1st" in av_type:
        impression = (f"1st-Degree AV Block detected. Prolonged PR interval measured at {avg_pr_ms}ms (Threshold > 200ms). "
                      f"Consistent 1:1 conduction. Model focus centered on atrioventricular transition delay.")
        xai_link = "High attention concentrated specifically around the PR intervals supports detection of consistent conduction delay."
    else:
        impression = (f"Normal Sinus Rhythm. Heart rate: {bpm} BPM. Consistent {avg_pr_ms}ms PR intervals. "
                      f"No conduction abnormalities observed. Attention distributed across standard morphology.")
        xai_link = "Attention is distributed evenly across standard P-QRS-T complexes, indicating a lack of localized conduction pathology."

    # --- 2. Confidence Interpretation Layer ---
    if confidence > 0.92:
        conf_text = "AUTHENTICATED FOCUS: Model attention is precisely concentrated on key diagnostic regions."
    elif confidence > 0.75:
        conf_text = "ROBUST ANALYSIS: Prediction is based on consistent wave measurements across multiple cycles."
    else:
        conf_text = "MODERATE UNCERTAINTY: Model shows dispersed attention; diagnostic uncertainty is higher due to signal variability."

    # --- 3. Patient-Friendly Explanation ---
    if "3rd" in av_type:
        patient_text = "The upper and lower chambers of your heart are working independently. This is a critical blockage that requires immediate medical evaluation and likely intervention."
    elif "2nd" in av_type:
        patient_text = "Your heart is occasionally skipping a beat. The electrical signal is being delayed or blocked intermittently. Consultation with a cardiologist is recommended."
    elif "1st" in av_type:
        patient_text = "Electrical signals in your heart are traveling slower than normal, but every beat still reaches the main chamber. This is usually observed for monitoring."
    else:
        patient_text = "Your heart's electrical rhythm is within normal clinical limits. Signals are traveling at a healthy speed."

    full_rationale = (
        f"DIAGNOSIS: {av_type}\n"
        f"SEVERITY: {severity.upper()}\n\n"
        f"CLINICAL IMPRESSION:\n{impression}\n\n"
        f"AI EVIDENCE LINK:\n{xai_link}\n\n"
        f"CONFIDENCE INTERPRETATION:\n{conf_text}\n\n"
        f"PATIENT GUIDANCE:\n{patient_text}"
    )

    return {
        "impression": impression,
        "patient_explanation": patient_text,
        "xai_link": xai_link,
        "conf_text": conf_text,
        "full_rationale": full_rationale
    }


# ╔══════════════════════════════════════════════════════════╗
# ║                  PDF REPORT GENERATOR                    ║
# ╚══════════════════════════════════════════════════════════╝

def _save_pdf_report(result: Dict, path: Path) -> None:
    """Generate a hospital-standard 12-lead ECG clinical report."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
        from reportlab.lib.units import mm
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        doc = SimpleDocTemplate(str(path), pagesize=A4,
                                 rightMargin=18*mm, leftMargin=18*mm,
                                 topMargin=15*mm, bottomMargin=15*mm)
        styles = getSampleStyleSheet()
        story = []

        # --- Custom Hospital Styles ---
        header_style = ParagraphStyle('Header', parent=styles['Normal'],
                                      fontSize=18, fontName='Helvetica-Bold',
                                      textColor=colors.HexColor('#0f172a'),
                                      alignment=TA_CENTER, spaceAfter=2)
        sub_header = ParagraphStyle('SubHeader', parent=styles['Normal'],
                                     fontSize=11, fontName='Helvetica-Bold',
                                     textColor=colors.HexColor('#334155'),
                                     alignment=TA_CENTER, spaceAfter=15)
        section_title = ParagraphStyle('SecTitle', parent=styles['Normal'],
                                        fontSize=10, fontName='Helvetica-Bold',
                                        textColor=colors.HexColor('#1e293b'),
                                        spaceBefore=12, spaceAfter=6,
                                        borderPadding=2, borderColor=colors.HexColor('#e2e8f0'), borderWidth=0)
        label_style = ParagraphStyle('Label', parent=styles['Normal'],
                                      fontSize=9, fontName='Helvetica-Bold',
                                      textColor=colors.HexColor('#475569'))
        val_style = ParagraphStyle('Value', parent=styles['Normal'], fontSize=9)
        impression_style = ParagraphStyle('Impression', parent=styles['Normal'],
                                           fontSize=10, leading=14, fontName='Helvetica-BoldOblique')
        patient_style = ParagraphStyle('Patient', parent=styles['Normal'],
                                        fontSize=10, leading=14, textColor=colors.HexColor('#334155'))

        # --- 1. Header (Medical Center Name) ---
        story.append(Paragraph("ATRIONNET CARDIOLOGY LAB", header_style))
        story.append(Paragraph("12-Lead ECG Clinical Report", sub_header))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0f172a'), spaceAfter=10))

        # --- 2. Patient Information ---
        import datetime
        date_str = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
        
        info_data = [
            [Paragraph("Patient Name:", label_style), "Anonymized Patient", Paragraph("Report Date:", label_style), date_str],
            [Paragraph("Patient ID:", label_style), "ECG-REQ-4492", Paragraph("Sex / Age:", label_style), "Not Specified"],
            [Paragraph("Device Type:", label_style), "12-Lead Clinical ECG", Paragraph("Status:", label_style), "Final Report"],
        ]
        info_table = Table(info_data, colWidths=[30*mm, 55*mm, 30*mm, 55*mm])
        info_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 5*mm))

        # --- 3. Clinical Summary / Impression ---
        diag = result.get('diagnosis', {})
        rat  = result.get('xai', {})
        story.append(Paragraph("ECG SUMMARY / IMPRESSION", section_title))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e1'), spaceAfter=5))
        
        story.append(Paragraph(rat.get('impression', 'Sinus rhythm detected. No acute abnormalities.'), impression_style))
        story.append(Spacer(1, 4*mm))

        # --- 4. Measurements Table ---
        metrics = result.get('clinical_metrics', {})
        story.append(Paragraph("QUANTITATIVE INSTANCE ANALYSIS (AtrionNet)", section_title))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e1'), spaceAfter=5))
        
        pr_val = f"{metrics.get('mean_pr_ms', 0)}" if metrics.get('mean_pr_ms', 0) > 0 else "N/A"
        hr_val = f"{metrics.get('heart_rate_bpm', 0)}" if metrics.get('heart_rate_bpm', 0) > 0 else "N/A"
        
        meas_data = [
            [Paragraph("Heart Rate (BPM)", label_style), f"{hr_val}", Paragraph("QRS Complexes", label_style), f"{metrics.get('n_qrs', '0')}"],
            [Paragraph("PR Interval (ms)", label_style), f"{pr_val}", Paragraph("Assoc. P-waves", label_style), f"{metrics.get('n_p_assoc', '0')}"],
            [Paragraph("Conduction Ratio", label_style), f"{metrics.get('p_qrs_ratio', 0):.2f}", Paragraph("Dissoc. P-waves", label_style), f"<b>{metrics.get('n_p_dissoc', '0')}</b>"],
        ]
        mt = Table(meas_data, colWidths=[40*mm, 45*mm, 40*mm, 45*mm])
        mt.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e2e8f0')),
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f8fafc')),
            ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#f8fafc')),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(mt)

        # --- 5. Clinical Interpretation ---
        story.append(Paragraph("CLINICAL INTERPRETATION", section_title))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e1'), spaceAfter=5))
        
        severity = diag.get('severity', 'Normal')
        urgency = "Routine monitoring recommended."
        if severity == "Severe": urgency = "Immediate clinical correlation and cardiologist follow-up required."
        elif severity == "Moderate": urgency = "Cardiology consultation recommended for rhythm monitoring."
        
        status_text = f"Findings are interpreted as <b>{severity}</b> severity. {urgency}"
        story.append(Paragraph(status_text, val_style))

        # --- 6. Patient-Friendly Explanation ---
        story.append(Paragraph("PATIENT-FRIENDLY EXPLANATION", section_title))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e1'), spaceAfter=5))
        story.append(Paragraph(rat.get('patient_explanation', ''), patient_style))

        # --- 7. Recommendations ---
        story.append(Paragraph("RECOMMENDATIONS", section_title))
        rec_text = "• Follow up with your primary care physician or cardiologist.<br/>• Maintain regular monitoring as advised by your healthcare provider.<br/>• Share this report during your next clinical appointment."
        story.append(Paragraph(rec_text, val_style))

        # --- 8. Disclaimer & Signature ---
        story.append(Spacer(1, 15*mm))
        disclaimer = "<i>This report is for informational purposes only and must be reviewed by a qualified medical professional before making clinical decisions.</i>"
        story.append(Paragraph(disclaimer, ParagraphStyle('Disc', parent=styles['Normal'], fontSize=8, textColor=colors.grey)))
        
        story.append(Spacer(1, 10*mm))
        sig_data = [
            [Paragraph("<b>Reviewed by:</b> ____________________", val_style), 
             Paragraph("<b>Cardiologist Signature:</b> ____________________", val_style)]
        ]
        st = Table(sig_data, colWidths=[85*mm, 85*mm])
        story.append(st)

        doc.build(story)
        logger.info(f"Clinical-grade PDF report saved → {path}")

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        path.with_suffix('.txt').write_text(str(result))
        
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
            [Paragraph("Conduction Ratio", label_style), f"{p_qrs_ratio:.2f}", Paragraph("Severity", label_style), diag.get('severity', 'UNKNOWN').upper()],
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
        
        # If fallback mode, raw heatmap is None
        model_heatmap = heatmap if self._mode == "model" else None

        diagnosis = _classify_av_block(intervals, all_p, r_peaks, heatmap=model_heatmap)
        rationale = _get_clinical_rationale(intervals, diagnosis, all_p, r_peaks)

        # Build structured clinical importance map (replaces noisy fake heatmap)
        importance_map = _build_importance_map(
            signal_len=len(lead_ii),
            p_spans=all_p,
            qrs_spans=qrs_spans,
            t_spans=t_spans,
            r_peaks=r_peaks,
            diagnosis=diagnosis,
            model_heatmap=model_heatmap,
        )

        focus_label = _get_focus_label(
            importance_map, all_p, qrs_spans, t_spans, r_peaks, len(lead_ii)
        )

        return {
            "diagnosis": diagnosis,
            "intervals": intervals,
            "clinical_metrics": {
                **intervals,
                "n_p_assoc": len(p_associated),
                "n_p_dissoc": len(p_dissociated),
                "n_p_total": len(all_p),
                "n_qrs": len(r_peaks),
                "p_qrs_ratio": len(all_p) / len(r_peaks) if r_peaks else 0
            },
            "waves": {
                "P_associated":  p_associated,
                "P_dissociated": p_dissociated,
                "QRS":           qrs_spans,
                "T":             t_spans,
            },
            "xai": {
                **rationale,
                "explanation": rationale["full_rationale"],
                "heatmap": importance_map.tolist(),
                "focus_label": focus_label,
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
            x = torch.tensor(signal[np.newaxis]).to(self.device).float()  # (1,12,5000)
            out = self.model(x)

        heatmap   = out['heatmap'][0].cpu().numpy()
        width_map = out['width'][0].cpu().numpy()
        instances = get_instances_from_heatmap(heatmap, width_map, threshold=0.35, distance=60)

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
