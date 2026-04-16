"""
PHASE 4 — Clinical XAI & Visual Logic
[Sup #7, #11]

Generates:
  1. High-fidelity P-wave overlay plots (model predictions vs expert annotations)
  2. Attention Heatmaps — Squeeze-and-Excitation channel weights per encoder block
  3. Model Logic vs Expert Annotation comparison panels (side-by-side)
  4. Lead-II signal with coloured confidence-shaded P-wave spans

All figures saved as 300 DPI PNGs to tests/outputs/attention_maps/

CLI:
  cd AtrionNet_Implementation
  python tests/xai_audit/visualize_logic.py [--weights path] [--n_records 5]

[CLINICAL-VAL] All logs prefixed with [CLINICAL-VAL].
[Sup #7, #11]
"""

from __future__ import annotations

# ── stdlib ───────────────────────────────────────────────────────────────────
import os
import sys
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")  # Headless rendering — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks

# ── path bootstrap ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT      = PROJECT_ROOT / "ml_component"
TESTS_OUT    = PROJECT_ROOT / "tests" / "outputs" / "attention_maps"
TESTS_OUT.mkdir(parents=True, exist_ok=True)

if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[CLINICAL-VAL] %(levelname)s — %(message)s"
)
log = logging.getLogger("phase4")

# ── config ───────────────────────────────────────────────────────────────────
CFG_PATH = PROJECT_ROOT / "tests" / "configs" / "clinical_config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

SEED      = CFG["seed"]
SR        = CFG["dataset"]["sampling_rate"]    # 500 Hz
SEQ_LEN   = CFG["dataset"]["target_length"]    # 5000
N_LEADS   = CFG["dataset"]["n_leads"]          # 12
CONF      = CFG["inference"]["conf_threshold"]  # 0.45
DIST      = CFG["inference"]["distance_samples"] # 60
PROM      = CFG["inference"]["prominence"]       # 0.10
DPI       = CFG["reporting"]["figure_dpi"]       # 300

# Clinical colour palette (colourblind-safe)
CLR_SIGNAL     = "#2C3E50"   # Dark slate — ECG signal
CLR_GT_SPAN    = "#27AE60"   # Green — expert annotation
CLR_PRED_SPAN  = "#E74C3C"   # Red — model prediction
CLR_HEATMAP    = "#3498DB"   # Blue — heatmap overlay
CLR_AGREE      = "#8E44AD"   # Purple — overlap (agree)
CLR_ATTN_MAP   = "plasma"    # Colormap for attention


# ════════════════════════════════════════════════════════════════════════════
# § 0 — Determinism
# ════════════════════════════════════════════════════════════════════════════
def _seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


_seed_all()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════════════════
# § 1 — Data Loader
# ════════════════════════════════════════════════════════════════════════════
def _synthetic_ecg(n: int = 10) -> Tuple[np.ndarray, List[Dict]]:
    rng = np.random.default_rng(SEED)
    signals = rng.standard_normal((n, N_LEADS, SEQ_LEN)).astype(np.float32)
    # Inject realistic P-wave bumps on Lead II (index 1)
    anns = []
    for i in range(n):
        n_p = rng.integers(6, 10)
        spacing = SEQ_LEN // (n_p + 1)
        pws = []
        for k in range(n_p):
            pk     = spacing * (k + 1)
            width  = rng.integers(18, 28)
            amp    = rng.uniform(0.3, 0.8)
            # Inject Gaussian bump
            t      = np.arange(SEQ_LEN)
            bump   = amp * np.exp(-((t - pk) ** 2) / (2 * (width / 2) ** 2))
            signals[i, 1] += bump.astype(np.float32)
            pws.append((max(0, pk - width), pk, min(SEQ_LEN - 1, pk + width)))
        anns.append({"p_waves": pws})
    return signals, anns


def _load_test_split(n_records: int = 5) -> Tuple[np.ndarray, List[Dict]]:
    data_dir = PROJECT_ROOT / CFG["paths"]["data_dir"]
    if data_dir.exists() and any(data_dir.rglob("*.hea")):
        try:
            from src.data_pipeline.ludb_loader import LUDBLoader
            loader = LUDBLoader(str(data_dir))
            sigs, anns = loader.get_all_data()
            N = len(sigs)
            np.random.seed(SEED)
            perm     = np.random.permutation(N)
            idx_test = perm[int(N * 0.85):]
            sigs_test = sigs[idx_test[:n_records]]
            anns_test = [anns[i] for i in idx_test[:n_records]]
            log.info(f"[CLINICAL-VAL] XAI: {len(sigs_test)} LUDB test records.")
            return sigs_test, anns_test
        except Exception as e:
            log.warning(f"LUDB failed ({e}). Using synthetic.")
    log.warning("[CLINICAL-VAL] XAI: synthetic ECG fallback.")
    sigs, anns = _synthetic_ecg(n_records)
    return sigs, anns


# ════════════════════════════════════════════════════════════════════════════
# § 2 — Attention Hook Registration
# ════════════════════════════════════════════════════════════════════════════
class AttentionCapturer:
    """
    Registers forward hooks on all AttentionBlock1D modules.
    Captures channel attention weights (SE-block outputs).
    [Sup #7, #11]
    """
    def __init__(self):
        self.weights: Dict[str, np.ndarray] = {}
        self._hooks  = []

    def register(self, model: torch.nn.Module) -> None:
        from src.modeling.atrion_net import AttentionBlock1D

        def _make_hook(name: str):
            def _hook(module, inp, out):
                # SE block output is x * y.expand_as(x) — shape [B, C, L]
                # Channel attention weights = mean over spatial dim
                attn = out.detach().cpu()
                # Average over batch and spatial → [C]
                self.weights[name] = attn.mean(dim=(0, 2)).numpy()
            return _hook

        idx = 0
        for name, module in model.named_modules():
            if isinstance(module, AttentionBlock1D):
                hook = module.register_forward_hook(_make_hook(f"attn_{idx}_{name}"))
                self._hooks.append(hook)
                idx += 1
        log.info(f"[CLINICAL-VAL] Registered {idx} attention hooks.")

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ════════════════════════════════════════════════════════════════════════════
# § 3 — Inference Utilities (SOC-safe inline)
# ════════════════════════════════════════════════════════════════════════════
def _decode_instances(hm_np: np.ndarray, wm_np: np.ndarray) -> List[Dict]:
    hm   = hm_np.squeeze()
    wmap = wm_np.squeeze()
    peaks, _ = find_peaks(hm, height=CONF, distance=DIST, prominence=PROM)
    instances = []
    for pk in peaks:
        w     = max(30, min(abs(float(wmap[pk])) * SEQ_LEN, 300))
        start = max(0, int(pk - w / 2))
        end   = min(SEQ_LEN, int(pk + w / 2))
        instances.append({
            "center":     int(pk),
            "span":       (start, end),
            "confidence": float(hm[pk])
        })
    return instances


def _time_axis(seq_len: int = SEQ_LEN, fs: int = SR) -> np.ndarray:
    return np.linspace(0, seq_len / fs, seq_len)


# ════════════════════════════════════════════════════════════════════════════
# § 4 — Figure Generators
# ════════════════════════════════════════════════════════════════════════════
def _plot_pwav_overlay(signal:      np.ndarray,
                       heatmap:     np.ndarray,
                       gt_p_waves:  List[Tuple],
                       pred_insts:  List[Dict],
                       rec_idx:     int,
                       out_dir:     Path) -> None:
    """
    Figure 1 — P-Wave Detection Overlay
    Top   : Lead II signal with GT spans (green) and Predicted spans (red)
    Bottom: Raw heatmap output with confidence threshold line
    [Sup #11]
    """
    t     = _time_axis()
    lead2 = signal[1] if signal.shape[0] > 1 else signal[0]
    hm    = heatmap.squeeze()

    fig, axes = plt.subplots(2, 1, figsize=(18, 6), sharex=True,
                              facecolor="#F8F9FA")
    fig.suptitle(
        f"[CLINICAL-VAL] P-Wave Detection Overlay — Record {rec_idx+1}",
        fontsize=13, fontweight="bold", color=CLR_SIGNAL
    )

    # ── Ax0: ECG signal ──────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.plot(t, lead2, color=CLR_SIGNAL, lw=0.7, label="Lead II (ECG)")
    ax0.set_facecolor("#FAFAFA")

    for (onset, peak, offset) in gt_p_waves:
        ax0.axvspan(onset / SR, offset / SR,
                    alpha=0.20, color=CLR_GT_SPAN, zorder=2)
        ax0.axvline(peak / SR, color=CLR_GT_SPAN, lw=0.8,
                    ls="--", alpha=0.6)

    for inst in pred_insts:
        s, e = inst["span"]
        ax0.axvspan(s / SR, e / SR,
                    alpha=0.18, color=CLR_PRED_SPAN, zorder=3)
        ax0.axvline(inst["center"] / SR, color=CLR_PRED_SPAN,
                    lw=1.0, ls="-", alpha=0.8)

    # Legend
    patches = [
        mpatches.Patch(color=CLR_GT_SPAN, alpha=0.5, label="Expert Annotation (GT)"),
        mpatches.Patch(color=CLR_PRED_SPAN, alpha=0.5, label="Model Prediction"),
    ]
    ax0.legend(handles=patches, loc="upper right", fontsize=8)
    ax0.set_ylabel("Amplitude (z-scored)", fontsize=9)
    ax0.set_ylim(lead2.min() - 0.5, lead2.max() + 0.5)
    ax0.grid(True, lw=0.3, alpha=0.5)

    # ── Ax1: Heatmap ─────────────────────────────────────────────────────
    ax1 = axes[1]
    ax1.fill_between(t, hm, alpha=0.4, color=CLR_HEATMAP, label="Detection Heatmap")
    ax1.plot(t, hm, color=CLR_HEATMAP, lw=0.8)
    ax1.axhline(CONF, color="orange", ls="--", lw=1.0,
                label=f"Threshold (conf={CONF})")
    ax1.set_xlabel("Time (seconds)", fontsize=9)
    ax1.set_ylabel("Heatmap Confidence", fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, lw=0.3, alpha=0.5)
    ax1.set_facecolor("#FAFAFA")

    plt.tight_layout()
    out_path = out_dir / f"record_{rec_idx+1:03d}_pwave_overlay.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[CLINICAL-VAL] Saved P-wave overlay → {out_path.name}")


def _plot_attention_heatmap(attn_weights: Dict[str, np.ndarray],
                            rec_idx:      int,
                            out_dir:      Path) -> None:
    """
    Figure 2 — SE-Block Channel Attention Heatmap
    Visualises which channels (feature maps) are being attended to in each
    AttentionBlock1D layer. [Sup #7]
    """
    if not attn_weights:
        log.warning("[CLINICAL-VAL] No attention weights captured — skipping heatmap.")
        return

    block_names = list(attn_weights.keys())
    n_blocks    = len(block_names)
    max_ch      = max(len(v) for v in attn_weights.values())

    # Pad each block to max_ch for uniform matrix
    heat_matrix = np.zeros((n_blocks, max_ch))
    for i, name in enumerate(block_names):
        w = attn_weights[name]
        heat_matrix[i, :len(w)] = w

    # Normalise per block
    row_max = heat_matrix.max(axis=1, keepdims=True) + 1e-9
    heat_matrix = heat_matrix / row_max

    fig, ax = plt.subplots(figsize=(max(10, max_ch // 8), max(4, n_blocks)),
                            facecolor="#F8F9FA")
    fig.suptitle(
        f"[CLINICAL-VAL] SE-Block Channel Attention Weights — Record {rec_idx+1}",
        fontsize=12, fontweight="bold", color=CLR_SIGNAL
    )

    im = ax.imshow(heat_matrix, aspect="auto", cmap=CLR_ATTN_MAP,
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(n_blocks))
    ax.set_yticklabels(
        [f"Block {i+1} ({b.split('.')[-1]})" for i, b in enumerate(block_names)],
        fontsize=7
    )
    ax.set_xlabel("Channel Index", fontsize=9)
    ax.set_ylabel("AttentionBlock1D Layer", fontsize=9)
    plt.colorbar(im, ax=ax, label="Normalised Attention Weight", fraction=0.02, pad=0.04)
    plt.tight_layout()

    out_path = out_dir / f"record_{rec_idx+1:03d}_attention_heatmap.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[CLINICAL-VAL] Saved attention heatmap → {out_path.name}")


def _plot_model_vs_expert(signal:     np.ndarray,
                          heatmap:    np.ndarray,
                          gt_p_waves: List[Tuple],
                          pred_insts: List[Dict],
                          rec_idx:    int,
                          out_dir:    Path) -> None:
    """
    Figure 3 — Model Logic vs Expert Annotation Panel
    3-row layout:
      Row 1: Expert annotations on Lead II
      Row 2: Model predictions on Lead II
      Row 3: Agreement map — overlapping spans highlighted
    [Sup #7, #11]
    """
    t     = _time_axis()
    lead2 = signal[1] if signal.shape[0] > 1 else signal[0]

    fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=True,
                              facecolor="#F8F9FA")
    fig.suptitle(
        f"[CLINICAL-VAL] Model Logic vs Expert Annotation — Record {rec_idx+1}",
        fontsize=13, fontweight="bold", color=CLR_SIGNAL
    )

    # Build binary occupancy masks
    gt_mask   = np.zeros(SEQ_LEN, dtype=np.float32)
    pred_mask = np.zeros(SEQ_LEN, dtype=np.float32)
    for (o, p, f) in gt_p_waves:
        gt_mask[o:f] = 1.0
    for inst in pred_insts:
        s, e = inst["span"]
        pred_mask[s:e] = 1.0
    agree_mask = gt_mask * pred_mask    # intersection
    gt_only    = gt_mask * (1 - pred_mask)
    pred_only  = pred_mask * (1 - gt_mask)

    def _decorate(ax, title):
        ax.set_facecolor("#FAFAFA")
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=4)
        ax.grid(True, lw=0.3, alpha=0.4)

    # ── Row 0: Expert ─────────────────────────────────────────────────────
    axes[0].plot(t, lead2, color=CLR_SIGNAL, lw=0.7)
    axes[0].fill_between(t, lead2.min(), lead2.max(),
                          where=gt_mask.astype(bool),
                          alpha=0.25, color=CLR_GT_SPAN, interpolate=True)
    _decorate(axes[0], "Expert Annotation (Ground Truth)")

    # ── Row 1: Model ──────────────────────────────────────────────────────
    axes[1].plot(t, lead2, color=CLR_SIGNAL, lw=0.7)
    axes[1].fill_between(t, lead2.min(), lead2.max(),
                          where=pred_mask.astype(bool),
                          alpha=0.25, color=CLR_PRED_SPAN, interpolate=True)
    _decorate(axes[1], "Model Prediction (AtrionNet)")

    # ── Row 2: Agreement ──────────────────────────────────────────────────
    axes[2].plot(t, lead2, color=CLR_SIGNAL, lw=0.7)
    axes[2].fill_between(t, lead2.min(), lead2.max(),
                          where=agree_mask.astype(bool),
                          alpha=0.35, color=CLR_AGREE,
                          label="Agreed (TP)", interpolate=True)
    axes[2].fill_between(t, lead2.min(), lead2.max(),
                          where=gt_only.astype(bool),
                          alpha=0.25, color=CLR_GT_SPAN,
                          label="GT only (FN)", interpolate=True)
    axes[2].fill_between(t, lead2.min(), lead2.max(),
                          where=pred_only.astype(bool),
                          alpha=0.25, color=CLR_PRED_SPAN,
                          label="Pred only (FP)", interpolate=True)
    _decorate(axes[2], "Agreement Map (TP / FN / FP)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_xlabel("Time (seconds)", fontsize=9)

    plt.tight_layout()
    out_path = out_dir / f"record_{rec_idx+1:03d}_model_vs_expert.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[CLINICAL-VAL] Saved model_vs_expert → {out_path.name}")


def _plot_confidence_spectrum(pred_insts: List[List[Dict]],
                              n_records:  int,
                              out_dir:    Path) -> None:
    """
    Figure 4 — Confidence Distribution across all test records.
    Histograms of predicted confidence scores split by TP/FP.
    """
    all_conf = [inst["confidence"] for insts in pred_insts for inst in insts]
    if not all_conf:
        log.warning("[CLINICAL-VAL] No predictions — skipping confidence spectrum.")
        return

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#F8F9FA")
    fig.suptitle("[CLINICAL-VAL] Prediction Confidence Score Distribution",
                 fontsize=12, fontweight="bold", color=CLR_SIGNAL)
    ax.hist(all_conf, bins=30, color=CLR_HEATMAP, alpha=0.75, edgecolor="white", lw=0.5)
    ax.axvline(CONF, color="orange", ls="--", lw=1.5, label=f"Threshold = {CONF}")
    ax.set_xlabel("Heatmap Confidence Score", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_facecolor("#FAFAFA")
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    plt.tight_layout()
    out_path = out_dir / "confidence_distribution.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[CLINICAL-VAL] Saved confidence distribution → {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# § 5 — Main
# ════════════════════════════════════════════════════════════════════════════
def run_xai_audit(weights_path: Optional[str] = None,
                  n_records: int = 5) -> None:

    print("\n" + "=" * 72)
    print("[CLINICAL-VAL] PHASE 4 — Clinical XAI & Visual Logic [Sup #7, #11]")
    print(f"  Records     : {n_records}")
    print(f"  Output dir  : {TESTS_OUT}")
    print("=" * 72 + "\n")

    # ── Load data ──────────────────────────────────────────────────────────
    test_signals, test_anns = _load_test_split(n_records)

    # ── Resolve weights ────────────────────────────────────────────────────
    if weights_path is None:
        for cand in [CFG["paths"]["weights_best"], CFG["paths"]["weights_alt"]]:
            full = PROJECT_ROOT / cand
            if full.exists():
                weights_path = str(full)
                break

    # ── Load model ─────────────────────────────────────────────────────────
    from src.modeling.atrion_net import AtrionNetHybrid
    model = AtrionNetHybrid(in_channels=N_LEADS).to(DEVICE)
    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        log.info(f"[CLINICAL-VAL] Weights: {weights_path}")
    else:
        log.warning("[CLINICAL-VAL] Weights not found — random init.")
    model.eval()

    # ── Register attention hooks ───────────────────────────────────────────
    capturer = AttentionCapturer()
    capturer.register(model)

    # ── Per-record visualisation ───────────────────────────────────────────
    all_pred_insts = []
    for rec_idx, (sig, ann) in enumerate(zip(test_signals, test_anns)):
        sig_t = torch.tensor(sig[np.newaxis], dtype=torch.float32).to(DEVICE)

        capturer.weights.clear()
        with torch.no_grad():
            out = model(sig_t)

        hm_np = out["heatmap"][0].cpu().numpy()
        wm_np = out["width"][0].cpu().numpy()
        pred_insts = _decode_instances(hm_np, wm_np)
        all_pred_insts.append(pred_insts)

        log.info(f"[CLINICAL-VAL] Record {rec_idx+1}: "
                 f"GT={len(ann['p_waves'])} P-waves | "
                 f"Pred={len(pred_insts)} detections | "
                 f"Attention blocks captured={len(capturer.weights)}")

        # Figure 1 — Overlay
        _plot_pwav_overlay(sig, hm_np, ann["p_waves"], pred_insts,
                           rec_idx, TESTS_OUT)
        # Figure 2 — Attention heatmap
        _plot_attention_heatmap(dict(capturer.weights), rec_idx, TESTS_OUT)
        # Figure 3 — Model vs Expert panel
        _plot_model_vs_expert(sig, hm_np, ann["p_waves"], pred_insts,
                              rec_idx, TESTS_OUT)

    capturer.remove()

    # Figure 4 — Global confidence distribution
    _plot_confidence_spectrum(all_pred_insts, n_records=len(test_signals),
                              out_dir=TESTS_OUT)

    print(f"\n[CLINICAL-VAL] Phase 4 complete.")
    print(f"  Figures → {TESTS_OUT}  ({3 * len(test_signals) + 1} PNGs)")
    print("=" * 72 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# § 6 — CLI
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[CLINICAL-VAL] Phase 4 — Clinical XAI & Visual Logic [Sup #7, #11]"
    )
    parser.add_argument("--weights",   type=str, default=None,
                        help="Path to AtrionNet weights (.pth)")
    parser.add_argument("--n_records", type=int, default=5,
                        help="Number of test records to visualise (default: 5)")
    args = parser.parse_args()
    run_xai_audit(weights_path=args.weights, n_records=args.n_records)
