"""
PHASE 2 — SOTA Competitive Benchmarking
[Sup #3, #6]

Compares ATRIONNET against three baselines on the held-out test split:
  1. Rule-Based  — Pan-Tompkins Variant (P-wave post-processing on QRS detections)
  2. 1D U-Net    — AtrionNetBaseline trained with IDENTICAL pipeline (train_baselines.py)
  3. CNN-LSTM    — CNN + BiLSTM trained with IDENTICAL pipeline (train_baselines.py)

Fair comparison: ALL deep learning baselines use pre-trained weights from
  tests/outputs/baseline_weights/unet_baseline_best.pth
  tests/outputs/baseline_weights/cnn_lstm_best.pth

Run train_baselines.py FIRST:
  python tests/benchmark/train_baselines.py

Outputs:
  tests/outputs/benchmark_results.csv   — metrics + MACs + params + latency
  tests/outputs/benchmark_summary.txt   — human-readable summary

CLI:
  cd AtrionNet_Implementation
  python tests/benchmark/compare_baselines.py [--weights path/to/best.pth]

[CLINICAL-VAL] All metric logs prefixed with [CLINICAL-VAL].
[Sup #3, #6]
"""

from __future__ import annotations

# ── stdlib ───────────────────────────────────────────────────────────────────
import os
import sys
import csv
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import yaml

# ── path bootstrap ───────────────────────────────────────────────────────────
PROJECT_ROOT          = Path(__file__).resolve().parents[2]
ML_ROOT      = PROJECT_ROOT
TESTS_OUT             = PROJECT_ROOT / "tests" / "outputs"
BASELINE_WEIGHTS_DIR  = TESTS_OUT / "baseline_weights"
TESTS_OUT.mkdir(parents=True, exist_ok=True)
BASELINE_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[CLINICAL-VAL] %(levelname)s — %(message)s"
)
log = logging.getLogger("phase2")

# ── config ───────────────────────────────────────────────────────────────────
CONFIG_PATH = PROJECT_ROOT / "tests" / "configs" / "clinical_config.yaml"
with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

SEED     = CFG["seed"]
SR       = CFG["dataset"]["sampling_rate"]        # 500 Hz
SEQ_LEN  = CFG["dataset"]["target_length"]        # 5000
N_LEADS  = CFG["dataset"]["n_leads"]              # 12
CONF     = CFG["inference"]["conf_threshold"]      # 0.45
DIST     = CFG["inference"]["distance_samples"]    # 60
PROM     = CFG["inference"]["prominence"]          # 0.10
TIGHT    = CFG["aami_ec57"]["tolerance_tight_smp"] # 25 samples
LOOSE    = CFG["aami_ec57"]["tolerance_loose_smp"] # 50 samples


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
log.info(f"[CLINICAL-VAL] Device: {DEVICE} | Seed: {SEED}")


# ════════════════════════════════════════════════════════════════════════════
# § 1 — Synthetic / LUDB Data Loader
# ════════════════════════════════════════════════════════════════════════════
def _synthetic_ecg(n: int = 30) -> Tuple[np.ndarray, List[Dict]]:
    rng = np.random.default_rng(SEED)
    signals = rng.standard_normal((n, N_LEADS, SEQ_LEN)).astype(np.float32)
    anns    = []
    for _ in range(n):
        n_p  = rng.integers(8, 13)
        spacing = SEQ_LEN // (n_p + 1)
        pws = []
        for k in range(n_p):
            pk = spacing * (k + 1)
            pws.append((max(0, pk - 20), pk, min(SEQ_LEN - 1, pk + 20)))
        anns.append({"p_waves": pws})
    return signals, anns


def _load_test_split() -> Tuple[np.ndarray, List[Dict]]:
    """Reproduce 70/15/15 split and return ONLY the test partition."""
    data_dir = PROJECT_ROOT / CFG["paths"]["data_dir"]
    if data_dir.exists() and any(data_dir.rglob("*.hea")):
        try:
            from src.data_pipeline.ludb_loader import LUDBLoader
            loader = LUDBLoader(str(data_dir))
            signals, anns = loader.get_all_data()
            log.info(f"[CLINICAL-VAL] LUDB loaded — {len(signals)} records total.")
        except Exception as e:
            log.warning(f"[CLINICAL-VAL] LUDB load failed ({e}). Using synthetic.")
            signals, anns = _synthetic_ecg(30)
    else:
        log.warning("[CLINICAL-VAL] LUDB not found. Using 30-record synthetic fallback.")
        signals, anns = _synthetic_ecg(30)

    N = len(signals)
    np.random.seed(SEED)
    perm     = np.random.permutation(N)
    val_end  = int(N * 0.85)
    idx_test = perm[val_end:]

    test_signals = signals[idx_test]
    test_anns    = [anns[i] for i in idx_test]
    log.info(f"[CLINICAL-VAL] Test partition: {len(test_signals)} records.")
    return test_signals, test_anns


# ════════════════════════════════════════════════════════════════════════════
# § 2 — Baseline Model Definitions
# ════════════════════════════════════════════════════════════════════════════

# ── 2a. Rule-Based Pan-Tompkins Variant ─────────────────────────────────────
class PanTompkinsVariant:
    """
    Rule-based P-wave detector.
    Strategy: detect QRS complexes via derivative + threshold, then search
    PR interval (120–200ms before QRS) for local maxima as P-wave candidates.
    [Sup #3]
    """
    name = "Rule-Based (Pan-Tompkins Variant)"

    def __init__(self, fs: int = SR):
        self.fs = fs
        self.pr_min = int(0.12 * fs)   # 120ms
        self.pr_max = int(0.20 * fs)   # 200ms
        self.p_half = int(0.04 * fs)   # ±40ms half-width

    def _detect_qrs(self, lead: np.ndarray) -> np.ndarray:
        """Simple derivative-based QRS detector on a single lead."""
        deriv  = np.diff(lead, prepend=lead[0])
        sq     = deriv ** 2
        # Moving-window integration (150ms)
        window = int(0.15 * self.fs)
        kernel = np.ones(window) / window
        mwi    = np.convolve(sq, kernel, mode="same")
        thresh = 0.35 * mwi.max()
        qrs_idx = []
        above   = mwi > thresh
        in_qrs  = False
        start   = 0
        for i, v in enumerate(above):
            if v and not in_qrs:
                in_qrs = True
                start  = i
            elif not v and in_qrs:
                in_qrs = False
                peak   = start + np.argmax(mwi[start:i])
                qrs_idx.append(int(peak))
        return np.array(qrs_idx)

    def predict(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Returns heatmap-like peak list and per-record instances.
        signal: [12, 5000]
        """
        lead_ii = signal[1] if signal.shape[0] > 1 else signal[0]
        qrs_locs = self._detect_qrs(lead_ii)

        instances = []
        for qrs in qrs_locs:
            search_start = max(0, qrs - self.pr_max)
            search_end   = max(0, qrs - self.pr_min)
            if search_end <= search_start:
                continue
            segment = lead_ii[search_start:search_end]
            if len(segment) == 0:
                continue
            local_peak = search_start + int(np.argmax(np.abs(segment)))
            onset  = max(0, local_peak - self.p_half)
            offset = min(SEQ_LEN, local_peak + self.p_half)
            instances.append({
                "center":     local_peak,
                "span":       (onset, offset),
                "confidence": float(np.abs(lead_ii[local_peak]))
            })
        return {"instances": instances}


# ── 2b. CNN-LSTM Baseline ────────────────────────────────────────────────────
class CNNLSTM(nn.Module):
    """
    Lightweight CNN encoder + bidirectional LSTM decoder.
    Produces heatmap, width, and mask heads identical to AtrionNet.
    [Sup #3, #6]
    """
    name = "CNN-LSTM"

    def __init__(self, in_channels: int = N_LEADS, hidden: int = 128):
        super().__init__()
        # ── Encoder ──────────────────────────────────────
        self.enc = nn.Sequential(
            nn.Conv1d(in_channels, 64,  kernel_size=7, padding=3), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.MaxPool1d(2),   # → 2500
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),   # → 1250
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(2),   # → 625
        )
        # ── Temporal Context ─────────────────────────────
        self.lstm = nn.LSTM(256, hidden, batch_first=True,
                            bidirectional=True, num_layers=1)
        # ── Decoder ──────────────────────────────────────
        self.up   = nn.Sequential(
            nn.ConvTranspose1d(hidden * 2, 128, kernel_size=2, stride=2),  # → 1250
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),           # → 2500
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2),            # → 5000
            nn.ReLU(),
        )
        # ── Output Heads ─────────────────────────────────
        self.heatmap_head = nn.Sequential(nn.Conv1d(32, 1, 1), nn.Sigmoid())
        self.width_head   = nn.Conv1d(32, 1, 1)
        self.mask_head    = nn.Conv1d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.enc(x)                            # [B, 256, 625]
        feat_t = feat.permute(0, 2, 1)               # [B, 625, 256]
        out_t, _ = self.lstm(feat_t)                  # [B, 625, 256]
        out_c = out_t.permute(0, 2, 1)               # [B, 256, 625]
        dec   = self.up(out_c)                        # [B, 32, 5000]
        return {
            "heatmap": self.heatmap_head(dec),
            "width":   self.width_head(dec),
            "mask":    self.mask_head(dec)
        }


# ════════════════════════════════════════════════════════════════════════════
# § 3 — Shared Inference Utilities
# ════════════════════════════════════════════════════════════════════════════
def _heatmap_to_instances(heatmap_np: np.ndarray,
                          width_np:   np.ndarray) -> List[Dict]:
    """Peak-picking inference — mirrors atrion_evaluator logic (SOC-safe copy)."""
    from scipy.signal import find_peaks
    hm   = heatmap_np.squeeze()
    wmap = width_np.squeeze()
    peaks, _ = find_peaks(hm, height=CONF, distance=DIST, prominence=PROM)
    instances = []
    for pk in peaks:
        w      = abs(float(wmap[pk])) * 5000
        w      = max(30, min(w, 300))
        start  = max(0, int(pk - w / 2))
        end    = min(SEQ_LEN, int(pk + w / 2))
        instances.append({
            "center":     int(pk),
            "span":       (start, end),
            "confidence": float(hm[pk])
        })
    return instances


def _compute_aami_record(pred_instances: List[Dict],
                         gt_p_waves:    List[Tuple],
                         tolerance:     int) -> Dict[str, float]:
    """AAMI EC57 per-record metrics (SOC-safe, inline)."""
    gt_spans = [(o, f) for o, p, f in gt_p_waves]
    pred_peaks = [inst["center"] for inst in pred_instances]
    matched = set()
    tp = 0
    for pk in pred_peaks:
        for idx, (onset, offset) in enumerate(gt_spans):
            if idx in matched:
                continue
            gc = (onset + offset) // 2
            if abs(pk - gc) <= tolerance or (onset - tolerance) <= pk <= (offset + tolerance):
                tp += 1
                matched.add(idx)
                break
    fp = max(0, len(pred_peaks) - tp)
    fn = max(0, len(gt_spans) - tp)
    se  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pp  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    jac = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"Se": se, "+P": pp, "FPR": fpr, "Jaccard": jac,
            "TP": tp, "FP": fp, "FN": fn}


# ════════════════════════════════════════════════════════════════════════════
# § 4 — MACs & Parameter Counter
# ════════════════════════════════════════════════════════════════════════════
def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _estimate_macs(model: nn.Module,
                   input_shape: Tuple = (1, N_LEADS, SEQ_LEN)) -> int:
    """
    Lightweight MAC estimator using torch hooks.
    Falls back to parameter-based heuristic if hooks fail.
    """
    total_macs = [0]

    def _conv1d_hook(module, inp, out):
        batch, c_in, l_in = inp[0].shape
        _, c_out, l_out   = out.shape
        k  = module.kernel_size[0]
        g  = module.groups
        macs = batch * c_out * l_out * (c_in // g) * k
        total_macs[0] += macs

    def _linear_hook(module, inp, out):
        batch = inp[0].shape[0]
        total_macs[0] += batch * module.in_features * module.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(_conv1d_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))

    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(*input_shape).to(next(model.parameters()).device)
        try:
            model(dummy)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    return total_macs[0]


# ════════════════════════════════════════════════════════════════════════════
# § 5 — Latency Benchmark
# ════════════════════════════════════════════════════════════════════════════
def _measure_latency_ms(model_or_fn,
                        signal: np.ndarray,
                        n_runs: int = 30,
                        is_nn: bool = True) -> float:
    """Returns mean inference latency in milliseconds over n_runs."""
    sig_t = torch.tensor(signal[np.newaxis]).to(DEVICE)
    # Warm-up
    for _ in range(5):
        if is_nn:
            with torch.no_grad():
                model_or_fn(sig_t)
        else:
            model_or_fn.predict(signal)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        if is_nn:
            with torch.no_grad():
                model_or_fn(sig_t)
        else:
            model_or_fn.predict(signal)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


# ════════════════════════════════════════════════════════════════════════════
# § 6 — Unified Inference Runner
# ════════════════════════════════════════════════════════════════════════════
def _run_nn_model(model: nn.Module,
                  test_signals: np.ndarray,
                  test_anns:    List[Dict],
                  tolerance:    int) -> Dict[str, float]:
    """
    Run a PyTorch nn.Module model over all test records.
    Returns aggregated AAMI metrics.
    """
    model.eval()
    model.to(DEVICE)
    all_se, all_pp, all_fpr, all_jac = [], [], [], []
    total_tp = total_fp = total_fn = 0

    for sig, ann in zip(test_signals, test_anns):
        sig_t = torch.tensor(sig[np.newaxis], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            out = model(sig_t)

        if isinstance(out, dict):
            hm = out["heatmap"][0].cpu().numpy()
            wm = out["width"][0].cpu().numpy()
            instances = _heatmap_to_instances(hm, wm)
        else:
            raise ValueError("Unexpected model output format.")

        m = _compute_aami_record(instances, ann["p_waves"], tolerance)
        all_se.append(m["Se"]); all_pp.append(m["+P"])
        all_fpr.append(m["FPR"]); all_jac.append(m["Jaccard"])
        total_tp += m["TP"]; total_fp += m["FP"]; total_fn += m["FN"]

    # Micro-averaged
    micro_se  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_pp  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_fpr = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_jac = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
    return {
        "Se_micro": micro_se, "+P_micro": micro_pp,
        "FPR_micro": micro_fpr, "Jaccard_micro": micro_jac,
        "Se_mean":  float(np.mean(all_se)),
        "+P_mean":  float(np.mean(all_pp)),
        "TP": total_tp, "FP": total_fp, "FN": total_fn
    }


def _run_rule_model(model: PanTompkinsVariant,
                    test_signals: np.ndarray,
                    test_anns:    List[Dict],
                    tolerance:    int) -> Dict[str, float]:
    """Run Pan-Tompkins over all test records."""
    all_se, all_pp = [], []
    total_tp = total_fp = total_fn = 0

    for sig, ann in zip(test_signals, test_anns):
        out = model.predict(sig)
        m   = _compute_aami_record(out["instances"], ann["p_waves"], tolerance)
        all_se.append(m["Se"]); all_pp.append(m["+P"])
        total_tp += m["TP"]; total_fp += m["FP"]; total_fn += m["FN"]

    micro_se  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_pp  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_fpr = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_jac = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
    return {
        "Se_micro": micro_se, "+P_micro": micro_pp,
        "FPR_micro": micro_fpr, "Jaccard_micro": micro_jac,
        "Se_mean":  float(np.mean(all_se)),
        "+P_mean":  float(np.mean(all_pp)),
        "TP": total_tp, "FP": total_fp, "FN": total_fn
    }


# ════════════════════════════════════════════════════════════════════════════
# § 7 — Main Benchmark Orchestrator
# ════════════════════════════════════════════════════════════════════════════
def _load_atrionnet(weights_path: str) -> nn.Module:
    """Load AtrionNetHybrid from checkpoint."""
    from src.modeling.atrion_net import AtrionNetHybrid
    model = AtrionNetHybrid(in_channels=N_LEADS)
    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        log.info(f"[CLINICAL-VAL] AtrionNet weights loaded from {weights_path}")
    else:
        log.warning("[CLINICAL-VAL] AtrionNet weights not found — using random init.")
    return model


def _load_unet_baseline() -> nn.Module:
    """
    Load AtrionNetBaseline (1D U-Net) with TRAINED weights from train_baselines.py.
    Falls back to random init with a clear warning if weights are missing.
    [Sup #3, #6] — Fair apples-to-apples comparison.
    """
    from src.modeling.atrion_net import AtrionNetBaseline
    model = AtrionNetBaseline(in_channels=N_LEADS)

    trained_path = BASELINE_WEIGHTS_DIR / "unet_baseline_best.pth"
    if trained_path.exists():
        state = torch.load(trained_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        log.info(f"[CLINICAL-VAL] ✅ U-Net Baseline: loaded trained weights → {trained_path.name}")
    else:
        log.warning(
            "[CLINICAL-VAL] ⚠️  U-Net trained weights NOT found at:\n"
            f"     {trained_path}\n"
            "  Run: python tests/benchmark/train_baselines.py --model unet\n"
            "  Falling back to RANDOM INIT — comparison will be UNFAIR."
        )
    return model


def _load_cnn_lstm() -> nn.Module:
    """
    Load CNNLSTM with TRAINED weights from train_baselines.py.
    Falls back to random init with a clear warning if weights are missing.
    [Sup #3, #6] — Fair apples-to-apples comparison.
    """
    model = CNNLSTM(in_channels=N_LEADS)
    trained_path = BASELINE_WEIGHTS_DIR / "cnn_lstm_best.pth"
    if trained_path.exists():
        state = torch.load(trained_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        log.info(f"[CLINICAL-VAL] ✅ CNN-LSTM: loaded trained weights → {trained_path.name}")
    else:
        log.warning(
            "[CLINICAL-VAL] ⚠️  CNN-LSTM trained weights NOT found at:\n"
            f"     {trained_path}\n"
            "  Run: python tests/benchmark/train_baselines.py --model cnn_lstm\n"
            "  Falling back to RANDOM INIT — comparison will be UNFAIR."
        )
    return model


def run_benchmark(weights_atrionnet: str = None,
                  tolerance_label:   str = "±50ms",
                  tolerance_smp:     int = TIGHT) -> None:

    print("\n" + "=" * 72)
    print("[CLINICAL-VAL] PHASE 2 — SOTA Competitive Benchmarking [Sup #3, #6]")
    print(f"  Tolerance       : {tolerance_label} ({tolerance_smp} samples @ {SR}Hz)")
    print(f"  AAMI EC57 Mode  : Sensitivity (Se), +P, FPR, Temporal Jaccard")
    print("=" * 72 + "\n")

    # ── Load test data ──────────────────────────────────────────────────────
    test_signals, test_anns = _load_test_split()
    sample_sig = test_signals[0]

    # ── Resolve weight paths ────────────────────────────────────────────────
    if weights_atrionnet is None:
        for candidate in [
            CFG["paths"]["weights_best"],
            CFG["paths"]["weights_alt"],
        ]:
            full = PROJECT_ROOT / candidate
            if full.exists():
                weights_atrionnet = str(full)
                break

    # ── Fair comparison check ──────────────────────────────────────────────
    unet_wts     = BASELINE_WEIGHTS_DIR / "unet_baseline_best.pth"
    cnn_lstm_wts = BASELINE_WEIGHTS_DIR / "cnn_lstm_best.pth"
    if not unet_wts.exists() or not cnn_lstm_wts.exists():
        print(
            "\n" + "=" * 72 + "\n"
            "[CLINICAL-VAL] ⚠️  WARNING — UNFAIR COMPARISON DETECTED\n"
            "  Baseline weights not found. Run this FIRST for fair results:\n"
            "  python tests/benchmark/train_baselines.py\n"
            "  Continuing with random init (results will be biased).\n"
            + "=" * 72 + "\n"
        )

    # ── Instantiate models ──────────────────────────────────────────────────
    atrionnet   = _load_atrionnet(weights_atrionnet).to(DEVICE)
    unet        = _load_unet_baseline().to(DEVICE)          # trained weights
    cnn_lstm    = _load_cnn_lstm().to(DEVICE)               # trained weights
    rule_model  = PanTompkinsVariant(fs=SR)

    models_nn = [
        ("AtrionNet (Ours)", atrionnet),
        ("1D U-Net Baseline", unet),
        ("CNN-LSTM", cnn_lstm),
    ]

    # ── Run inference & metrics ─────────────────────────────────────────────
    results = []

    for name, mdl in models_nn:
        log.info(f"[CLINICAL-VAL] Running inference: {name} ...")
        metrics   = _run_nn_model(mdl, test_signals, test_anns, tolerance_smp)
        latency   = _measure_latency_ms(mdl, sample_sig, n_runs=20, is_nn=True)
        n_params  = _count_params(mdl)
        macs      = _estimate_macs(mdl)

        row = {
            "Model":        name,
            "Se (micro)":   f"{metrics['Se_micro']:.4f}",
            "+P (micro)":   f"{metrics['+P_micro']:.4f}",
            "FPR (micro)":  f"{metrics['FPR_micro']:.4f}",
            "Jaccard":      f"{metrics['Jaccard_micro']:.4f}",
            "TP":           metrics["TP"],
            "FP":           metrics["FP"],
            "FN":           metrics["FN"],
            "Params (M)":   f"{n_params / 1e6:.3f}",
            "MACs (M)":     f"{macs / 1e6:.1f}",
            "Latency (ms)": f"{latency:.2f}",
            "Tolerance":    tolerance_label,
        }
        results.append(row)
        print(f"  [{name}]")
        print(f"    Se={metrics['Se_micro']:.4f}  +P={metrics['+P_micro']:.4f}  "
              f"FPR={metrics['FPR_micro']:.4f}  Jaccard={metrics['Jaccard_micro']:.4f}")
        print(f"    Params={n_params/1e6:.3f}M  MACs={macs/1e6:.1f}M  "
              f"Latency={latency:.2f}ms\n")

    # ── Rule-Based ──────────────────────────────────────────────────────────
    log.info("[CLINICAL-VAL] Running inference: Rule-Based (Pan-Tompkins) ...")
    r_metrics = _run_rule_model(rule_model, test_signals, test_anns, tolerance_smp)
    r_latency = _measure_latency_ms(rule_model, sample_sig, n_runs=20, is_nn=False)
    results.append({
        "Model":        "Rule-Based (Pan-Tompkins)",
        "Se (micro)":   f"{r_metrics['Se_micro']:.4f}",
        "+P (micro)":   f"{r_metrics['+P_micro']:.4f}",
        "FPR (micro)":  f"{r_metrics['FPR_micro']:.4f}",
        "Jaccard":      f"{r_metrics['Jaccard_micro']:.4f}",
        "TP":           r_metrics["TP"],
        "FP":           r_metrics["FP"],
        "FN":           r_metrics["FN"],
        "Params (M)":   "N/A",
        "MACs (M)":     "N/A",
        "Latency (ms)": f"{r_latency:.2f}",
        "Tolerance":    tolerance_label,
    })
    print(f"  [Rule-Based (Pan-Tompkins)]")
    print(f"    Se={r_metrics['Se_micro']:.4f}  +P={r_metrics['+P_micro']:.4f}  "
          f"FPR={r_metrics['FPR_micro']:.4f}  Jaccard={r_metrics['Jaccard_micro']:.4f}")
    print(f"    Latency={r_latency:.2f}ms\n")

    # ── Save CSV ────────────────────────────────────────────────────────────
    csv_path = TESTS_OUT / "benchmark_results.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    log.info(f"[CLINICAL-VAL] Benchmark CSV saved → {csv_path}")

    # ── Save human-readable summary ─────────────────────────────────────────
    _write_summary(results, TESTS_OUT / "benchmark_summary.txt", tolerance_label)

    print("=" * 72)
    print(f"[CLINICAL-VAL] Outputs saved to: {TESTS_OUT}")
    print("=" * 72)


def _write_summary(results: List[Dict],
                   out_path: Path,
                   tolerance_label: str) -> None:
    lines = [
        "=" * 72,
        "ATRIONNET BENCHMARK SUMMARY — PHASE 2 [Sup #3, #6]",
        f"AAMI EC57 Tolerance: {tolerance_label}",
        "Metrics: Sensitivity (Se), Positive Predictivity (+P), FPR, Jaccard",
        "=" * 72,
        ""
    ]
    header = f"{'Model':<32} {'Se':>8} {'+P':>8} {'FPR':>8} {'Jaccard':>8} {'Params(M)':>10} {'MACs(M)':>9} {'Lat(ms)':>9}"
    lines.append(header)
    lines.append("-" * 72)
    for r in results:
        lines.append(
            f"{r['Model']:<32} {r['Se (micro)']:>8} {r['+P (micro)']:>8} "
            f"{r['FPR (micro)']:>8} {r['Jaccard']:>8} {r['Params (M)']:>10} "
            f"{r['MACs (M)']:>9} {r['Latency (ms)']:>9}"
        )
    lines.append("")
    lines.append("[CLINICAL-VAL] All metrics per AAMI EC57 §3.1.")
    lines.append("No IoU/Dice used for signal segmentation evaluation.")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    log.info(f"[CLINICAL-VAL] Summary saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# § 8 — CLI Entry Point
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[CLINICAL-VAL] Phase 2 — SOTA Benchmark [Sup #3, #6]"
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to AtrionNet best weights (.pth). Auto-resolved if not set."
    )
    parser.add_argument(
        "--tolerance", choices=["50ms", "100ms"], default="50ms",
        help="AAMI EC57 tolerance window (default: 50ms)"
    )
    args = parser.parse_args()

    tol_label  = f"±{args.tolerance}"
    tol_smp    = TIGHT if args.tolerance == "50ms" else LOOSE

    run_benchmark(
        weights_atrionnet=args.weights,
        tolerance_label=tol_label,
        tolerance_smp=tol_smp,
    )
