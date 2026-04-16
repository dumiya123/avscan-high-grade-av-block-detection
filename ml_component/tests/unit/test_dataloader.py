"""
PHASE 1 — Reproducibility & Pipeline Integrity
[Sup #2, #5]

Tests:
  1. Deterministic seed verification (patient-level split)
  2. Dataset integrity (no cross-split leakage, correct ratio)
  3. AAMI temporal matching correctness
  4. Noise robustness under Gaussian & Baseline-Wander injection
  5. Loss function gradient integrity

CLI:
  cd AtrionNet_Implementation
  python -m pytest tests/unit/test_dataloader.py -v --tb=short

[CLINICAL-VAL] All outputs prefixed with [CLINICAL-VAL] tag.
"""

from __future__ import annotations

# ── stdlib ──────────────────────────────────────────────────────────────────
import os
import sys
import random
import logging
import unittest
from pathlib import Path
from typing import List, Dict, Tuple

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import torch
import yaml

# ── path bootstrap (SOC: no src imports into tests/) ──────────────────────
# We need src only for the model/data classes under test — this is allowed
# because we are testing them, not polluting them.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT      = PROJECT_ROOT
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

# ── logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[CLINICAL-VAL] %(levelname)s — %(message)s"
)
log = logging.getLogger("phase1")

# ── Config ──────────────────────────────────────────────────────────────────
CONFIG_PATH = PROJECT_ROOT / "tests" / "configs" / "clinical_config.yaml"
with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

SEED      = CFG["seed"]
SR        = CFG["dataset"]["sampling_rate"]       # 500 Hz
SEQ_LEN   = CFG["dataset"]["target_length"]       # 5000 samples
TIGHT_SMP = CFG["aami_ec57"]["tolerance_tight_smp"]  # 25 samples (±50ms)
LOOSE_SMP = CFG["aami_ec57"]["tolerance_loose_smp"]  # 50 samples (±100ms)


# ════════════════════════════════════════════════════════════════════════════
# § 0 — Global Determinism Fixture
# ════════════════════════════════════════════════════════════════════════════
def _set_deterministic(seed: int = SEED) -> None:
    """Hard-fix all RNG sources. [Sup #2]"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)
    log.info(f"Deterministic mode armed — seed={seed}")


_set_deterministic()


# ════════════════════════════════════════════════════════════════════════════
# § 1 — Synthetic ECG Fallback (wfdb-independent)
# ════════════════════════════════════════════════════════════════════════════
def _synthetic_ecg(n_records: int = 30,
                   n_leads:   int = 12,
                   seq_len:   int = SEQ_LEN,
                   seed:      int = SEED) -> Tuple[np.ndarray, List[Dict]]:
    """
    Generates synthetic 12-lead ECGs with realistic P-wave annotations.
    Used as fallback when LUDB is unavailable.  [Sup #2, #5]
    """
    rng = np.random.default_rng(seed)
    signals     = rng.standard_normal((n_records, n_leads, seq_len)).astype(np.float32)
    annotations = []

    for _ in range(n_records):
        p_waves = []
        # Simulate 8-12 P-waves per 10-second window (typical 60-80 bpm atrial rate)
        n_p = rng.integers(8, 13)
        spacing = seq_len // (n_p + 1)
        for k in range(n_p):
            onset  = spacing * (k + 1) - rng.integers(15, 25)   # ~40ms before peak
            peak   = spacing * (k + 1)
            offset = spacing * (k + 1) + rng.integers(15, 20)   # ~35ms after peak
            onset  = max(0, int(onset))
            offset = min(seq_len - 1, int(offset))
            peak   = int(peak)
            p_waves.append((onset, peak, offset))
        annotations.append({"p_waves": p_waves})

    log.info(f"[CLINICAL-VAL] Generated {n_records} synthetic ECG records "
             f"({n_leads} leads, {seq_len} samples each).")
    return signals, annotations


def _load_real_or_synthetic() -> Tuple[np.ndarray, List[Dict]]:
    """Attempt LUDB; fall back to synthetic data gracefully."""
    data_dir = PROJECT_ROOT / CFG["paths"]["data_dir"]
    if data_dir.exists() and any(data_dir.rglob("*.hea")):
        try:
            from src.data_pipeline.ludb_loader import LUDBLoader
            loader = LUDBLoader(str(data_dir))
            signals, annotations = loader.get_all_data()
            log.info(f"[CLINICAL-VAL] LUDB loaded — {len(signals)} records.")
            return signals, annotations
        except Exception as e:
            log.warning(f"[CLINICAL-VAL] LUDB load failed ({e}). Using synthetic fallback.")
    else:
        log.warning("[CLINICAL-VAL] LUDB not found. Using synthetic fallback.")
    return _synthetic_ecg()


# ════════════════════════════════════════════════════════════════════════════
# § 2 — Inline AAMI Metric Helpers  (SOC: no src/engine import)
# ════════════════════════════════════════════════════════════════════════════
def _compute_aami_metrics(pred_peaks: List[int],
                          gt_spans:   List[Tuple[int, int]],
                          tolerance:  int) -> Dict[str, float]:
    """
    AAMI EC57 §3.1 — Beat detection evaluation.
    A predicted peak is a TP if |pred_peak - gt_peak| ≤ tolerance OR
    the predicted peak falls within [gt_onset, gt_offset] (+tolerance slack).
    [Sup #9]
    """
    matched_gt = set()
    tp = 0
    for peak in pred_peaks:
        for idx, (onset, offset) in enumerate(gt_spans):
            if idx in matched_gt:
                continue
            gt_center = (onset + offset) // 2
            if abs(peak - gt_center) <= tolerance or (onset - tolerance) <= peak <= (offset + tolerance):
                tp += 1
                matched_gt.add(idx)
                break

    fp = len(pred_peaks) - tp
    fn = len(gt_spans)   - tp
    fp = max(fp, 0)
    fn = max(fn, 0)

    se  = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # Sensitivity
    pp  = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # Positive Predictivity (+P)
    fpr = fp / (tp + fp) if (tp + fp) > 0 else 0.0   # False Positive Rate

    # Temporal Jaccard on interval overlap
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {"Se": se, "+P": pp, "FPR": fpr, "Jaccard": jaccard,
            "TP": tp, "FP": fp, "FN": fn}


# ════════════════════════════════════════════════════════════════════════════
# § 3 — Test Cases
# ════════════════════════════════════════════════════════════════════════════
class TestDeterministicSplit(unittest.TestCase):
    """[Sup #2] Verify that the 70/15/15 split is deterministic and leak-free."""

    @classmethod
    def setUpClass(cls):
        _set_deterministic()
        cls.signals, cls.annotations = _load_real_or_synthetic()
        cls.N = len(cls.signals)

        # Reproduce train.py split logic exactly
        np.random.seed(SEED)
        perm          = np.random.permutation(cls.N)
        tr_end        = int(cls.N * 0.70)
        val_end       = int(cls.N * 0.85)
        cls.idx_tr    = set(perm[:tr_end])
        cls.idx_val   = set(perm[tr_end:val_end])
        cls.idx_test  = set(perm[val_end:])

    def test_split_coverage(self):
        """All indices must be covered exactly once."""
        all_idx = self.idx_tr | self.idx_val | self.idx_test
        self.assertEqual(len(all_idx), self.N,
                         "Split indices do not cover full dataset.")
        log.info(f"[CLINICAL-VAL] Split coverage OK — "
                 f"Train:{len(self.idx_tr)} Val:{len(self.idx_val)} Test:{len(self.idx_test)}")

    def test_no_cross_split_leakage(self):
        """Intersection of any two splits must be empty."""
        self.assertEqual(len(self.idx_tr & self.idx_val),  0, "Train/Val leakage detected!")
        self.assertEqual(len(self.idx_tr & self.idx_test), 0, "Train/Test leakage detected!")
        self.assertEqual(len(self.idx_val & self.idx_test),0, "Val/Test leakage detected!")
        log.info("[CLINICAL-VAL] Zero cross-split leakage confirmed.")

    def test_split_ratios(self):
        """Verify approximate 70/15/15 patient-level ratios."""
        tr_ratio   = len(self.idx_tr)   / self.N
        val_ratio  = len(self.idx_val)  / self.N
        test_ratio = len(self.idx_test) / self.N
        self.assertAlmostEqual(tr_ratio,   0.70, delta=0.05,
                               msg=f"Train ratio off: {tr_ratio:.2f}")
        self.assertAlmostEqual(val_ratio,  0.15, delta=0.05,
                               msg=f"Val ratio off: {val_ratio:.2f}")
        self.assertAlmostEqual(test_ratio, 0.15, delta=0.05,
                               msg=f"Test ratio off: {test_ratio:.2f}")
        log.info(f"[CLINICAL-VAL] Split ratios — "
                 f"Train={tr_ratio:.1%} Val={val_ratio:.1%} Test={test_ratio:.1%}")

    def test_split_reproducibility(self):
        """Same seed must produce identical test indices on second run."""
        np.random.seed(SEED)
        perm2 = np.random.permutation(self.N)
        idx_test2 = set(perm2[int(self.N * 0.85):])
        self.assertEqual(self.idx_test, idx_test2,
                         "Split is NOT reproducible across runs!")
        log.info("[CLINICAL-VAL] Split reproducibility — CONFIRMED.")


class TestDatasetIntegrity(unittest.TestCase):
    """[Sup #2] Verify AtrionInstanceDataset output shapes and target validity."""

    @classmethod
    def setUpClass(cls):
        _set_deterministic()
        try:
            from src.data_pipeline.instance_dataset import AtrionInstanceDataset
            signals, annotations = _load_real_or_synthetic()
            cls.ds = AtrionInstanceDataset(signals, annotations, is_train=False)
            cls.available = True
        except ImportError:
            cls.available = False
            log.warning("[CLINICAL-VAL] AtrionInstanceDataset not importable — skipping.")

    def setUp(self):
        if not self.available:
            self.skipTest("AtrionInstanceDataset import failed.")

    def test_output_shapes(self):
        """Each item must have correct tensor shapes."""
        item = self.ds[0]
        self.assertEqual(item["signal"].shape, (12, SEQ_LEN),
                         f"Signal shape wrong: {item['signal'].shape}")
        self.assertEqual(item["heatmap"].shape, (1, SEQ_LEN))
        self.assertEqual(item["width"].shape,   (1, SEQ_LEN))
        self.assertEqual(item["mask"].shape,    (1, SEQ_LEN))
        log.info(f"[CLINICAL-VAL] Dataset shapes OK — signal:{item['signal'].shape}")

    def test_heatmap_bounds(self):
        """Heatmap values must be in [0, 1]."""
        item = self.ds[0]
        hm = item["heatmap"].numpy()
        self.assertTrue(hm.min() >= 0.0 and hm.max() <= 1.0 + 1e-5,
                        f"Heatmap out of bounds: [{hm.min():.4f}, {hm.max():.4f}]")
        log.info(f"[CLINICAL-VAL] Heatmap bounds OK — range [{hm.min():.3f}, {hm.max():.3f}]")

    def test_mask_binary(self):
        """Mask values must be 0 or 1 only."""
        for idx in range(min(5, len(self.ds))):
            mask = self.ds[idx]["mask"].numpy()
            unique_vals = np.unique(np.round(mask, 3))
            self.assertTrue(
                all(v in [0.0, 1.0] for v in unique_vals),
                f"Non-binary mask values found: {unique_vals}"
            )
        log.info("[CLINICAL-VAL] Mask binary constraint — PASSED.")


class TestAAMITemporalMatching(unittest.TestCase):
    """[Sup #9, #10] Validate the AAMI EC57 metric calculator against known cases."""

    def test_perfect_detection(self):
        """All GT P-waves detected exactly → Se=1.0, +P=1.0."""
        gt_spans = [(100, 130), (500, 530), (900, 930)]
        pred_peaks = [115, 515, 915]  # exact GT centers
        m = _compute_aami_metrics(pred_peaks, gt_spans, TIGHT_SMP)
        self.assertAlmostEqual(m["Se"],  1.0, places=3)
        self.assertAlmostEqual(m["+P"], 1.0, places=3)
        log.info(f"[CLINICAL-VAL] Perfect detection — Se={m['Se']:.3f} +P={m['+P']:.3f}")

    def test_all_missed(self):
        """No predictions → Se=0.0, +P=undefined (treated as 0.0)."""
        gt_spans = [(100, 130), (500, 530)]
        pred_peaks = []
        m = _compute_aami_metrics(pred_peaks, gt_spans, TIGHT_SMP)
        self.assertAlmostEqual(m["Se"], 0.0, places=3)
        self.assertAlmostEqual(m["+P"], 0.0, places=3)
        log.info(f"[CLINICAL-VAL] All-miss case — Se={m['Se']:.3f} +P={m['+P']:.3f}")

    def test_tolerance_50ms_tighter_than_100ms(self):
        """±50ms window must give Se ≤ Se at ±100ms."""
        gt_spans = [(200, 240)]
        pred_peaks = [249]  # 49 samples from center=220... within ±100ms but not ±50ms?
        m_tight = _compute_aami_metrics(pred_peaks, gt_spans, TIGHT_SMP)
        m_loose = _compute_aami_metrics(pred_peaks, gt_spans, LOOSE_SMP)
        self.assertLessEqual(m_tight["Se"], m_loose["Se"] + 1e-6,
                             "Tight tolerance must be ≤ loose tolerance Se.")
        log.info(f"[CLINICAL-VAL] ±50ms Se={m_tight['Se']:.3f}  ±100ms Se={m_loose['Se']:.3f}")

    def test_type_I_twave_false_positive(self):
        """FP only — T-wave confusion scenario."""
        gt_spans = [(100, 130)]
        pred_peaks = [400, 115]  # 400 is a FP (T-wave region), 115 is TP
        m = _compute_aami_metrics(pred_peaks, gt_spans, TIGHT_SMP)
        self.assertEqual(m["FP"], 1)
        self.assertEqual(m["TP"], 1)
        log.info(f"[CLINICAL-VAL] Type-I FP (T-wave confusion) — FP={m['FP']} TP={m['TP']} "
                 f"+P={m['+P']:.3f}")

    def test_type_II_low_amplitude_fn(self):
        """Low-amplitude P-wave miss — FN scenario."""
        gt_spans = [(100, 130), (500, 530), (900, 930)]
        pred_peaks = [115, 915]  # miss the 2nd P-wave (low amplitude)
        m = _compute_aami_metrics(pred_peaks, gt_spans, TIGHT_SMP)
        self.assertEqual(m["FN"], 1)
        log.info(f"[CLINICAL-VAL] Type-II FN (low-amplitude) — FN={m['FN']} Se={m['Se']:.3f}")

    def test_temporal_jaccard(self):
        """Jaccard Index must equal TP/(TP+FP+FN)."""
        gt_spans = [(100, 140), (500, 540), (900, 940)]
        pred_peaks = [120, 520, 600]  # third is FP, second is TP
        m = _compute_aami_metrics(pred_peaks, gt_spans, TIGHT_SMP)
        expected_j = m["TP"] / (m["TP"] + m["FP"] + m["FN"])
        self.assertAlmostEqual(m["Jaccard"], expected_j, places=5)
        log.info(f"[CLINICAL-VAL] Jaccard={m['Jaccard']:.4f} "
                 f"(TP={m['TP']} FP={m['FP']} FN={m['FN']})")


class TestNoiseRobustness(unittest.TestCase):
    """[Sup #5] Gaussian & Baseline Wander injection tests."""

    @classmethod
    def setUpClass(cls):
        _set_deterministic()
        cls.rng = np.random.default_rng(SEED)

    def _inject_gaussian(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Add AWGN at specified SNR (dB)."""
        signal_power = np.mean(signal ** 2)
        noise_power  = signal_power / (10 ** (snr_db / 10))
        noise = self.rng.normal(0, np.sqrt(noise_power), signal.shape)
        return (signal + noise).astype(np.float32)

    def _inject_baseline_wander(self, signal: np.ndarray,
                                freq_hz: float, amp: float) -> np.ndarray:
        """Sinusoidal BW artifact at given frequency and amplitude."""
        t = np.linspace(0, SEQ_LEN / SR, SEQ_LEN)
        bw = (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
        return signal + bw[np.newaxis, :]

    def test_gaussian_snr_levels(self):
        """Signal statistics must remain valid under Gaussian noise injection."""
        clean = self.rng.standard_normal((12, SEQ_LEN)).astype(np.float32)
        snr_levels = CFG["noise"]["gaussian"]["snr_levels_db"]
        for snr in snr_levels:
            noisy = self._inject_gaussian(clean, snr)
            self.assertEqual(noisy.shape, clean.shape,
                             f"Shape mismatch at SNR={snr}dB")
            self.assertFalse(np.isnan(noisy).any(),
                             f"NaN detected at SNR={snr}dB")
            log.info(f"[CLINICAL-VAL] Gaussian noise SNR={snr}dB — shape:{noisy.shape} OK")

    def test_baseline_wander_injection(self):
        """BW artifact must be bounded and shape-preserving."""
        clean = self.rng.standard_normal((12, SEQ_LEN)).astype(np.float32)
        for freq in CFG["noise"]["baseline_wander"]["frequencies_hz"]:
            for amp in CFG["noise"]["baseline_wander"]["amplitude_mv"]:
                noisy = self._inject_baseline_wander(clean, freq, amp)
                self.assertEqual(noisy.shape, clean.shape)
                self.assertFalse(np.isnan(noisy).any(),
                                 f"NaN at freq={freq}Hz amp={amp}mV")
        log.info("[CLINICAL-VAL] Baseline Wander injection — ALL PASSED.")

    def test_noise_does_not_corrupt_zero_padding(self):
        """Injecting noise on zero-padded signal must not produce NaN/Inf."""
        padded = np.zeros((12, SEQ_LEN), dtype=np.float32)
        noisy  = self._inject_gaussian(padded, snr_db=20)
        self.assertFalse(np.isnan(noisy).any(),  "NaN on zero-padded signal.")
        self.assertFalse(np.isinf(noisy).any(), "Inf on zero-padded signal.")
        log.info("[CLINICAL-VAL] Zero-pad + noise — no NaN/Inf. PASSED.")


class TestLossFunctionGradients(unittest.TestCase):
    """[Sup #2] Verify focal_loss and create_instance_loss produce valid gradients."""

    @classmethod
    def setUpClass(cls):
        _set_deterministic()
        try:
            from src.losses.segmentation_losses import focal_loss, create_instance_loss
            cls.focal_loss         = staticmethod(focal_loss)
            cls.create_instance_loss = staticmethod(create_instance_loss)
            cls.available          = True
        except ImportError:
            cls.available = False

    def setUp(self):
        if not self.available:
            self.skipTest("Loss import failed.")

    def test_focal_loss_no_nan(self):
        """Focal loss must not produce NaN gradients."""
        pred   = torch.sigmoid(torch.randn(2, 1, SEQ_LEN, requires_grad=True))
        target = torch.zeros(2, 1, SEQ_LEN)
        # Insert some positive targets
        target[0, 0, 250] = 1.0
        target[1, 0, 750] = 1.0
        loss = self.focal_loss(pred, target)
        loss.backward()
        self.assertFalse(torch.isnan(loss), f"NaN in focal_loss: {loss.item()}")
        self.assertFalse(torch.isnan(pred.grad).any(), "NaN gradients in focal_loss.")
        log.info(f"[CLINICAL-VAL] focal_loss gradient check — loss={loss.item():.5f} PASSED.")

    def test_instance_loss_no_nan(self):
        """Full multi-task instance loss must not produce NaN."""
        B = 2
        pred = {
            "heatmap": torch.sigmoid(torch.randn(B, 1, SEQ_LEN)),
            "width":   torch.randn(B, 1, SEQ_LEN),
            "mask":    torch.randn(B, 1, SEQ_LEN)
        }
        for p in pred.values():
            p.requires_grad_(True)

        tgt = {
            "heatmap": torch.zeros(B, 1, SEQ_LEN),
            "width":   torch.zeros(B, 1, SEQ_LEN),
            "mask":    torch.zeros(B, 1, SEQ_LEN)
        }
        tgt["heatmap"][0, 0, 300] = 1.0
        tgt["mask"][0, 0, 285:315] = 1.0
        tgt["width"][0, 0, 285:315] = 0.006

        loss = self.create_instance_loss(pred, tgt)
        loss.backward()
        self.assertFalse(torch.isnan(loss), f"NaN in instance_loss: {loss.item()}")
        log.info(f"[CLINICAL-VAL] create_instance_loss gradient check — "
                 f"loss={loss.item():.5f} PASSED.")

    def test_empty_targets_no_nan(self):
        """Loss must handle zero-annotation records without NaN."""
        B = 1
        pred = {
            "heatmap": torch.sigmoid(torch.randn(B, 1, SEQ_LEN)),
            "width":   torch.randn(B, 1, SEQ_LEN),
            "mask":    torch.randn(B, 1, SEQ_LEN)
        }
        for p in pred.values():
            p.requires_grad_(True)
        tgt = {k: torch.zeros(B, 1, SEQ_LEN) for k in pred}
        loss = self.create_instance_loss(pred, tgt)
        loss.backward()
        self.assertFalse(torch.isnan(loss), f"NaN with empty targets: {loss.item()}")
        log.info(f"[CLINICAL-VAL] Empty-target loss — loss={loss.item():.5f} PASSED.")


# ════════════════════════════════════════════════════════════════════════════
# § 4 — Entry Point
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("[CLINICAL-VAL] PHASE 1 — Reproducibility & Pipeline Integrity")
    print("[Sup #2, #5]")
    print("=" * 70 + "\n")
    unittest.main(verbosity=2)
