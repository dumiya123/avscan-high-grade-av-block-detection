"""
ATRIONNET CLINICAL TESTING SUITE — Master Orchestrator
[Sup #1 → #12]

Entry point for the full 5-phase clinical validation pipeline.
Run individual phases or the complete suite.

Usage:
  # Full suite (all 5 phases)
  cd AtrionNet_Implementation
  python tests/run_clinical_suite.py --phase all

  # Individual phases
  python tests/run_clinical_suite.py --phase 1
  python tests/run_clinical_suite.py --phase 2 --weights ml_component/outputs/weights/atrion_hybrid_best.pth
  python tests/run_clinical_suite.py --phase 3 --tolerance 100ms
  python tests/run_clinical_suite.py --phase 4 --n_records 5
  python tests/run_clinical_suite.py --phase 5

[CLINICAL-VAL] All outputs prefixed with [CLINICAL-VAL].
[Sup #1 → #12]
"""

from __future__ import annotations

import os
import sys
import time
import logging
import argparse
import subprocess
from pathlib import Path

# ── Path bootstrap ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
ML_ROOT      = PROJECT_ROOT.parent

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[CLINICAL-VAL] %(levelname)s — %(message)s"
)
log = logging.getLogger("orchestrator")

# ── ANSI colours (Windows-safe) ──────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ════════════════════════════════════════════════════════════════════════════
# Phase Registry — all 5 phases now fully registered
# ════════════════════════════════════════════════════════════════════════════
PHASE_REGISTRY = {
    1: {
        "name":   "Reproducibility & Pipeline Integrity",
        "sup":    "[Sup #2, #5]",
        "script": None,   # pytest invocation
    },
    2: {
        "name":   "SOTA Competitive Benchmarking",
        "sup":    "[Sup #3, #6]",
        "script": "tests/benchmark/compare_baselines.py",
    },
    3: {
        "name":   "AAMI EC57 Performance Suite",
        "sup":    "[Sup #9, #10]",
        "script": "tests/performance/calculate_aami_metrics.py",
    },
    4: {
        "name":   "Clinical XAI & Visual Logic",
        "sup":    "[Sup #7, #11]",
        "script": "tests/xai_audit/visualize_logic.py",
    },
    5: {
        "name":   "Thesis Synthesis & Ablation",
        "sup":    "[Sup #8, #12]",
        "script": "tests/reporting/generate_ch8_assets.py",
    },
}


def _banner() -> None:
    w = 74
    print("\n" + "=" * w)
    print(f"{BOLD}  ATRIONNET CLINICAL TESTING SUITE  [v2.0 — All Phases Active]{RESET}")
    print(f"  Explainable Anchor-Free ECG Segmentation | High-Grade AV Block")
    print(f"  AAMI EC57 Compliant | SOC Architecture | seed=42")
    print("=" * w)
    print(f"  {'Ph.':<4}  {'Name':<44}  {'Supervisor':<16}  Status")
    print("-" * w)
    for ph, info in PHASE_REGISTRY.items():
        print(f"  {ph:<4}  {info['name']:<44}  {info['sup']:<16}  {GREEN}[READY]{RESET}")
    print("=" * w + "\n")


# ════════════════════════════════════════════════════════════════════════════
# § Phase Runners
# ════════════════════════════════════════════════════════════════════════════
def _run_subprocess(cmd: list, cwd: str, phase_label: str) -> bool:
    t0     = time.perf_counter()
    result = subprocess.run(cmd, cwd=cwd, capture_output=False, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode == 0:
        log.info(f"{GREEN}[CLINICAL-VAL] {phase_label} PASSED in {elapsed:.1f}s{RESET}")
        return True
    log.error(f"{RED}[CLINICAL-VAL] {phase_label} FAILED "
              f"(exit={result.returncode}) after {elapsed:.1f}s{RESET}")
    return False


def _run_phase1(cwd: str) -> bool:
    log.info("[CLINICAL-VAL] === PHASE 1: Reproducibility & Pipeline Integrity ===")
    try:
        return _run_subprocess(
            [sys.executable, "-m", "pytest",
             "tests/unit/test_dataloader.py", "-v", "--tb=short"],
            cwd=cwd, phase_label="Phase 1"
        )
    except FileNotFoundError:
        log.error("[CLINICAL-VAL] pytest not found — install: pip install pytest")
        return False


def _run_script_phase(ph: int, cwd: str,
                      weights: str = None,
                      tolerance: str = "50ms",
                      n_records: int = 5) -> bool:
    info   = PHASE_REGISTRY[ph]
    script = Path(cwd) / info["script"]
    cmd    = [sys.executable, str(script)]

    # Attach relevant CLI flags per phase
    if ph in [2, 3, 5]:
        cmd += ["--tolerance", tolerance]
    if ph in [2, 3, 4, 5] and weights:
        cmd += ["--weights", weights]
    if ph == 4:
        cmd += ["--n_records", str(n_records)]

    log.info(f"[CLINICAL-VAL] === PHASE {ph}: {info['name']} ===")
    return _run_subprocess(cmd, cwd=cwd, phase_label=f"Phase {ph}")


# ════════════════════════════════════════════════════════════════════════════
# § Main
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    os.system("")  # Enable ANSI escape codes on Windows

    parser = argparse.ArgumentParser(
        description="[CLINICAL-VAL] AtrionNet Clinical Testing Suite — All 5 Phases"
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["all", "1", "2", "3", "4", "5"],
        help="Phase to run. 'all' runs Phases 1→5 sequentially (default: all)."
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to AtrionNet best weights (.pth). Auto-resolved if omitted."
    )
    parser.add_argument(
        "--tolerance", choices=["50ms", "100ms"], default="50ms",
        help="AAMI EC57 tolerance window for Phases 2, 3, 5 (default: 50ms)."
    )
    parser.add_argument(
        "--n_records", type=int, default=5,
        help="Number of records to visualise in Phase 4 XAI audit (default: 5)."
    )
    args = parser.parse_args()

    _banner()
    cwd = str(PROJECT_ROOT.parent)   # AtrionNet_Implementation/

    # ── Determine which phases to run ──────────────────────────────────────
    phases_to_run = (
        list(PHASE_REGISTRY.keys()) if args.phase == "all"
        else [int(args.phase)]
    )

    outcomes: dict[int, bool] = {}
    for ph in phases_to_run:
        if ph == 1:
            outcomes[ph] = _run_phase1(cwd)
        else:
            outcomes[ph] = _run_script_phase(
                ph, cwd,
                weights=args.weights,
                tolerance=args.tolerance,
                n_records=args.n_records
            )
        # In full-suite mode, stop on Phase 1 failure (integrity gate)
        if args.phase == "all" and ph == 1 and not outcomes[ph]:
            log.error("[CLINICAL-VAL] Phase 1 integrity gate FAILED — "
                      "aborting suite to prevent contaminated results.")
            break

    # ── Execution Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("[CLINICAL-VAL] SUITE EXECUTION SUMMARY")
    print("-" * 74)
    for ph, passed in outcomes.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        sup    = PHASE_REGISTRY[ph]["sup"]
        print(f"  Phase {ph}  {PHASE_REGISTRY[ph]['name']:<44}  {sup:<16}  [{status}]")

    not_run = [ph for ph in PHASE_REGISTRY if ph not in outcomes]
    for ph in not_run:
        print(f"  Phase {ph}  {PHASE_REGISTRY[ph]['name']:<44}  "
              f"{PHASE_REGISTRY[ph]['sup']:<16}  [{YELLOW}SKIPPED{RESET}]")

    failed = [ph for ph, ok in outcomes.items() if not ok]
    print("-" * 74)
    if failed:
        print(f"\n{RED}[CLINICAL-VAL] {len(failed)} phase(s) failed: {failed}{RESET}")
        print(f"  → Check logs above. Rerun individual phases with --phase N{RESET}")
        sys.exit(1)
    else:
        passed_count = len(outcomes)
        print(f"\n{GREEN}[CLINICAL-VAL] All {passed_count} executed phase(s) PASSED.{RESET}")
        print(f"  Outputs → {PROJECT_ROOT / 'outputs'}")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
