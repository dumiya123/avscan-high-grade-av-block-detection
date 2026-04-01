import os

# Base Directory of the Project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Standard Research Paths (For Portability across Local & Colab)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ludb")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "weights")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Ensure research folders always exist
for path in [CHECKPOINT_DIR, REPORTS_DIR, os.path.join(REPORTS_DIR, "plots")]:
    os.makedirs(path, exist_ok=True)
