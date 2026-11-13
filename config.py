"""
Shared configuration. All scripts import from here so the pipeline is
reproducible and any single constant can be changed in one place.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "Supplementary Material.xlsx"
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
TAB_DIR = RESULTS / "tables"
MODEL_DIR = RESULTS / "models"

for d in (FIG_DIR, TAB_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- audit
SHEETS = {
    "Pb": "Pb Adsorption Data",
    "Cd": "Cd Adsorption Data",
}

# Data rows start at Excel row 5 (row 1-4 are multi-level headers).
HEADER_ROW_PB = 4      # 0-indexed -> pandas reads row 3 as header for Pb
HEADER_ROW_CD = 4

# ---------------------------------------------------------------- split
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ---------------------------------------------------------------- models
N_RANDOM_SEARCH = 25
TARGET = "Qe"
GROUP_COL = "Biomass charcoal name"
LOG_TARGET = True      # model log(1 + Qe)