"""
06_pdp.py
1D partial dependence plots for the top NON-definitional features.
These are the plots that are safe to interpret chemically.
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

from config import TAB_DIR, MODEL_DIR, FIG_DIR, SHEETS


def run(name, top_k=4):
    pipe = joblib.load(MODEL_DIR / f"best_{name}.joblib")
    df = pd.read_parquet(TAB_DIR / f"02_clean_{name}.parquet")
    X = df.drop(columns=["__target__", "__group__"])

    # pick top non-definitional numeric features
    imp = pd.read_csv(TAB_DIR / f"05_shap_importance_{name}.csv")
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # map transformed names back to original (OHE names won't match; skip)
    candidates = []
    for _, row in imp.iterrows():
        f = row["feature"]
        if row["definitional"]:
            continue
        if f in num_cols:
            candidates.append(f)
        if len(candidates) >= top_k:
            break

    if not candidates:
        print(f"[{name}] no non-definitional numeric features available")
        return

    fig, ax = plt.subplots(1, len(candidates), figsize=(4*len(candidates), 4),
                           sharey=True)
    if len(candidates) == 1:
        ax = [ax]
    PartialDependenceDisplay.from_estimator(
        pipe, X, features=candidates, ax=ax, kind="average",
        grid_resolution=30,
    )
    for a, c in zip(ax, candidates):
        a.set_title(c, fontsize=10)
    plt.suptitle(f"{name}: Partial dependence (top non-definitional features)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"06_pdp_{name}.png", dpi=150)
    plt.close()
    print(f"[{name}] PDP saved for features: {candidates}")


if __name__ == "__main__":
    for k in SHEETS:
        run(k)