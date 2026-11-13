"""
07_screening.py
Virtual-screening grid. The point of this script (following the paper's
logic) is to REPORT whether the model discriminates across the grid —
not to claim it can propose optimal conditions.

Grid: pyrolysis temperature x solution pH x initial concentration x
adsorbent mass. All other features are held at training medians/modes.
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from config import TAB_DIR, MODEL_DIR, FIG_DIR, SHEETS, LOG_TARGET


def run(name):
    pipe = joblib.load(MODEL_DIR / f"best_{name}.joblib")
    df = pd.read_parquet(TAB_DIR / f"02_clean_{name}.parquet")
    X = df.drop(columns=["__target__", "__group__"])

    # defaults: median for numeric, mode for object
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]
    base = {}
    for c in numeric_cols:
        base[c] = float(X[c].median())
    for c in cat_cols:
        mode = X[c].mode(dropna=True)
        base[c] = mode.iloc[0] if not mode.empty else ""

    # pick sweepable columns if present
    def find(*keys):
        for c in X.columns:
            lc = c.lower()
            if all(k in lc for k in keys):
                return c
        return None

    col_T = find("pyrolysis", "temperature")
    col_pH = find("solution", "ph") or find("ph")
    col_C0 = find("initial", "concentration")
    col_mass = find("biomass addition") or find("mass") or find("loading")

    grids = {}
    if col_T:
        vals = pd.to_numeric(X[col_T], errors="coerce").dropna()
        lo, hi = vals.min(), vals.max()
        grids[col_T] = np.linspace(lo, hi, 5) if hi > lo else [lo]
    if col_pH:
        grids[col_pH] = [3.0, 5.0, 7.0, 9.0, 11.0]
    if col_C0:
        vals = pd.to_numeric(X[col_C0], errors="coerce").dropna()
        lo, hi = vals.quantile(.1), vals.quantile(.9)
        grids[col_C0] = np.linspace(lo, hi, 4)
    if col_mass:
        vals = pd.to_numeric(X[col_mass], errors="coerce").dropna()
        lo, hi = vals.quantile(.1), vals.quantile(.9)
        grids[col_mass] = np.linspace(lo, hi, 4)

    if not grids:
        print(f"[{name}] no sweepable columns found")
        return

    print(f"[{name}] screening over: {list(grids.keys())}")

    import itertools
    keys = list(grids.keys())
    combos = list(itertools.product(*[grids[k] for k in keys]))
    rows = []
    for combo in combos:
        row = dict(base)
        for k, v in zip(keys, combo):
            row[k] = v
        rows.append(row)
    grid_df = pd.DataFrame(rows)

    # predict
    pred_log = pipe.predict(grid_df)
    pred = np.expm1(pred_log) if LOG_TARGET else pred_log
    grid_df["predicted_Qe"] = pred

    # summary statistics
    print(f"\n[{name}] Screening summary")
    print(f"  n points         = {len(grid_df)}")
    print(f"  min Qe           = {pred.min():.4f} mg/g")
    print(f"  median Qe        = {np.median(pred):.4f} mg/g")
    print(f"  max Qe           = {pred.max():.4f} mg/g")
    print(f"  IQR              = {np.percentile(pred,75)-np.percentile(pred,25):.4f}")
    print(f"  std              = {pred.std():.4f}")

    # <-- the honesty diagnostic: does the model discriminate per axis?
    print(f"\n[{name}] discrimination per sweep axis "
          f"(range of mean predicted Qe across axis values):")
    for k in keys:
        grouped = grid_df.groupby(k)["predicted_Qe"].mean()
        spread = grouped.max() - grouped.min()
        print(f"  {k:35s}  spread = {spread:.4f} mg/g  "
              f"({'DISCRIMINATIVE' if spread > 0.1 else 'FLAT'})")

    grid_df.to_csv(TAB_DIR / f"07_screening_{name}.csv", index=False)

    # figure
    n_axes = len(keys)
    fig, axes = plt.subplots(1, n_axes, figsize=(4*n_axes, 4))
    if n_axes == 1:
        axes = [axes]
    for a, k in zip(axes, keys):
        agg = grid_df.groupby(k)["predicted_Qe"].agg(["mean", "std"]).reset_index()
        a.errorbar(agg[k], agg["mean"], yerr=agg["std"], marker="o", capsize=3)
        a.set_xlabel(k)
        a.set_ylabel("predicted Qe (mg/g)")
        a.set_title(k.split("__")[-1])
    plt.suptitle(f"{name}: virtual screening (does the model discriminate?)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"07_screening_{name}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    for k in SHEETS:
        run(k)