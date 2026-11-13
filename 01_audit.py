"""
01_audit.py
The Audit-First step: report ion/groups, target distribution, outliers,
and the definitional-coupling geometry of Qe vs adsorbent mass.
This is a RESULT, not a preparatory step.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import SHEETS, TAB_DIR, FIG_DIR, DATA_FILE, TARGET, GROUP_COL

def load_flat(name, sheet):
    raw = pd.read_excel(DATA_FILE, sheet_name=sheet, header=None)
    top = raw.iloc[2].ffill()
    bottom = raw.iloc[3]
    combined = []
    for t, b in zip(top, bottom):
        t = "" if pd.isna(t) else str(t).strip()
        b = "" if pd.isna(b) else str(b).strip()
        combined.append(f"{t}__{b}" if t and b and t != b else (b or t))
    data = raw.iloc[4:].copy()
    data.columns = combined
    data = data.reset_index(drop=True)
    data = data.dropna(how="all")
    data = data.loc[:, ~data.columns.duplicated()]
    return data


def audit_one(name, sheet):
    df = load_flat(name, sheet)
    print(f"\n{'='*70}\nAUDIT: {name}\n{'='*70}")
    print(f"Records: {len(df)}   Columns: {df.shape[1]}")

    # --- locate target and group columns robustly
    tgt_candidates = [c for c in df.columns if c.upper().startswith("QE") or "qe" in c.lower()]
    grp_candidates = [c for c in df.columns if "biomass charcoal name" in c.lower()]
    tgt = tgt_candidates[0] if tgt_candidates else None
    grp = grp_candidates[0] if grp_candidates else None
    print(f"Target column -> {tgt}\nGroup column  -> {grp}")

    df[tgt] = pd.to_numeric(df[tgt], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=[tgt])
    print(f"Rows dropped for missing target: {n_before - len(df)}")

    # --- target distribution
    s = df[tgt]
    stats = dict(
        count=len(s), mean=s.mean(), std=s.std(), min=s.min(),
        q25=s.quantile(.25), median=s.median(), q75=s.quantile(.75),
        max=s.max(), skew=s.skew(),
    )
    print("\nTarget summary (Qe, mg/g):")
    for k, v in stats.items():
        print(f"  {k:>8}: {v:>10.4f}")

    # --- group composition
    comp = (df[grp].astype(str).str.strip()
            .value_counts(dropna=False)
            .to_frame("records"))
    comp["proportion"] = comp["records"] / comp["records"].sum()
    print("\nGroup composition:")
    print(comp.to_string())
    comp.to_csv(TAB_DIR / f"01_group_composition_{name}.csv")

    # --- per-group target stats
    per = (df.groupby(df[grp].astype(str).str.strip())[tgt]
             .agg(["count", "median", "mean", "max", "std"])
             .sort_values("count", ascending=False))
    print("\nPer-group Qe stats:")
    print(per.to_string())
    per.to_csv(TAB_DIR / f"01_per_group_stats_{name}.csv")

    # --- figures
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.histplot(s, bins=40, ax=axes[0], color="steelblue")
    axes[0].set_title(f"{name}: Qe distribution (raw)")
    axes[0].set_xlabel("Qe (mg/g)")

    sns.boxplot(x=s, ax=axes[1], color="salmon")
    axes[1].set_title(f"{name}: Qe boxplot (outliers visible)")

    log_s = np.log1p(s.clip(lower=0))
    sns.histplot(log_s, bins=40, ax=axes[2], color="seagreen")
    axes[2].set_title(f"{name}: log(1+Qe) distribution")
    axes[2].set_xlabel("log(1+Qe)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"01_target_dist_{name}.png", dpi=150)
    plt.close()

    # --- definitional-coupling diagnostic (the core audit finding)
    # Qe = (C0 - Ce) * V / m  =>  Qe inversely proportional to mass.
    # Show this directly when both columns exist.
    mass_cols = [c for c in df.columns
                 if any(k in c.lower() for k in ["biomass addition", "loading", "mass"])]
    if mass_cols:
        m = pd.to_numeric(df[mass_cols[0]], errors="coerce")
        sub = pd.DataFrame({"mass": m, "Qe": s}).dropna()
        sub = sub[sub["mass"] > 0]
        if len(sub) > 2:
            r = np.corrcoef(np.log1p(sub["mass"]), np.log1p(sub["Qe"]))[0, 1]
            print(f"\nDefinitional coupling: corr(log mass, log Qe) = {r:.3f}")
            plt.figure(figsize=(6, 5))
            plt.scatter(np.log1p(sub["mass"]), np.log1p(sub["Qe"]),
                        s=12, alpha=0.5)
            plt.xlabel("log(1 + adsorbent mass, mg)")
            plt.ylabel("log(1 + Qe, mg/g)")
            plt.title(f"{name}: definitional mass–Qe coupling (r={r:.2f})")
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"01_mass_coupling_{name}.png", dpi=150)
            plt.close()

    return df, tgt, grp


if __name__ == "__main__":
    for k, s in SHEETS.items():
        audit_one(k, s)