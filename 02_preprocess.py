"""
02_preprocess.py
Reconstruct Qe = (C0 - Ce) * V / m and prepare the clean parquet.

The Ce column is matched ONLY by exact name (Ce, Ce/Ct, Ce/C, Ce (mg/L)).
Never by substring — prevents 'Ce' matching 'Cellulose (%)'.
"""
import numpy as np
import pandas as pd
from config import SHEETS, DATA_FILE, TAB_DIR, MIN_GROUP_SIZE

LEAKY_PATTERNS = [
    "qe", "removal", "ce/ct", "ce", "ct", "qt",
    "removal rate", "output result",
]


def is_leaky(col):
    c = col.lower()
    return any(p in c for p in LEAKY_PATTERNS)


def is_group_col(col):
    return "biomass charcoal name" in col.lower()


def _exact(df, *names):
    """
    Return the column whose NAME equals one of `names` (case-insensitive,
    stripped). No substring, no tail matching. If no exact hit, return None.
    """
    targets = {n.strip().lower() for n in names}
    for c in df.columns:
        if c.strip().lower() in targets:
            return c
    return None


def recompute_qe(df, name):
    """Fill missing Qe from raw inputs; only NaN cells are touched."""
    def sub(*keys):
        for c in df.columns:
            lc = c.lower()
            if all(k in lc for k in keys):
                return c
        return None

    C0 = sub("initial", "concentration")
    V  = sub("liquid", "volume")
    m  = sub("biomass", "addition")
    # EXACT match only - never substring
    Ce = _exact(df, "ce", "ce/c", "ce/ct", "ce (mg/l)")

    print(f"[{name}] recompute columns -> "
          f"C0={C0}  V={V}  m={m}  Ce={Ce}")

    if not all([C0, V, m, Ce]):
        print(f"[{name}] cannot recompute Qe - a raw column is missing")
        return df

    for c in [C0, V, m, Ce]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        recomputed = ((df[C0] - df[Ce]) * df[V]) / df[m]

    mask = df["Qe"].isna() & recomputed.notna() & (recomputed > 0)
    n_fill = int(mask.sum())
    df.loc[mask, "Qe"] = recomputed[mask]
    print(f"[{name}] recomputed {n_fill} missing Qe values")
    return df


def clean_one(name, sheet):
    raw = pd.read_excel(DATA_FILE, sheet_name=sheet, header=None)
    top = raw.iloc[2].ffill()
    bottom = raw.iloc[3]
    combined = []
    for t, b in zip(top, bottom):
        t = "" if pd.isna(t) else str(t).strip()
        b = "" if pd.isna(b) else str(b).strip()
        combined.append(f"{t}__{b}" if t and b and t != b else (b or t))
    df = raw.iloc[4:].copy()
    df.columns = combined
    df = df.reset_index(drop=True).dropna(how="all")
    df = df.loc[:, ~df.columns.duplicated()]

    tgt = next(c for c in df.columns
               if c.upper().startswith("QE") or "qe" in c.lower())
    grp = next(c for c in df.columns
               if "biomass charcoal name" in c.lower())

    df[tgt] = pd.to_numeric(df[tgt], errors="coerce")
    df = recompute_qe(df, name)
    df = df.dropna(subset=[tgt]).reset_index(drop=True)

    grp_str = df[grp].astype(str).str.strip()
    bad = grp_str.str.lower().isin(["nan", "none", ""])
    n_bad = int(bad.sum())
    if n_bad:
        print(f"[{name}] dropping {n_bad} rows with no adsorbent identity")
        df = df[~bad].reset_index(drop=True)
        grp_str = df[grp].astype(str).str.strip()

    counts = grp_str.value_counts()
    tiny = counts[counts < MIN_GROUP_SIZE].index.tolist()
    if tiny:
        print(f"[{name}] dropping groups smaller than {MIN_GROUP_SIZE}: {tiny}")
        df = df[~grp_str.isin(tiny)].reset_index(drop=True)
        grp_str = df[grp].astype(str).str.strip()

    print(f"[{name}] retained {len(df)} records with a valid target")

    feat_cols = [c for c in df.columns
                 if c != tgt and not is_leaky(c) and not is_group_col(c)]
    X = df[feat_cols].copy()
    y = df[tgt].astype(float).copy()
    groups = grp_str

    for c in X.columns:
        coerced = pd.to_numeric(X[c], errors="coerce")
        nonnull = X[c].notna().sum()
        if nonnull > 0 and coerced.notna().sum() / nonnull >= 0.70:
            X[c] = coerced
        else:
            X[c] = X[c].astype("object")

    missing_frac = X.isna().mean()
    to_drop = missing_frac[missing_frac > 0.95].index.tolist()
    if to_drop:
        print(f"[{name}] dropping >95%-missing cols: {to_drop}")
        X = X.drop(columns=to_drop)

    out = X.copy()
    out["__target__"] = y.values
    out["__group__"] = groups.values
    out.to_parquet(TAB_DIR / f"02_clean_{name}.parquet")
    print(f"[{name}] clean shape={out.shape}  features={X.shape[1]}  "
          f"groups={groups.nunique()}  "
          f"target range=[{y.min():.3f}, {y.max():.3f}]")
    return out


if __name__ == "__main__":
    for k, s in SHEETS.items():
        clean_one(k, s)