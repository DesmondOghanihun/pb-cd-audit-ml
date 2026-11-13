"""
03_split.py
Grouped train/test split by adsorbent name, with an explicit overlap check.
The point of this script is not to split well — it is to REPORT the split
honestly (n_groups_train, n_groups_test, overlap).
"""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from config import (TAB_DIR, SHEETS, RANDOM_STATE, TEST_SIZE,
                    GROUP_COL, FIG_DIR)
import matplotlib.pyplot as plt
import numpy as np


def split_one(name):
    df = pd.read_parquet(TAB_DIR / f"02_clean_{name}.parquet")
    y = df["__target__"].values
    groups = df["__group__"].values
    X = df.drop(columns=["__target__", "__group__"])

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(X, y, groups))

    tr_groups = set(groups[tr_idx])
    te_groups = set(groups[te_idx])
    overlap = tr_groups & te_groups

    print(f"\n[{name}] grouped split")
    print(f"  train rows={len(tr_idx):5d}  groups={len(tr_groups)}")
    print(f"  test  rows={len(te_idx):5d}  groups={len(te_groups)}")
    print(f"  group overlap = {len(overlap)}  "
          f"({'PASS' if not overlap else 'LEAK!'})")
    print(f"  test groups: {sorted(te_groups)}")

    # save indices
    np.save(TAB_DIR / f"03_train_idx_{name}.npy", tr_idx)
    np.save(TAB_DIR / f"03_test_idx_{name}.npy", te_idx)

    # figure: group balance
    fig, ax = plt.subplots(figsize=(8, 4))
    tr_counts = pd.Series(groups[tr_idx]).value_counts()
    te_counts = pd.Series(groups[te_idx]).value_counts()
    labels = sorted(set(tr_counts.index) | set(te_counts.index))
    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w/2, [tr_counts.get(l, 0) for l in labels], w, label="train")
    ax.bar(x + w/2, [te_counts.get(l, 0) for l in labels], w, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("records")
    ax.set_title(f"{name}: grouped split by adsorbent")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"03_split_balance_{name}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    for k in SHEETS:
        split_one(k)