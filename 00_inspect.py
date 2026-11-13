"""
00_inspect.py
Raw structural inspection of both sheets BEFORE any cleaning.
Reports: shape, column names, dtypes, missingness, and whether the file
matches what the README/paper claims.
"""
import pandas as pd
from config import DATA_FILE, SHEETS, TAB_DIR

def inspect_sheet(name, sheet):
    print(f"\n{'='*70}\nSHEET: {name}  ({sheet})\n{'='*70}")
    xls = pd.ExcelFile(DATA_FILE)
    raw = pd.read_excel(xls, sheet_name=sheet, header=None)
    print(f"Raw (no header) shape: {raw.shape}")
    print("\nFirst 6 rows, first 8 cols (header layout):")
    print(raw.iloc[:6, :8].to_string())

    # Two-level header: row index 2 and 3 (0-based) hold the real names.
    # We combine them into a single flat header.
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

    # Drop fully-empty trailing rows / cols (Excel padding)
    data = data.dropna(how="all")
    data = data.loc[:, ~data.columns.duplicated()]

    print(f"\nParsed shape: {data.shape}")
    print("\nColumns:")
    for i, c in enumerate(data.columns):
        print(f"  [{i:02d}] {c}")

    miss = data.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    print(f"\nMissing values (top 20):\n{miss.head(20).to_string()}")

    data.to_csv(TAB_DIR / f"00_raw_{name}.csv", index=False)
    return data


if __name__ == "__main__":
    for k, s in SHEETS.items():
        inspect_sheet(k, s)