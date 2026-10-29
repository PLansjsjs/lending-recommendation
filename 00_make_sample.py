"""Step 0 - read the big raw Lending Club file and save a 50,000-row sample.

Usage:  python 00_make_sample.py                 (auto-picks the largest CSV in data/)
        python 00_make_sample.py data\\loan.csv    (or name the file yourself)
"""
import sys
from pathlib import Path

import pandas as pd
from common import DATA_DIR, OPTIONAL_RAW, RAW_COLUMNS, SAMPLE_FILE, SAMPLE_SIZE, SEED

FINISHED = ["Fully Paid", "Charged Off"]
SKIP_NAMES = {SAMPLE_FILE.name, "loan_clean.csv"}


def find_raw_file():
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if not p.exists():
            raise SystemExit(f"File not found: {p}")
        return p
    files = [p for p in DATA_DIR.iterdir()
             if p.is_file() and p.name not in SKIP_NAMES
             and (p.name.endswith(".csv") or p.name.endswith(".csv.gz"))]
    if not files:
        raise SystemExit("No raw CSV found in the data/ folder. Download the Lending Club "
                         "file from Kaggle and put it there (see README.md).")
    return max(files, key=lambda p: p.stat().st_size)


def read_filtered(path, frac, skiprows):
    parts = []
    for chunk in pd.read_csv(path, usecols=lambda c: c in RAW_COLUMNS, chunksize=200_000,
                             low_memory=False, skiprows=skiprows):
        chunk = chunk[chunk["loan_status"].isin(FINISHED)]
        if frac < 1:
            chunk = chunk.sample(frac=frac, random_state=SEED)
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True)


def main():
    path = find_raw_file()
    print(f"Reading: {path}")
    if "synthetic" in path.name.lower():
        print("*** WARNING: this is the SYNTHETIC demo file, not real Lending Club data. ***")

    # Some older Lending Club files have a note line above the header.
    header = pd.read_csv(path, nrows=0)
    skiprows = None if "loan_status" in header.columns else 1

    df = read_filtered(path, 0.30, skiprows)       # 30% of finished loans per chunk...
    if len(df) < SAMPLE_SIZE:                      # ...unless that is too few rows
        df = read_filtered(path, 1.0, skiprows)
    missing = [c for c in RAW_COLUMNS if c not in df.columns and c not in OPTIONAL_RAW]
    if missing:
        raise SystemExit(f"These columns are missing from the raw file: {missing}\n"
                         "Check the column names in your CSV and edit RAW_COLUMNS in common.py.")

    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=SEED)
    df.to_csv(SAMPLE_FILE, index=False)
    rate = (df["loan_status"] == "Charged Off").mean()
    print(f"Saved {len(df):,} rows to {SAMPLE_FILE.name}  (Charged Off share: {rate:.1%})")


if __name__ == "__main__":
    main()
