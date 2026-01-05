"""Clean and fix TSLA.csv to remove corrupted multi-index columns."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def clean_tsla_csv() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    csv_path = base_dir / "data" / "TSLA.csv"

    print(f"Cleaning {csv_path}...")

    # Read with all columns
    df = pd.read_csv(csv_path)

    # Keep only valid columns
    valid_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Raw_close", "Change_percent", "Avg_vol_20d"]
    cols_to_keep = [c for c in valid_cols if c in df.columns]
    
    print(f"Original columns: {list(df.columns)}")
    print(f"Keeping: {cols_to_keep}")

    df_clean = df[cols_to_keep].copy()

    # Save back
    df_clean.to_csv(csv_path, index=False)
    print(f"✓ Cleaned and saved: {len(df_clean)} rows, {len(cols_to_keep)} columns")


if __name__ == "__main__":
    clean_tsla_csv()
