"""Diagnose data availability and status."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def check_data() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    raw_path = base_dir / "data" / "TSLA.csv"
    features_path = base_dir / "data" / "TSLA_features.csv"

    print("Data Status Check")
    print("=" * 60)

    # Check raw data
    print(f"\n1. Raw data: {raw_path}")
    if raw_path.exists():
        try:
            df_raw = pd.read_csv(raw_path)
            
            # Keep only valid columns
            valid_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            cols = [c for c in valid_cols if c in df_raw.columns]
            
            if len(cols) < 6:
                print(f"   ⚠ Incomplete — missing columns. Found: {list(df_raw.columns)[:6]}")
                print(f"   Fix: python Modelo/clean_data.py")
            else:
                df_raw = df_raw[cols]
                print(f"   ✓ Exists | Shape: {df_raw.shape} | Columns: {cols}")
                
                # Try parsing dates
                try:
                    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
                    print(f"   Date range: {df_raw['Date'].min()} to {df_raw['Date'].max()}")
                except Exception as e:
                    print(f"   ⚠ Date parsing error: {str(e)[:50]}")
        except Exception as e:
            print(f"   ✗ Error reading file: {str(e)[:80]}")
            print(f"   Fix: python Modelo/clean_data.py")
    else:
        print(f"   ✗ Missing — please add TSLA.csv to data/ folder")

    # Check features data
    print(f"\n2. Features data: {features_path}")
    if features_path.exists():
        df_feat = pd.read_csv(features_path)
        print(f"   ✓ Exists | Shape: {df_feat.shape} | Columns: {len(df_feat.columns)}")
        
        # Check NaN
        nan_count = df_feat.isna().sum().sum()
        if nan_count > 0:
            print(f"   ⚠ Contains {nan_count} NaN values")
            nan_cols = df_feat.columns[df_feat.isna().any()].tolist()
            print(f"   Columns with NaN: {nan_cols[:5]}{'...' if len(nan_cols) > 5 else ''}")
        else:
            print(f"   ✓ No NaN values")
        
        # Check after dropna
        df_clean = df_feat.dropna()
        print(f"   After dropna(): {len(df_clean)} rows (removed {len(df_feat) - len(df_clean)})")
        
        if len(df_clean) == 0:
            print(f"   ✗ All rows have NaN — cannot train models")
        else:
            print(f"   ✓ Ready for model training")
    else:
        print(f"   ✗ Missing — please run: python Modelo/use_features.py")

    # Next steps
    print(f"\n" + "=" * 60)
    print("Next Steps:")
    if not raw_path.exists():
        print("  1. Add TSLA.csv to data/ folder")
    if not features_path.exists() and raw_path.exists():
        print("  1. python Modelo/use_features.py")
    if features_path.exists() and features_path.stat().st_size > 0:
        df_test = pd.read_csv(features_path)
        if len(df_test.dropna()) > 0:
            print("  2. python Modelo/model_dev.py")
            print("  3. python Modelo/model_comparison.py")
            print("  4. python Modelo/plot_features_TSLA.py")


if __name__ == "__main__":
    check_data()
