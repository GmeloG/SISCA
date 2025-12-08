from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_features_dataframe() -> pd.DataFrame:
    """
    Lê o ficheiro TSLA_features.csv a partir da pasta data.
    """
    base_dir = Path(r"C:\Users\tomas\OneDrive\Documents\GitHub\SISCA/data/TSLA.csv")
    features_path = Path(r"C:\Users\tomas\OneDrive\Documents\GitHub\SISCA/data/TSLA_features.csv")

    df = pd.read_csv(features_path, parse_dates=["Date"])
    df = df.sort_values("Date")
    return df


def plot_price_with_moving_averages(df: pd.DataFrame) -> None:
    
    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Close"], label="Close", linewidth=1.5)

    # some MA 
    for col in ["SMA_20", "SMA_50", "EMA_20", "EMA_50"]:
        if col in df.columns:
            plt.plot(df["Date"], df[col], label=col, linewidth=1)

    plt.title("TSLA - Close Price with MA")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_rsi(df: pd.DataFrame) -> None:
   
    rsi_col = "RSI_14"
   
    plt.figure(figsize=(12, 4))
    plt.plot(df["Date"], df[rsi_col], label=rsi_col, linewidth=1.2)
    plt.axhline(70, color="red", linestyle="--", linewidth=0.8, label="Overbuy (70)")
    plt.axhline(30, color="green", linestyle="--", linewidth=0.8, label="Oversell (30)")

    plt.title("TSLA - RSI (14 days)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_atr_and_volatility(df: pd.DataFrame) -> None:

    plt.figure(figsize=(12, 5))

    if "ATR_14" in df.columns:
        plt.plot(df["Date"], df["ATR_14"], label="ATR_14", linewidth=1.2)

    if "volatility_20" in df.columns:
        plt.plot(df["Date"], df["volatility_20"], label="volatility_20", linewidth=1.2)

    plt.title("TSLA - ATR (14) and Historic Volatility (20 days)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def main() -> None:
    
    print("Plotting")
    df = load_features_dataframe()
    
    plot_price_with_moving_averages(df)
    plot_rsi(df)
    plot_atr_and_volatility(df)

    plt.show()


if __name__ == "__main__":
    main()
