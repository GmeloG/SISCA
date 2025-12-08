from __future__ import annotations

import numpy as np
import pandas as pd


def add_basic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    return df


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return_close"] = df["Close"].pct_change()
    df["log_return_close"] = np.log(df["Close"]).diff()
    return df


def add_sma_features(
    df: pd.DataFrame,
    time_frame: tuple[int, ...] = (10, 20, 50, 200),
) -> pd.DataFrame:
    df = df.copy()
    for days in time_frame:
        df[f"SMA_{days}"] = df["Close"].rolling(window=days, min_periods=days).mean()
    return df


def add_ema_features(
    df: pd.DataFrame,
    time_frame: tuple[int, ...] = (10, 20, 50, 200),
) -> pd.DataFrame:
    df = df.copy()
    for days in time_frame:
        df[f"EMA_{days}"] = df["Close"].ewm(span=days, adjust=False).mean()
    return df


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(to_replace=0, method="bfill")
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_rsi_feature(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    df[f"RSI_{window}"] = compute_rsi(df["Close"], window=window)
    return df


def add_atr_feature(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[f"ATR_{window}"] = true_range.rolling(window=window, min_periods=window).mean()
    return df


def add_volatility_feature(
    df: pd.DataFrame,
    window: int = 20,
    annualize: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    if "return_close" not in df.columns:
        df["return_close"] = df["Close"].pct_change()

    rolling_std = df["return_close"].rolling(window=window, min_periods=window).std()
    if annualize:
        df[f"volatility_{window}"] = rolling_std * np.sqrt(252)
    else:
        df[f"volatility_{window}"] = rolling_std
    return df


def add_volume_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (10, 20),
) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"Volume_SMA_{w}"] = df["Volume"].rolling(window=w, min_periods=w).mean()
    long_window = max(windows)
    df[f"Volume_norm_{long_window}"] = df["Volume"] / df[f"Volume_SMA_{long_window}"]
    return df


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    #import all features 
    df_feat = add_basic_time_features(df)
    df_feat = add_return_features(df_feat)
    df_feat = add_sma_features(df_feat)
    df_feat = add_ema_features(df_feat)
    df_feat = add_rsi_feature(df_feat, window=14)
    df_feat = add_atr_feature(df_feat, window=14)
    df_feat = add_volatility_feature(df_feat, window=20, annualize=True)
    df_feat = add_volume_features(df_feat)
    return df_feat
