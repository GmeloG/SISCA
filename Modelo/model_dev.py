from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_feature_dataset() -> pd.DataFrame:
    
    base_dir = Path(__file__).resolve().parents[1]  # .../SISCA
    features_path = base_dir / "data" / "TSLA_features.csv"

    df = pd.read_csv(features_path, parse_dates=["Date"])
    df = df.sort_values("Date")

    df = df.dropna().reset_index(drop=True) # remove NaN (first 200 days)


    return df


def make_features_and_target(df: pd.DataFrame):
    df = df.copy()

    
    df["target_close_next"] = df["Close"].shift(-1) # target: next day price 

    df = df.dropna(subset=["target_close_next"]).reset_index(drop=True) #deleting NaN lines 
    
    drop_cols = ["Date", "target_close_next"] # unused columns 

    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["target_close_next"]

    return X, y, feature_cols


def train_test_split_time_series(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
):
    n = len(X) #total samples number 
    split_idx = int(n * (1 - test_size)) #80% train and 20% test

    #splitting samples according 80-20
    X_train = X.iloc[:split_idx, :].copy() 
    X_test = X.iloc[split_idx:, :].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test


def evaluate_regression_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
) -> None:
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n=== Results {model_name} ===")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


def main() -> None:
    
    df = load_feature_dataset() #features dataset loading 
    print(f"Dataset line final line number: {len(df)}")

    X, y, feature_cols = make_features_and_target(df)
    print(f"Number of features: {len(feature_cols)}")

    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y, test_size=0.2)
    print(f"Percentage of used data for training: {len(X_train)}, Percentage of used data for teste: {len(X_test)}")

    # Modelo 1: Linear Reg 
   
    linreg_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    print("\n Training LinearRegression")
    linreg_pipeline.fit(X_train, y_train)
    y_pred_lin = linreg_pipeline.predict(X_test)
    evaluate_regression_model(y_test, y_pred_lin, model_name="LinearRegression")

    # Modelo 2: RandomForest

    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )

    print("\n Training RandomForest")

    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    evaluate_regression_model(y_test, y_pred_rf, model_name="RandomForest")

    # Predictions vs Real Values 
    comparison = pd.DataFrame(
        {
            "Date": df["Date"].iloc[len(X_train) + 1 : len(df)].reset_index(drop=True),
            "y_true": y_test.values,
            "y_pred_lin": y_pred_lin,
            "y_pred_rf": y_pred_rf,
        }
    )
    print("\nPrediction last 10 lines")
    print(comparison.tail(10))


if __name__ == "__main__":
    main()
